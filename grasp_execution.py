"""Grasp geometry and step sequencing, independent of Qt/ROS and robot I/O."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from grasp_trajectory import exceeds, sample as sample_profiled_trajectory


TCP_POSITION_TOLERANCE_M = .015
TCP_ROTATION_TOLERANCE_RAD = np.deg2rad(10.0)
IK_ADJACENT_JUMP_LIMIT_RAD = .18
WRENCH_MAX_AGE_S = .5


def evaluate_wrench(sample, force_limit_n, torque_limit_nm, now=None, policy=None):
    """Return (violation, force_norm, torque_norm) for a wrist sample."""
    now = time.time() if now is None else float(now)
    if not isinstance(sample, dict):
        return '未收到六维力反馈', None, None
    age = now-float(sample.get('stamp') or 0.)
    if not 0 <= age <= WRENCH_MAX_AGE_S:
        return f'六维力反馈过期（{age:.2f}s）', None, None
    state = int(sample.get('force_state', 0) or 0)
    if state != 0:
        return f'六维力传感器状态异常 force_state={state}', None, None
    force = np.asarray(sample.get('force'), dtype=float)
    torque = np.asarray(sample.get('torque'), dtype=float)
    if force.shape != (3,) or torque.shape != (3,) or not np.isfinite([force, torque]).all():
        return '六维力数据格式无效', None, None
    force_norm = float(np.linalg.norm(force))
    torque_norm = float(np.linalg.norm(torque))
    if force_norm > float(force_limit_n):
        return f'合力 {force_norm:.2f}N 超过 {force_limit_n:.2f}N', force_norm, torque_norm
    if torque_norm > float(torque_limit_nm):
        return f'合力矩 {torque_norm:.3f}N·m 超过 {torque_limit_nm:.3f}N·m', force_norm, torque_norm
    if policy and policy.get('calibrated'):
        # Raw norm limits above always remain active, even with compensation.
        from click_pose import validate_transform
        try:
            if 'force_coordinate' in sample and int(sample['force_coordinate'])!=int(policy.get('input_coordinate',0)):
                raise ValueError('六维力坐标类型与标定不一致')
            T=validate_transform(policy['T_mount_sensor'])
            bias=np.asarray(policy.get('bias',[0.]*6),float)
            limits=np.asarray(policy['component_limits'],float)
            if bias.shape!=(6,) or limits.shape!=(6,) or not np.isfinite([bias,limits]).all() or np.any(limits<=0):
                raise ValueError('六维偏置/分量阈值无效')
            mount_force=T[:3,:3]@(force-bias[:3])
            mount_torque=T[:3,:3]@(torque-bias[3:])+np.cross(T[:3,3],mount_force)
            values=np.r_[mount_force,mount_torque]
            if np.any(abs(values)>limits):
                axis=int(np.argmax(abs(values)/limits))
                return f'连接件方向分量 {axis+1} 超限：{values[axis]:.3f}（阈值 {limits[axis]:.3f}）',force_norm,torque_norm
        except Exception as exc:return '连接件力保护配置无效：'+str(exc),force_norm,torque_norm
    return None, force_norm, torque_norm

from click_pose import validate_transform


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def hand_positions(values):
    a = np.asarray(values, dtype=float)
    if a.shape != (6,) or not np.isfinite(a).all() or np.any(a < 0) or np.any(a > 1):
        raise ValueError("手部配置必须是按实际驱动顺序排列的 6 个 [0,1] 开合量")
    return a.tolist()


def rotation_from_quaternion(values):
    q = np.asarray(values, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-9:
        raise ValueError("四元数无效")
    x, y, z, w = q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def read_feedback(path, now=None):
    now = time.time() if now is None else now
    data = json.loads(Path(path).read_text())
    # No fallback to target joints: those are commands, not hardware feedback.
    for key, maximum in (("arm_received_at", .5), ("waist_received_at", 2.)):
        age = now-float(data.get(key, 0))
        if not 0 <= age <= maximum:
            raise ValueError(f"{key} 缺失/过期；请重启更新后的 host_fk 并确认真实关节反馈")
    q = np.asarray(data.get('q16'), dtype=float)
    if q.shape != (16,) or not np.isfinite(q).all():
        raise ValueError('需要完整、有限的 16 关节实测状态')
    return data


def timed_waypoints(rows, velocity, acceleration):
    rows = np.asarray(rows, dtype=float)
    times = [0.]
    for a, b in zip(rows, rows[1:]):
        delta = float(np.max(np.abs(b-a)))
        # Quintic smoothstep extrema: max s'=1.875, max |s''|<5.78.
        duration = max(.08, 1.875*delta/velocity, (5.78*delta/acceleration)**.5)
        times.append(times[-1]+duration)
    return times


def sample_trajectory(rows, times, elapsed):
    rows, times = np.asarray(rows), np.asarray(times)
    if elapsed >= times[-1]:
        return rows[-1].copy()
    index = max(0, int(np.searchsorted(times, elapsed, side='right'))-1)
    u = float(np.clip((elapsed-times[index])/(times[index+1]-times[index]), 0, 1))
    s = 10*u**3-15*u**4+6*u**5
    return rows[index]+s*(rows[index+1]-rows[index])


class JointRunner:
    """Tick-driven checked joint playback, completion based on fresh feedback.

    This is NOT a collision planner or hardware emergency stop. Cancellation
    holds the measured joints and waits for successive stationary observations.
    """
    def __init__(self, plan, publish, feedback, enabled, clock=time.monotonic):
        self.plan, self.publish, self.feedback, self.enabled, self.clock = plan, publish, feedback, enabled, clock
        self.rows = np.asarray(plan['joint_waypoints'], dtype=float)
        self.times = np.asarray(plan['times_s'], dtype=float)
        if (self.rows.ndim != 2 or self.rows.shape[1] != 16 or len(self.rows) < 2
                or len(self.times) != len(self.rows) or not np.isfinite(self.rows).all()
                or not np.isfinite(self.times).all() or self.times[0] != 0 or np.any(np.diff(self.times) <= 0)):
            raise ValueError('预检轨迹格式无效')
        measured = self.feedback()
        if measured.get('urdf_sha256') != plan.get('urdf_sha256'):
            raise ValueError('反馈与轨迹使用了不同的机器人模型')
        if exceeds(measured['q16'], self.rows[0], 'start'):
            raise ValueError('机器人已偏离轨迹起点，请重新预检')
        self.started = clock()
        self.state, self.message = 'running', ''
        self.last_command = self.rows[0].copy()
        self.last_feedback_stamp = None
        self.stable_count = 0
        self.failure = False
        self.trajectory_elapsed = 0.0

    def pause(self):
        if self.state != 'running':
            return
        self.pause_started = self.clock()
        self.stable_count = 0
        self.last_feedback_stamp = None
        self.state, self.message = 'pausing', '正在保持当前位置，等待静止反馈'

    def resume(self):
        if self.state != 'paused':
            raise ValueError('尚未确认暂停到位，不能继续')
        measured = self.feedback()
        if not self.enabled() or measured.get('urdf_sha256') != self.plan.get('urdf_sha256'):
            raise ValueError('使能或机器人模型不匹配，不能继续')
        if exceeds(measured['q16'], self.last_command, 'hold'):
            raise ValueError('暂停后关节位置发生变化，请停止并重新预检')
        self.started = self.clock() - self.trajectory_elapsed
        self.stable_count = 0
        self.last_feedback_stamp = None
        self.state, self.message = 'running', '继续执行本段'

    def stop(self, reason='操作员取消', failed=False):
        if self.state not in ('running', 'pausing', 'paused', 'stopping'):
            return
        if self.state != 'stopping':
            self.stop_started = self.clock()
            self.hold = None
            self.stable_count = 0
        self.state, self.message = 'stopping', reason
        self.failure = self.failure or failed

    def tick(self):
        if self.state not in ('running', 'pausing', 'paused', 'stopping'):
            return
        now = self.clock()
        try:
            measured = self.feedback()
            q = np.asarray(measured['q16'], dtype=float)
            if not self.enabled():
                self.stop('使能丢失，停止后续发送', failed=True)
            if self.state == 'stopping':
                if self.hold is None:
                    self.hold = q.copy()
                if self.enabled():
                    self.publish(self.hold.tolist())
                if measured['arm_received_at'] != self.last_feedback_stamp:
                    self.stable_count = self.stable_count+1 if not exceeds(q, self.hold, 'hold') else 0
                    self.last_feedback_stamp = measured['arm_received_at']
                if self.stable_count >= 3:
                    self.state = 'failed' if self.failure else 'stopped'
                elif now-self.stop_started > 3:
                    self.state, self.message = 'stop_unconfirmed', '保持位置未获反馈确认，请使用机器人停止装置'
                return
            if self.state in ('pausing', 'paused'):
                if measured.get('urdf_sha256') != self.plan.get('urdf_sha256'):
                    raise ValueError('暂停期间机器人模型变化')
                error = np.max(np.abs(q-self.last_command))
                if exceeds(q, self.last_command, 'tracking') or (self.state == 'paused' and exceeds(q, self.last_command, 'hold')):
                    raise ValueError('暂停期间关节偏离保持位置')
                self.publish(self.last_command.tolist())
                if measured['arm_received_at'] != self.last_feedback_stamp:
                    self.stable_count = self.stable_count+1 if not exceeds(q, self.last_command, 'hold') else 0
                    self.last_feedback_stamp = measured['arm_received_at']
                if self.stable_count >= 3:
                    self.state, self.message = 'paused', '已暂停，可点击继续执行'
                elif now-self.pause_started > 3:
                    self.stop('暂停未获静止反馈确认', failed=True)
                return
            if exceeds(q, self.last_command, 'tracking'):
                self.stop('实际关节偏离指令超过逐关节阈值（升降0.02m，旋转0.15rad）', failed=True)
                return
            side = self.plan['arm_side']
            waist_fixed = [] if self.plan.get('waist_included') else [0, 1]
            fixed = ([] if self.plan.get('other_tcp_held') else
                     ([0, 1] if side == 'both' else
                      (list(range(9, 16)) + waist_fixed if side == 'left'
                       else list(range(2, 9)) + waist_fixed)))
            if fixed and exceeds(q, self.rows[0], 'fixed', fixed):
                self.stop('另一只手臂或腰部发生移动', failed=True)
                return
            if self.plan.get('other_tcp_held'):
                held=measured[self.plan['held_side']]
                target=validate_transform(self.plan['held_target'])
                rotation=rotation_from_quaternion(held['xyzw'])
                angle=np.arccos(np.clip((np.trace(target[:3,:3].T@rotation)-1)/2,-1,1))
                if np.linalg.norm(np.asarray(held['xyz'])-target[:3,3])>TCP_POSITION_TOLERANCE_M or angle>TCP_ROTATION_TOLERANCE_RAD:
                    self.stop('另一只手TCP偏离保持目标',failed=True)
                    return
            elapsed = now-self.started
            # Large event-loop stalls must not skip forward along a trajectory.
            command = (sample_profiled_trajectory(self.plan, elapsed) if 'trajectory_coefficients' in self.plan
                       else sample_trajectory(self.rows, self.times, elapsed))
            if exceeds(command, self.last_command, 'tick'):
                self.stop('控制循环延迟过大，取消后续轨迹', failed=True)
                return
            self.publish(command.tolist())
            self.last_command = command
            self.trajectory_elapsed = elapsed
            if elapsed >= self.times[-1]:
                reached = not exceeds(q, self.rows[-1], 'goal')
                for target_side in (('left','right') if side == 'both' else (side,)):
                    tcp = measured[target_side]
                    T = validate_transform(self.plan['targets'][target_side] if side == 'both' else self.plan['target'])
                    R = rotation_from_quaternion(tcp['xyzw'])
                    angle = np.arccos(np.clip((np.trace(T[:3, :3].T @ R)-1)/2, -1, 1))
                    reached = (reached
                               and np.linalg.norm(np.asarray(tcp['xyz'])-T[:3, 3]) < self.plan.get('position_tolerance_m',TCP_POSITION_TOLERANCE_M)
                               and angle < self.plan.get('rotation_tolerance_rad',TCP_ROTATION_TOLERANCE_RAD))
                if self.plan.get('other_tcp_held'):
                    held_side = self.plan['held_side']
                    held_tcp = measured[held_side]
                    held_target = validate_transform(self.plan['held_target'])
                    held_rotation = rotation_from_quaternion(held_tcp['xyzw'])
                    held_angle = np.arccos(np.clip(
                        (np.trace(held_target[:3,:3].T@held_rotation)-1)/2, -1, 1))
                    reached = (reached
                               and np.linalg.norm(np.asarray(held_tcp['xyz'])-held_target[:3,3]) < TCP_POSITION_TOLERANCE_M
                               and held_angle < TCP_ROTATION_TOLERANCE_RAD)
                if measured['arm_received_at'] != self.last_feedback_stamp:
                    self.stable_count = self.stable_count+1 if reached else 0
                    self.last_feedback_stamp = measured['arm_received_at']
                if self.stable_count >= 3:
                    self.state, self.message = 'complete', '关节与 TCP 实际反馈已到位'
                elif elapsed > self.times[-1]+5:
                    self.stop('到位反馈超时', failed=True)
        except Exception as exc:
            self.stop(str(exc), failed=True)
            # With stale feedback no new holding target can be trusted.
            if now-self.stop_started > 3:
                self.state, self.message = 'stop_unconfirmed', str(exc)+'；停止未确认'


def load_profile(path, require_calibration=True):
    with open(path, encoding="utf-8") as stream:
        profile = json.load(stream)
    profile["calibrated"] = profile.get("calibrated") is True
    if require_calibration and not profile["calibrated"]:
        raise ValueError("抓取配置尚未完成手型/TCP 标定")
    for key in ("hand_model", "base_frame", "tcp_link", "planner_python", "urdf_path"):
        if not isinstance(profile.get(key), str) or not profile[key].strip():
            raise ValueError(f"抓取配置缺少 {key}")
    if profile.get("arm_side") not in ("left", "right"):
        raise ValueError("必须明确选择 left 或 right，禁止自动切换手臂")
    validate_transform(profile["T_grasp_tcp"])
    hand_positions(profile["open_positions"])
    hand_positions(profile["close_positions"])
    expected_tcp = "Link7_l" if profile["arm_side"] == "left" else "Link7_r"
    if profile["tcp_link"] != expected_tcp:
        raise ValueError(f"当前宿主机 IK 控制 {expected_tcp}；请把实际手的 TCP 偏移标定到这个 link")
    if profile.get("base_frame") != "base_link":
        raise ValueError("当前宿主机链路要求 base_link 坐标，不能默认为 map 与 base_link 相同")
    for key, lo, hi in (("joint_velocity_rad_s", .01, .3), ("joint_acceleration_rad_s2", .01, .5)):
        value = float(profile[key])
        if not np.isfinite(value) or not lo <= value <= hi:
            raise ValueError(f"{key} 必须在 [{lo}, {hi}] 内")
    axis = np.asarray(profile["approach_axis_grasp"], dtype=float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or not np.isclose(np.linalg.norm(axis), 1):
        raise ValueError("approach_axis_grasp 必须是已标定的单位接近方向")
    for key, lower, upper in (("pregrasp_distance_m", .01, .30), ("lift_distance_m", .01, .20),
                              ("max_plan_age_s", .1, 3600)):
        value = float(profile[key])
        if not np.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f"{key} 必须在 [{lower}, {upper}] 内")
    bounds = np.asarray(profile["workspace_m"], dtype=float)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[0] >= bounds[1]):
        raise ValueError("workspace_m 需要实测工作空间的最小、最大 XYZ")
    for key in ("planner_python", "urdf_path"):
        candidate = Path(profile[key]).expanduser()
        if not candidate.is_absolute():
            candidate = Path(path).resolve().parent / candidate
        profile[key] = os.path.abspath(candidate)  # Preserve venv Python symlink; resolving it loses the environment.
    return profile


def make_preview(observation, candidate_index, profile, now=None):
    del now  # 观测年龄不再限制预览或执行
    if observation.get("execution_blockers"):
        raise ValueError("；".join(observation["execution_blockers"]))
    if not observation.get("ok"):
        raise ValueError("没有成功的点击估计")
    captured = float(observation.get("captured_at", 0) or 0)
    if observation.get("base_frame") != profile["base_frame"] or "T_base_camera" not in observation:
        raise ValueError("缺少与机器人一致的相机外参，不能把相机 XYZ 当作机器人坐标")
    grasps = observation.get("grasps", {})
    if not grasps.get("ok") or grasps.get("gripper") != profile["hand_model"]:
        raise ValueError("没有与当前实际手型匹配的抓取候选")
    candidates = grasps.get("candidates", [])
    if not 0 <= candidate_index < len(candidates):
        raise ValueError("请先选择抓取候选")
    candidate = candidates[candidate_index]
    for key, profile_key in (("open_positions", "open_positions"), ("close_positions", "close_positions")):
        if key in candidate and not np.allclose(hand_positions(candidate[key]), profile[profile_key]):
            raise ValueError("候选手指姿态与当前已标定开合映射不同；需专用手型配置才能执行")
    T_base_grasp = validate_transform(observation["T_base_camera"]) @ validate_transform(candidate["T_camera_grasp"])
    grasp = T_base_grasp @ validate_transform(profile["T_grasp_tcp"])
    pregrasp = grasp.copy()
    # The approach direction belongs to the model grasp frame, not the TCP frame.
    pregrasp[:3, 3] -= T_base_grasp[:3, :3] @ np.asarray(profile["approach_axis_grasp"]) * profile["pregrasp_distance_m"]
    lift = grasp.copy()
    lift[2, 3] += profile["lift_distance_m"]
    bounds = np.asarray(profile["workspace_m"])
    violations = []
    for label, T in (("预抓取", pregrasp), ("接近", grasp), ("抬起", lift)):
        if np.any(T[:3, 3] < bounds[0]) or np.any(T[:3, 3] > bounds[1]):
            violations.append(f"{label}目标超出已标定工作空间：XYZ={T[:3, 3].round(4).tolist()}，范围={bounds.tolist()}")
    preview = {"arm_side": profile["arm_side"], "hand_model": profile["hand_model"],
               "base_frame": profile["base_frame"], "tcp_link": profile["tcp_link"],
               "captured_at": captured, "observation_id": observation["observation_id"],
               "candidate_index": candidate_index,
               "goals": {"pregrasp": pregrasp.tolist(), "approach": grasp.tolist(), "lift": lift.tolist()},
               "open_positions": hand_positions(candidate.get("open_positions", profile["open_positions"])),
               "close_positions": hand_positions(candidate.get("close_positions", profile["close_positions"]))}
    preview["id"] = fingerprint(preview)
    if violations:
        preview['diagnostic_only'] = True
        preview['workspace_errors'] = violations
        error = ValueError('；'.join(violations))
        error.preview = preview
        raise error
    return preview


def manual_candidate(observation, profile, feedback, offset_base):
    """Human-selected contact point and current wrist orientation, not learned grasp."""
    if observation.get('execution_blockers'):
        raise ValueError('；'.join(observation['execution_blockers']))
    if observation.get('base_frame') != profile['base_frame'] or 'T_base_camera' not in observation:
        raise ValueError('人工候选仍需真实相机外参')
    offset = np.asarray(offset_base, dtype=float)
    if offset.shape != (3,) or not np.isfinite(offset).all() or np.any(np.abs(offset) > .15):
        raise ValueError('人工接触偏移限于各轴 ±0.15m')
    if feedback.get('urdf_sha256') != hashlib.sha256(Path(profile['urdf_path']).read_bytes()).hexdigest():
        raise ValueError('当前 TCP 反馈与抓取配置模型不一致')
    T_base_camera = validate_transform(observation['T_base_camera'])
    T_base_object = T_base_camera @ validate_transform(observation['object_pose']['T_camera_object'])
    T_grasp_tcp = validate_transform(profile['T_grasp_tcp'])
    T_base_grasp = np.eye(4)
    T_base_grasp[:3, :3] = rotation_from_quaternion(feedback[profile['arm_side']]['xyzw']) @ T_grasp_tcp[:3, :3].T
    T_base_grasp[:3, 3] = T_base_object[:3, 3] + offset
    return {'ok': True, 'gripper': profile['hand_model'], 'model': 'manual_current_hand_orientation',
            'observation_id': observation['observation_id'],
            'candidates': [{'T_camera_grasp': (np.linalg.inv(T_base_camera) @ T_base_grasp).tolist(),
                            'model_score': 0., 'source': 'manual',
                            'note': '人工接触点+当前手朝向，不是自动抓取预测'}]}


class Sequence:
    """Manual boundaries prevent a timer or command-sent event proving contact.

    Opening and closing require explicit confirmation. Every arm segment is
    prepared separately from current joint feedback, then explicitly executed.
    """
    NEXT = {"idle": "open", "opened": "pregrasp", "at_pregrasp": "approach",
            "at_grasp": "close", "grasp_confirmed": "lift"}

    def __init__(self):
        self.state = "idle"
        self.preview = None
        self.profile = None
        self.plan = None
        self.error = ""

    def reset(self, preview, profile):
        if self.state in ("executing", "stopping", "planning", "await_open", "await_grasp",
                          "opened", "at_pregrasp", "at_grasp", "grasp_confirmed", "planned"):
            raise ValueError("当前抓取尚未结束；请先停止，不能替换目标")
        self.preview, self.profile = copy.deepcopy(preview), copy.deepcopy(profile)
        self.state, self.plan, self.error = "idle", None, ""

    @property
    def stage(self):
        return self.NEXT.get(self.state)

    def hand_sent(self):
        if self.state == "idle":
            self.state = "await_open"
        elif self.state == "at_grasp":
            self.state = "await_grasp"
        else:
            raise ValueError("当前阶段不能发送开合命令")

    def confirm_hand(self):
        if self.state == "await_open":
            self.state = "opened"
        elif self.state == "await_grasp":
            self.state = "grasp_confirmed"
        else:
            raise ValueError("当前没有待确认的手部动作")

    def begin_plan(self, stage=None):
        stage = self.stage if stage is None else stage
        if stage not in ("pregrasp", "approach", "lift"):
            raise ValueError("当前阶段不能规划机械臂")
        self.pending_stage = stage
        self.state = "planning"
        return stage

    def accept_plan(self, plan, now=None):
        if self.state != "planning":
            raise ValueError("已停止或过期的规划结果")
        if (not plan.get("ok") or plan.get("preview_id") != self.preview["id"]
                or plan.get("stage") != self.pending_stage
                or plan.get("backend") != "host_ik_manual"
                or plan.get("joint_limits_checked") is not True
                or plan.get("ik_checked") is not True):
            self.fail("规划失败或规划结果与当前目标不一致")
            raise ValueError(self.error)
        self.plan = copy.deepcopy(plan)
        self.plan["accepted_at"] = time.monotonic() if now is None else now
        self.state = "planned"

    def begin_execute(self, now=None, manual_path_confirmed=False):
        now = time.monotonic() if now is None else now
        if self.state != "planned" or self.plan is None:
            raise ValueError("先规划并检查当前阶段")
        if now - self.plan["accepted_at"] > self.profile["max_plan_age_s"]:
            self.fail("规划已过期，请停止后重新估计/规划")
            raise ValueError(self.error)
        if not manual_path_confirmed:
            raise ValueError("请先确认现场与碰撞场景模型一致，本段路径没有未建模障碍")
        self.state = "executing"
        return copy.deepcopy(self.plan)

    def motion_finished(self, ok, reason=""):
        if self.state == "stopping":
            self.state = "stopped" if ok else "stop_unconfirmed"
            self.error = reason
            return
        if self.state != "executing":
            return
        if not ok:
            self.fail(reason or "执行失败")
        else:
            self.state = {"pregrasp": "at_pregrasp", "approach": "at_grasp", "lift": "complete", "restore": "complete"}[self.pending_stage]
            self.plan = None

    def stop(self, motion_active=False):
        self.plan = None
        self.state = "stopping" if motion_active else "stopped"

    def fail(self, reason):
        self.error, self.plan, self.state = reason, None, "failed"
