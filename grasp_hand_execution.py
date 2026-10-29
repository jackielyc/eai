"""Feedback-confirmed hand actions, independent of Qt and robot transport.

Position completion only confirms a motor command, never object contact.
"""
import time
import numpy as np

from grasp_execution import hand_positions


class HandRunner:
    def __init__(self, side, target, publish, feedback, enabled,
                 clock=time.monotonic, wall_clock=time.time, timeout=10., tolerance=.03):
        self.side, self.target = side, np.asarray(hand_positions(target))
        self.publish, self.feedback, self.enabled = publish, feedback, enabled
        self.clock, self.wall_clock = clock, wall_clock
        self.timeout, self.tolerance = float(timeout), float(tolerance)
        if not np.isfinite([self.timeout,self.tolerance]).all() or not 1<=self.timeout<=60 or not .001<=self.tolerance<=.1:
            raise ValueError('手部超时需1~60秒、归一化反馈容差需0.001~0.1')
        self.state, self.message = 'ready', ''
        self.last_stamp, self.stable_count = None, 0
        self.failure = False

    def _sample(self):
        sample = self.feedback()
        stamp = float(sample['stamp'])
        if not np.isfinite(stamp) or not 0 <= self.wall_clock()-stamp <= .5:
            raise ValueError('手部反馈缺失或过期（需500ms内反馈）')
        return np.asarray(hand_positions(sample['positions'])), stamp

    def start(self):
        if self.state != 'ready':
            raise ValueError('手部动作已经启动')
        self._sample()  # Never send a command without real, fresh feedback.
        if not self.enabled():
            raise ValueError('手部未使能')
        self.started = self.clock()
        self.command_stamp = self.wall_clock()
        try:
            self.publish(self.target.tolist())
        except Exception:
            self.state, self.message = 'stop_unconfirmed', '手部命令发送异常，停止未确认'
            raise
        self.state, self.message = 'running', '等待手部实际反馈到位；不代表已抓稳物体'

    def _hold(self, state):
        q, stamp = self._sample()
        if not self.enabled():
            raise ValueError('手部使能丢失')
        self.hold = q
        self.command_stamp = self.wall_clock()
        self.publish(q.tolist())
        self.last_stamp, self.stable_count = stamp, 0
        self.hold_started = self.clock()
        self.state = state

    def pause(self):
        if self.state == 'running':
            try:
                self._hold('pausing')
                self.message = '手部正在保持实测位置，等待暂停确认'
            except Exception as exc:
                self.state, self.message = 'stop_unconfirmed', str(exc)+'；手部停止未确认'

    def resume(self):
        if self.state != 'paused':
            raise ValueError('手部尚未确认暂停')
        q, _ = self._sample()
        if not self.enabled() or np.max(np.abs(q-self.hold)) > self.tolerance:
            raise ValueError('手部暂停状态变化，不能继续')
        self.started, self.command_stamp = self.clock(), self.wall_clock()
        self.last_stamp, self.stable_count = None, 0
        try:
            self.publish(self.target.tolist())
        except Exception:
            self.state, self.message = 'stop_unconfirmed', '继续手部动作时发送异常，停止未确认'
            raise
        self.state, self.message = 'running', '继续等待手部实际反馈到位'

    def stop(self, reason='手部动作已取消', failed=False):
        if self.state in ('complete', 'stopped', 'failed', 'stop_unconfirmed'):
            return
        self.failure = self.failure or failed
        if self.state != 'stopping':
            try:
                self._hold('stopping')
            except Exception as exc:
                self.state, self.message = 'stop_unconfirmed', str(exc)+'；手部停止未确认'
                return
        self.message = reason

    def tick(self):
        if self.state not in ('running', 'pausing', 'paused', 'stopping'):
            return
        try:
            q, stamp = self._sample()
            if not self.enabled():
                raise ValueError('手部使能丢失')
            target = self.target if self.state == 'running' else self.hold
            error = float(np.max(np.abs(q-target)))
            if self.state == 'paused' and error > self.tolerance:
                raise ValueError('暂停期间手部偏离保持位置')
            # Repeated polls of one sample cannot prove completion.
            if stamp > self.command_stamp and (self.last_stamp is None or stamp > self.last_stamp):
                self.stable_count = self.stable_count+1 if error <= self.tolerance else 0
                self.last_stamp = stamp
            if self.stable_count >= 3:
                if self.state == 'running':
                    self.state, self.message = 'complete', '手部位置反馈已到位；仍需确认夹持结果'
                elif self.state == 'pausing':
                    self.state, self.message = 'paused', '手部已暂停'
                elif self.state == 'stopping':
                    self.state = 'failed' if self.failure else 'stopped'
                return
            if self.state == 'running' and self.clock()-self.started > self.timeout:
                self.stop('手部到位超时；可能接触物体，请检查夹持后重试', failed=True)
            elif self.state in ('pausing', 'stopping') and self.clock()-self.hold_started > 3:
                self.state, self.message = 'stop_unconfirmed', '手部保持位置未获反馈确认，请检查机器人'
        except Exception as exc:
            self.state, self.message = 'stop_unconfirmed', str(exc)+'；手部停止未确认'
