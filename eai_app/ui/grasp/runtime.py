"""Runtime settings, scene preparation and execution adapters for the grasp panel."""
import copy
import json
import os
from pathlib import Path
import time
import numpy as np
from PyQt5.QtWidgets import QHBoxLayout,QPushButton,QCheckBox,QDialog,QVBoxLayout,QTextEdit,QDialogButtonBox,QLabel,QFormLayout,QDoubleSpinBox
from eai_app.domain.grasp.execution import fingerprint
from eai_app.domain.tasks.world import ManipulationWorld,object_matrix


class GraspRuntimeMixin:
    def init_runtime(self, verify_observation=None):
        self.verify_observation_callback=verify_observation
        self.world=ManipulationWorld();self.pending_attachment=None
        self._unverified_grips=set()
        self.scene_path=self.root/'config/grasp_scene.json'
        self.assembly_path=self.root/'config/grasp_assembly.json'
        self._runtime_verification=None
        self._hand_purpose='workflow'
        self._perception_pending=False;self._perception_token=None;self._reestimated_observation=None

    def build_runtime_controls(self,layout):
        row=QHBoxLayout()
        self.collision_check=QCheckBox('碰撞复检（含双手/支架/相机）')
        self.collision_check.setChecked(True);row.addWidget(self.collision_check)
        self.reestimate_check=QCheckBox('接近前重新估计目标')
        self.reestimate_check.setChecked(True);row.addWidget(self.reestimate_check)
        self.runtime_verify_button=QPushButton('现场复核目标')
        self.runtime_verify_button.clicked.connect(self.verify_runtime_target);row.addWidget(self.runtime_verify_button)
        layout.addLayout(row);row=QHBoxLayout()
        self.scene_edit_button=QPushButton('场景标定配置…')
        self.scene_edit_button.clicked.connect(lambda:self.edit_runtime_json(self.scene_path));row.addWidget(self.scene_edit_button)
        self.assembly_edit_button=QPushButton('装配关系标定…')
        self.assembly_edit_button.clicked.connect(lambda:self.edit_runtime_json(self.assembly_path));row.addWidget(self.assembly_edit_button)
        self.rank_candidates_button=QPushButton('采样并评估抓取朝向')
        self.rank_candidates_button.clicked.connect(self.rank_runtime_candidates);row.addWidget(self.rank_candidates_button)
        self.motion_limits_button=QPushButton('执行约束…')
        self.motion_limits_button.clicked.connect(self.edit_motion_limits);row.addWidget(self.motion_limits_button)
        self.attach_button=QPushButton('确认当前目标已夹持')
        self.attach_button.clicked.connect(self.confirm_runtime_attachment);row.addWidget(self.attach_button)
        self.empty_hand_button=QPushButton('确认当前手为空')
        self.empty_hand_button.clicked.connect(self.confirm_empty_hand);row.addWidget(self.empty_hand_button)
        layout.addLayout(row)
        self.runtime_status=QLabel('真机执行需要已标定场景和现场复核；历史缓存仍可预览。')
        self.runtime_status.setWordWrap(True);layout.addWidget(self.runtime_status)
        self.update_runtime_labels()

    def update_runtime_labels(self):
        try:scene_ready=json.loads(self.scene_path.read_text()).get('calibrated') is True
        except Exception:scene_ready=False
        try:
            relations=json.loads(self.assembly_path.read_text()).get('relations',{})
            assembly_ready=bool(relations) and all(r.get('calibrated') for r in relations.values())
        except Exception:assembly_ready=False
        self.scene_edit_button.setText('场景标定配置…' if scene_ready else '场景标定（待完成）…')
        self.assembly_edit_button.setText('装配关系标定…' if assembly_ready else '装配关系（待完成）…')

    @property
    def use_execution_service(self):
        # Tests supply an explicit fake backend; production never silently falls back.
        return getattr(self.node,'_grasp_use_process_runner',True)

    def edit_motion_limits(self):
        if self.reserved or self.runner or self.hand_runner or self.process:return
        try:
            path=Path(self.profile_path.text());data=json.loads(path.read_text())
            dialog=QDialog(self);dialog.setWindowTitle('执行约束：升降用米，旋转用弧度')
            layout=QVBoxLayout(dialog);form=QFormLayout();layout.addLayout(form);fields={}
            specs=(('joint_velocity_rad_s','旋转速度 rad/s',.1,.01,.3),
                ('joint_acceleration_rad_s2','旋转加速度 rad/s²',.2,.01,.5),
                ('joint_jerk_rad_s3','旋转 jerk rad/s³',2.,.01,10.),
                ('lift_velocity_m_s','升降速度 m/s',.03,.001,.1),
                ('lift_acceleration_m_s2','升降加速度 m/s²',.05,.001,.2),
                ('lift_jerk_m_s3','升降 jerk m/s³',.2,.01,1.),
                ('hand_timeout_s','手部到位超时 s',10.,1.,60.),
                ('hand_feedback_tolerance','手部位置容差（0~1）',.03,.001,.1))
            for key,label,default,lo,hi in specs:
                spin=QDoubleSpinBox();spin.setDecimals(3);spin.setRange(lo,hi);spin.setValue(data.get(key,default))
                form.addRow(label,spin);fields[key]=spin
            force=QTextEdit(json.dumps(data.get('force_policy',{'calibrated':False,
                'T_mount_sensor':None,'bias':[0.]*6,'component_limits':None}),ensure_ascii=False,indent=2))
            form.addRow('连接件六维力标定（待实测）',force)
            buttons=QDialogButtonBox(QDialogButtonBox.Save|QDialogButtonBox.Cancel);layout.addWidget(buttons)
            buttons.accepted.connect(dialog.accept);buttons.rejected.connect(dialog.reject)
            if dialog.exec_()!=QDialog.Accepted:return
            policy=json.loads(force.toPlainText())
            if policy.get('calibrated'):
                from eai_app.domain.grasp.execution import evaluate_wrench
                violation,*_=evaluate_wrench({'stamp':time.time(),'force':[0]*3,'torque':[0]*3},1e6,1e6,policy=policy)
                if violation and '配置无效' in violation:raise ValueError(violation)
            data.update({k:v.value() for k,v in fields.items()});data['force_policy']=policy
            from eai_app.adapters.robot.execution_service import atomic_json
            atomic_json(path,data);self.invalidate_preview();self.message='执行约束已保存，请重新预检。'
        except Exception as exc:self.message='执行约束未保存：'+str(exc)
        self.refresh()

    def edit_runtime_json(self,path):
        if self.reserved or self.runner or self.hand_runner or self.process:return
        dialog=QDialog(self);dialog.setWindowTitle(path.name+'（实测参数，未标定不可执行）');dialog.resize(800,600)
        layout=QVBoxLayout(dialog);editor=QTextEdit(path.read_text());layout.addWidget(editor)
        buttons=QDialogButtonBox(QDialogButtonBox.Save|QDialogButtonBox.Cancel)
        layout.addWidget(buttons);buttons.accepted.connect(dialog.accept);buttons.rejected.connect(dialog.reject)
        if dialog.exec_()!=QDialog.Accepted:return
        try:
            data=json.loads(editor.toPlainText())
            if data.get('version')!=1:raise ValueError('配置版本必须为1')
            if path==self.scene_path:
                from eai_app.planning.scene import load_scene
                tmp=path.with_suffix('.validate.json');tmp.write_text(json.dumps(data))
                try:load_scene(tmp)
                finally:tmp.unlink(missing_ok=True)
            else:
                from eai_app.domain.geometry.transforms import validate_transform
                for relation in data.get('relations',{}).values():
                    if relation.get('calibrated'):validate_transform(relation['T_reference_object'])
            from eai_app.adapters.robot.execution_service import atomic_json
            atomic_json(path,data);self._runtime_verification=None
            self.update_runtime_labels()
            if path==self.scene_path and self.scene_display_callback:
                self.scene_display_callback(data,list(self.world.attached.values()))
            self.invalidate_preview();self.runtime_status.setText('配置已保存；原计划失效，请重新预检。')
        except Exception as exc:self.runtime_status.setText('保存失败：'+str(exc))

    def verify_runtime_target(self):
        try:
            if not self.observation:raise ValueError('请先选择目标')
            if self.verify_observation_callback is None:raise ValueError('未连接实时图像复核器')
            self._runtime_verification=self.verify_observation_callback(self.observation)
            self.runtime_status.setText('现场图像/标定复核通过；这不是新的6D估计或抓取成功判定。')
            self.record_workflow_event('observation_verified',evidence=self._runtime_verification)
            return self._runtime_verification
        except Exception as exc:
            self._runtime_verification=None;self.runtime_status.setText('现场复核失败：'+str(exc))
            raise ValueError(str(exc)) from exc

    def runtime_hands(self):
        if self.is_simulation:
            sim=self.simulation_backend()
            return {side:list(sim.hands.get(side,[0.]*6)) for side in ('left','right')}
        from eai_app.domain.grasp.execution import hand_positions
        result={}
        for side in ('left','right'):
            sample=self.node.get_grasp_hand_feedback(side)
            if not 0<=time.time()-sample['stamp']<=.5:raise ValueError('碰撞检查需要新鲜双手反馈')
            result[side]=hand_positions(sample['positions'])
        return result

    def confirm_runtime_attachment(self):
        if self.runner or self.hand_runner or self.process or self.reserved:return
        try:
            if not self.observation:raise ValueError('请先估计当前已夹持物体')
            target=str(self.grasp_target_combo.currentData() or '')
            roles=[role for role,combo in self.task_role_combos.items() if str(combo.currentData() or '')==target]
            if not target or len(roles)!=1:raise ValueError('当前目标需绑定唯一任务角色')
            if not self.is_simulation:
                from grasp_perception import require_verified
                require_verified(self.observation,self.verify_runtime_target())
            side=str(self.grasp_arm.currentData())
            self.world.attach_confirmed(side,roles[0],target,self.observation,self.feedback(),replace=True)
            self._unverified_grips.discard(side)
            self.pending_attachment=None;self.record_workflow_event('operator_attachment_confirmed',attached=self.world.attached)
            self.message='已按当前物体位姿与实测TCP记录夹持关系；请重新预检。'
            self.invalidate_preview()
        except Exception as exc:self.message='夹持关系未记录：'+str(exc)
        self.refresh()

    def confirm_empty_hand(self):
        if self.runner or self.hand_runner or self.process or self.reserved:return
        side=str(self.grasp_arm.currentData())
        self.world.attached.pop(side,None);self._unverified_grips.discard(side)
        if self.pending_attachment and self.pending_attachment['side']==side:self.pending_attachment=None
        self.record_workflow_event('operator_confirmed_empty_hand',side=side)
        self.invalidate_preview();self.message='已记录当前手为空的人工确认；未发送开合命令。';self.refresh()

    def maybe_reestimate_approach(self):
        if not self.use_execution_service or not self.reestimate_check.isChecked():return False
        if self._perception_pending:return True
        if self.motion_stage.currentData()!='approach' or self.reestimate_callback is None:return False
        held=self.world.attached.get(str(self.grasp_arm.currentData()))
        if held and (self._workflow_action() or {}).get('role')==held['role']:return False
        target_id=str(self.grasp_target_combo.currentData() or '')
        if not target_id:return False
        current=(self.observation or {}).get('observation_id')
        if self._reestimated_observation and self._reestimated_observation[0]==current and time.monotonic()-self._reestimated_observation[1]<30:
            return False
        import uuid
        token=uuid.uuid4().hex;self._perception_token=token;self._perception_pending=True
        self._perception_started=time.monotonic()
        self.set_reserved(True);method=str(self.source_combo.currentData())
        self.message='接近前重新估计目标；尚未发送运动命令。';self.refresh()
        def finished(observations,error=None):
            if self._perception_token!=token:return
            self._perception_pending=False;self._perception_token=None;self.set_reserved(False)
            if error or method not in (observations or {}):
                self._workflow_waiting=False;self._workflow_error='接近前重估失败：'+str(error or '所选方法无结果')
                self._workflow_auto_paused=self._workflow_auto_running;self.message=self._workflow_error
                self.refresh();return
            was_waiting=self._workflow_waiting
            self.target_observations[target_id]['methods']=copy.deepcopy(observations)
            self.set_method_observations(observations)
            self._workflow_waiting=was_waiting
            self._reestimated_observation=(self.observation['observation_id'],time.monotonic())
            self.record_workflow_event('approach_reestimated')
            if not self._workflow_auto_paused:self.execute()
            else:
                self._workflow_waiting=False
                self.message='重估完成；流程已暂停，继续后重新预检。';self.refresh()
        try:self.reestimate_callback(target_id,finished)
        except Exception as exc:finished(None,str(exc))
        return True

    def rank_runtime_candidates(self):
        if self.runner or self.process or self.hand_runner or self.reserved:return
        try:
            from grasp_candidates import orientation_proposals
            from eai_app.domain.grasp.execution import manual_candidate,make_preview,Sequence
            profile=self.selected_profile(require_calibration=False)
            if not self.observation:raise ValueError('请先选择目标')
            self._ensure_extrinsic();feedback=self.feedback()
            grasps=manual_candidate(self.observation,profile,feedback,[s.value()/100 for s in self.contact_offsets])
            grasps['candidates']=orientation_proposals(grasps['candidates'][0])
            self.observation['grasps']=grasps;self.populate_candidates()
            previews=[make_preview(self.observation,i,profile) for i in range(len(grasps['candidates']))]
            self.sequence=Sequence();self.sequence.reset(previews[0],profile)
            self.sequence.state='planning';self.set_reserved(True)
            self.launch('rank_candidates',start_joints=feedback['q16'],
                candidate_goals=[p['goals']['pregrasp'] for p in previews],
                observation_id=self.observation['observation_id'],allow_waist=self.waist_participation.isChecked())
        except Exception as exc:
            self.message='候选评估未启动：'+str(exc);self.set_reserved(False);self.refresh()

    def accept_candidate_ranking(self,result):
        from eai_app.domain.grasp.execution import Sequence
        if self.sequence.state=='stopped':return
        if not result.get('ok') or result.get('observation_id')!=(self.observation or {}).get('observation_id'):
            self.message='候选评估失败或结果已过期：'+str(result.get('error',''));return
        for row in result['rankings']:
            index=row['index']
            label=(f'候选 {index+1} / 代价 {row["score"]:.3f} / 关节余量 {row["joint_margin"]:.1%}'
                   if row['ok'] else f'候选 {index+1} / {row["reason"]}')
            self.candidates.setItemText(index,label)
        best=next((r for r in result['rankings'] if r['ok']),None)
        if best:self.candidates.setCurrentIndex(best['index'])
        self.sequence=Sequence();self.message=result['note']+'；已选择推荐候选，请预览。'

    def runtime_scene(self):
        from eai_app.planning.scene import load_scene
        scene=load_scene(self.scene_path,hardware=not self.is_simulation)
        if not self.is_simulation and not scene.get('world'):
            raise ValueError('真机环境几何为空，请至少配置实测桌面/工装')
        occupied={o['id'] for o in scene.get('world',[])}|set(scene.get('modeled_targets',[]))
        attached={o['id'] for o in self.world.attached.values()}
        for target_id,target in self.target_observations.items():
            if target_id in occupied or target_id in attached:continue
            methods=target.get('methods',{});obs=methods.get(str(self.source_combo.currentData()))
            if not obs:continue
            item=self.world.placed.get(target_id)
            scene['world'].append({'id':target_id,'T_base_object':item['T_base_object'] if item else object_matrix(obs).tolist(),
                                   'size_m':obs['object_pose']['extents_m']})
        if self.scene_display_callback:self.scene_display_callback(scene,list(self.world.attached.values()))
        return scene

    def runtime_plan_inputs(self):
        if not self.use_execution_service:return {}
        if not self.is_simulation and not self.collision_check.isChecked():
            raise ValueError('真机不能关闭碰撞复检')
        scene=self.runtime_scene() if self.collision_check.isChecked() else None
        return {'scene':scene,'scene_path':str(self.scene_path),
                'scene_file_sha256':fingerprint(json.loads(self.scene_path.read_text())),
                'hands':self.runtime_hands(),'attachments':list(self.world.attached.values()),
                'collision_required':bool(scene),
                'assembly_file_sha256':fingerprint(json.loads(self.assembly_path.read_text())),
                'profile_sha256':fingerprint(self.selected_profile(require_calibration=False))}

    def apply_world_preview(self,preview):
        if not self.use_execution_service:return preview
        action=self._workflow_action() or {};role=action.get('role')
        side=preview['arm_side']
        operation=action.get('operation')
        if operation in ('place','insert') and side not in self.world.attached and self.motion_stage.currentData()!='lift':
            raise ValueError('放置/插入动作缺少已确认的夹持物体')
        if not role:return preview
        settings=json.loads(self.assembly_path.read_text())
        if operation=='align' and side not in self.world.attached:
            from eai_app.domain.tasks.world import alignment_preview
            result=alignment_preview(preview,str(self.task_combo.currentData()),role,self.observation,settings)
        elif side in self.world.attached or self.motion_stage.currentData()=='lift':
            result=self.world.placement_preview(preview,str(self.task_combo.currentData()),side,role,
                self.observation,settings,hardware=not self.is_simulation)
        else:return preview
        if operation=='insert':
            relation=settings.get('relations',{}).get(result.get('assembly_relation'),{})
            if not relation.get('insert_calibrated'):raise ValueError('插入精度/方向/力保护尚未标定，不能执行插入')
            if not self.is_simulation and (not self.wrench_guard_check.isChecked() or not self.selected_profile().get('force_policy',{}).get('calibrated')):
                raise ValueError('真机插入需要启用已标定的连接件六维力保护')
            precision=relation.get('precision') or {}
            position=float(precision.get('position_tolerance_m',.001));angle=float(precision.get('rotation_tolerance_rad',.01))
            scale=float(precision.get('velocity_scale',.2))
            if not np.isfinite([position,angle,scale]).all() or not 0<position<=.015 or not 0<angle<=np.deg2rad(10) or not 0<scale<=1:
                raise ValueError('插入精度/速度参数无效')
            result=copy.deepcopy(result);result['operation']='insert'
            result['precision']={'position_tolerance_m':position,'rotation_tolerance_rad':angle,'velocity_scale':scale}
            result['id']=fingerprint(result)
        bounds=np.asarray(self.selected_profile()['workspace_m'])
        if any(np.any(np.asarray(T)[:3,3]<bounds[0]) or np.any(np.asarray(T)[:3,3]>bounds[1]) for T in result['goals'].values()):
            raise ValueError('装配位姿超出已标定工作空间')
        return result

    def new_process_runner(self,kind,plan=None,side=None,target=None,hand_operation=None,hand_velocity=None,hand_effort=None):
        from eai_app.application.execution_client import ProcessRunner
        profile=self.selected_profile();feedback=self.feedback()
        request={'kind':kind,'execution_mode':'simulation' if self.is_simulation else 'hardware',
                 'profile':profile,'plan':plan or {},'seed':feedback['q16'],'hands':self.runtime_hands(),
                 'side':side,'target':target,'feedback_path':os.environ.get('EAI_HOST_FK_STATE','/tmp/eai_host_fk_state.json'),
                 'wrench_guard':self.wrench_guard_check.isChecked(),'force_limit_n':self.force_limit.value(),
                 'torque_limit_nm':self.torque_limit.value()}
        request['force_policy']=profile.get('force_policy')
        if kind=='hand':
            request['hand_operation']=hand_operation or ('hand_open' if np.allclose(target,profile['open_positions']) else 'hand_close')
            if hand_velocity is not None:request['hand_velocity']=hand_velocity
            if hand_effort is not None:request['hand_effort']=hand_effort
        request.update(self.runtime_plan_inputs())
        request['observation_provenance']=copy.deepcopy((self.observation or {}).get('provenance',{}))
        request['observation_id']=(self.observation or {}).get('observation_id')
        if not self.is_simulation and kind=='arm' and plan.get('stage')!='restore':
            request['verification']=self._runtime_verification
            request['observation_id']=(self.observation or {}).get('observation_id')
        def display(q,hands):
            sim=self.simulation_backend();sim.hands=copy.deepcopy(hands);sim.publish(q)
        runner=ProcessRunner(request,self.root/'.cache/grasp_service',display if self.is_simulation else None)
        if kind=='hand':
            if side in self.world.attached:self.world.attached[side]['verified']=False
            if request['hand_operation']=='hand_close':self._unverified_grips.add(side)
        return runner

    def validate_runtime_execution(self,plan):
        if not self.use_execution_service:return
        if getattr(self,'_program_active',False) and getattr(self,'_program_validator',None):self._program_validator()
        if not self.is_simulation and self._unverified_grips:
            raise ValueError('闭合后夹持状态未确认；请确认当前目标已夹持或确认当前手为空')
        if any(not item.get('verified') for item in self.world.attached.values()):
            raise ValueError('手型变化后夹持关系未确认；请重新估计并确认当前目标已夹持')
        if fingerprint(self.selected_profile(require_calibration=False))!=plan.get('profile_sha256'):
            raise ValueError('执行约束/抓取标定变化，请重新预检')
        if fingerprint(json.loads(self.assembly_path.read_text()))!=plan.get('assembly_file_sha256'):
            raise ValueError('装配标定配置变化，请重新预检')
        if plan.get('collision_checked'):
            from eai_app.planning.scene import scene_digest
            if scene_digest(self.runtime_scene())!=plan.get('scene_sha256'):
                raise ValueError('场景/附着物变化，请重新预检')
            hands=self.runtime_hands()
            if any(np.max(abs(np.asarray(hands[s])-plan['hands'][s]))>.03 for s in hands):
                raise ValueError('手型变化，请重新进行碰撞预检')
        if not self.is_simulation:
            if not plan.get('collision_checked'):raise ValueError('缺少碰撞复检')
            from grasp_perception import require_verified
            if plan.get('stage')!='restore':
                held=self.world.attached.get(plan.get('arm_side'))
                action=self._workflow_action() or {}
                if getattr(self,'_program_active',False):action={'role':self._program_observation_role}
                if held and action.get('role')==held['role']:
                    feedback=self.feedback()
                    evidence={'observation_id':self.observation['observation_id'],'verified_at':time.time(),
                              'kind':'confirmed_attachment_fk','urdf_sha256':feedback['urdf_sha256']}
                    self._runtime_verification=evidence
                else:evidence=self.verify_runtime_target()
                require_verified(self.observation,evidence)
