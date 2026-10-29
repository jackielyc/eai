"""Task-focused UI for the shared phone assembly program."""
from __future__ import annotations
import json
from pathlib import Path
import time
import uuid
import numpy as np
from PyQt5.QtCore import Qt,QTimer,QSize
from PyQt5.QtGui import QPixmap,QImage
from PyQt5.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QLabel,QPushButton,QComboBox,
    QSpinBox,QDoubleSpinBox,QCheckBox,QProgressBar,QGroupBox,QTextEdit,QDialog,QDialogButtonBox)
from .program import AssemblyProgram,STEPS,STAGES

GOAL_LABELS=(('phone_inside','手机入盒'),('phone_oriented','手机朝向'),('lid_closed','盒盖盖合'),
             ('released','已释放'),('stable','稳定'),('tool_clear','工具撤离'))


class FrameView(QLabel):
    def __init__(self):
        super().__init__('加载场景后显示接触仿真画面')
        from PyQt5.QtWidgets import QSizePolicy
        self.setAlignment(Qt.AlignCenter);self.setMinimumHeight(280);self.setMaximumHeight(420)
        self.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Expanding)
        self.source=None;self.setStyleSheet('background:#18212b;color:#b6c8d8;border-radius:6px;')
    def sizeHint(self):return QSize(800,380)
    def set_frame(self,path):
        pixmap=QPixmap(str(path))
        if not pixmap.isNull():self.source=pixmap;self.resizeEvent(None)
    def set_image(self,bgr):
        rgb=np.ascontiguousarray(np.asarray(bgr)[:,:,:3][:,:,::-1]);h,w=rgb.shape[:2]
        self.source=QPixmap.fromImage(QImage(rgb.data,w,h,rgb.strides[0],QImage.Format_RGB888).copy());self.resizeEvent(None)
    def resizeEvent(self,event):
        if self.source is not None:self.setPixmap(self.source.scaled(self.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation))
        if event is not None:super().resizeEvent(event)


class PhoneAssemblyPanel(QWidget):
    def __init__(self,owner):
        super().__init__(owner);self.owner=owner;self.client=None;self.program=None;self.backend=None
        self.latest={};self.run_directory=None;self.frame_stamp=0;self.display_stamp=0;self.inspected_stage=None
        self.terminal_saved=False
        self.pending_view=None
        layout=QVBoxLayout(self);layout.setContentsMargins(0,6,0,6)
        self.intro=QLabel('目标：手机完整放入盒体，盒盖对齐盖合，释放后保持稳定。每个检查点通过后才推进。')
        self.intro.setWordWrap(True);layout.addWidget(self.intro)
        row=QHBoxLayout();self.mode=QComboBox()
        self.mode.addItem('MuJoCo 接触仿真','simulation');self.mode.addItem('真机 · 同一装配程序','hardware')
        self.mode.currentIndexChanged.connect(self.mode_changed);row.addWidget(self.mode)
        self.arm=QComboBox();self.arm.addItem('左臂','left');self.arm.addItem('右臂','right')
        self.arm.setCurrentIndex(self.arm.findData(owner.grasp_arm.currentData()))
        self.arm.currentIndexChanged.connect(lambda _index:owner.grasp_arm.setCurrentIndex(owner.grasp_arm.findData(self.arm.currentData())) if not self.simulation else None)
        row.addWidget(self.arm)
        self.seed=QSpinBox();self.seed.setRange(0,999999);row.addWidget(QLabel('场景种子'));row.addWidget(self.seed)
        self.randomize=QCheckBox('轻微扰动物体摆放');row.addWidget(self.randomize)
        self.speed=QDoubleSpinBox();self.speed.setRange(.5,8);self.speed.setValue(2);self.speed.setSuffix(' × 仿真速率');row.addWidget(self.speed)
        self.view=QComboBox();self.view.addItem('装配近景','close');self.view.addItem('机器人全景','robot')
        self.view.addItem('透视验收','inspect');self.view.setToolTip('透视只改变盒盖显示透明度，不改变物理状态和验收条件')
        self.view.currentIndexChanged.connect(self.change_view);row.addWidget(self.view)
        self.robot_button=QPushButton('查看机器人页')
        self.robot_button.clicked.connect(lambda:owner.open_robot_callback() if owner.open_robot_callback else None);row.addWidget(self.robot_button)
        layout.addLayout(row)
        actions=QHBoxLayout()
        self.load_button=QPushButton('加载 / 复位仿真场景');self.load_button.clicked.connect(self.load_scene);actions.addWidget(self.load_button)
        self.run_button=QPushButton('运行装配');self.run_button.clicked.connect(self.start);actions.addWidget(self.run_button)
        self.pause_button=QPushButton('暂停');self.pause_button.clicked.connect(self.pause_resume);actions.addWidget(self.pause_button)
        self.cancel_button=QPushButton('取消并保持');self.cancel_button.clicked.connect(self.cancel);actions.addWidget(self.cancel_button)
        self.confirm_button=QPushButton('确认当前夹持检查');self.confirm_button.clicked.connect(self.confirm_check);actions.addWidget(self.confirm_button)
        self.config_button=QPushButton('真机参数（待标定）…');self.config_button.clicked.connect(self.edit_hardware);actions.addWidget(self.config_button)
        layout.addLayout(actions)
        self.hardware_confirm=QCheckBox('已确认空手、手机和盒盖位于盒外，目标绑定/标定正确且工作区无未建模障碍')
        self.hardware_confirm.setVisible(False);layout.addWidget(self.hardware_confirm)
        self.status=QLabel('先加载仿真场景，然后运行装配。');self.status.setWordWrap(True)
        self.status.setStyleSheet('font-weight:600;padding:6px;background:#243447;color:#e6f1ff;border-radius:4px;');layout.addWidget(self.status)
        self.progress=QProgressBar();self.progress.setRange(0,len(STEPS));self.progress.setValue(0)
        self.progress.setFormat('%v / %m 步 · %p%');layout.addWidget(self.progress)
        self.criteria=QLabel('　｜　'.join(label+'：待检查' for _,label in GOAL_LABELS))
        self.criteria.setWordWrap(True);layout.addWidget(self.criteria)
        self.frame=FrameView();layout.addWidget(self.frame)
        self.detail=QLabel();self.detail.setWordWrap(True);layout.addWidget(self.detail)
        self.simulation_note='示例手机120×70×12mm，盒内132×82×25mm；盒体固定，双指接触代理，A2D运动学映射。真机参数独立标定。'
        self.note=QLabel(self.simulation_note)
        self.note.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.note.setWordWrap(True);layout.addWidget(self.note)
        self.timer=QTimer(self);self.timer.setInterval(50);self.timer.timeout.connect(self.tick);self.timer.start()
        self.update_controls();self.inspect_stage(0);self.inspected_stage=None

    @property
    def simulation(self):return self.mode.currentData()=='simulation'
    def is_current_task(self):
        from grasp_task_workflow import TASKS
        key=self.owner.task_combo.currentData()
        return key=='phone_box_assembly' or TASKS.get(key,{}).get('program_type')=='phone_assembly'
    def active(self):
        if self.client and self.client.state not in ('complete','failed','stopped','stop_unconfirmed'):return True
        return bool(self.program and (self.program.state in ('running','paused') or self.backend.busy))
    def external_busy(self):
        p=self.owner
        return not p._program_active and bool(p.runner or p.process or p.hand_runner or p.reserved)
    def mode_changed(self):
        if self.active() or self.external_busy():
            self.mode.blockSignals(True);self.mode.setCurrentIndex(self.owner.execution_mode.currentIndex());self.mode.blockSignals(False)
            return
        self.client=None;self.program=None;self.backend=None;self.latest={};self.terminal_saved=False
        self.owner._program_validator=None;self.owner._program_observation_role=None
        self.frame.source=None;self.frame.clear();self.frame.setText('加载仿真场景，或在真机模式查看现有相机画面')
        self.owner.execution_mode.setCurrentIndex(0 if self.simulation else 1)
        if not self.simulation:self.owner.grasp_arm.setCurrentIndex(self.owner.grasp_arm.findData(self.arm.currentData()))
        self.hardware_confirm.setVisible(not self.simulation);self.update_controls();self.refresh_visibility()
        self.note.setText(self.simulation_note if self.simulation else '真机使用现有视觉目标、实测TCP和睿研手预设；夹持检查需操作员确认，放置检查使用新视觉结果。')
        if not self.simulation:
            if self.owner.scene_display_callback:self.owner.scene_display_callback({'calibrated':False,'world':[]},[])
            self.status.setText('真机复用同一程序；先配置实测参数、绑定手机/盒体/盒盖目标，再核对现场。')
    def refresh_visibility(self):
        primary=self.is_current_task()
        advanced=self.owner.assembly_advanced_check.isChecked() if hasattr(self.owner,'assembly_advanced_check') else False
        self.setVisible(primary)
        if hasattr(self,'legacy_title'):
            self.legacy_title.setText('手机盒装配 · 程序化任务' if primary and not advanced else self.legacy_title_text)
            self.legacy_hint.setText('同一装配程序连接接触仿真与真机；按检查结果推进，点击流程图可查看子步骤。'
                                     if primary and not advanced else self.legacy_hint_text)
        if hasattr(self.owner,'assembly_advanced_body'):
            self.owner.assembly_advanced_check.setVisible(primary)
            self.owner.assembly_advanced_body.setVisible(not primary or advanced)
        for widget in (self.owner.workflow_next_button,self.owner.workflow_confirm_button,self.owner.workflow_all_button,
                       self.owner.workflow_auto_pause_button,self.owner.workflow_reset_button,self.owner.task_save_button):
            widget.setVisible(not primary or advanced or (widget is self.owner.task_save_button and not self.simulation))
        show_roles=not primary or not self.simulation or advanced
        for i in range(self.owner.task_role_row.count()):
            widget=self.owner.task_role_row.itemAt(i).widget()
            if widget:widget.setVisible(show_roles)
        self.owner.task_data_status.setVisible(show_roles)
        self.hardware_confirm.setVisible(primary and not self.simulation)
        self.arm.setVisible(primary and not self.simulation)
    def _reserve(self,active):
        self.owner._program_active=active
        if active:self.owner.set_reserved(True)
        elif self.owner.sequence.state!='stop_unconfirmed':self.owner.set_reserved(False)
        if not active:
            self.hardware_confirm.setChecked(False);self.owner.scene_confirm.setChecked(False)
            self.owner._workflow_auto_paused=False;self.owner._program_validator=None
        self.owner.refresh()
    def load_scene(self):
        if not self.simulation:return
        if self.external_busy():self.status.setText('分段动作仍在运行，请先停止并等待确认。');return
        if self.active():self.status.setText('先取消当前运行并等待停止，再复位场景。');return
        try:
            from grasp_execution_client import ProcessRunner
            self.owner.execution_mode.setCurrentIndex(0)
            self.owner.grasp_arm.setCurrentIndex(self.owner.grasp_arm.findData('right'))
            self.owner.simulation=None
            request={'kind':'phone_assembly_sim','execution_mode':'simulation',
                'profile':self.owner.selected_profile(require_calibration=False),
                'seed_value':self.seed.value(),'randomize':self.randomize.isChecked(),'realtime_factor':self.speed.value()}
            self.client=ProcessRunner(request,self.owner.root/'.cache/phone_assembly_runs')
            self.run_directory=self.client.directory;self.latest={};self.program=None;self.backend=None;self.frame_stamp=0
            self.terminal_saved=False
            self.inspected_stage=None
            self.pending_view=self.view.currentData()
            self.frame.source=None;self.frame.clear();self.frame.setText('正在加载物理场景…')
            self._reserve(True);self.status.setText('加载接触场景与A2D运动学模型…')
        except Exception as exc:self.status.setText('场景未加载：'+str(exc));self._reserve(False)
        self.update_controls()
    def start(self):
        try:
            if self.external_busy():raise ValueError('已有分段动作占用控制，请先停止')
            if self.simulation:
                if self.client is None or self.client.state!='idle':raise ValueError('先加载一个新的仿真场景')
                if self.client._command('start'):self.client.state='starting';self.pending_view=self.view.currentData()
                self.status.setText('装配程序启动中…');self.update_controls();return
            if self.active():raise ValueError('已有任务在运行')
            from .hardware_backend import HardwareAssemblyBackend
            if not self.hardware_confirm.isChecked():raise ValueError('请先核对空手、目标摆放、标定及工作区')
            self.owner.scene_confirm.setChecked(True)
            backend=HardwareAssemblyBackend(self.owner,self.owner.root/'config/phone_assembly_hardware.json')
            self.backend=backend;self.program=AssemblyProgram(backend,backend.config)
            self.run_directory=self.owner.root/'.cache/phone_assembly_hardware_runs'/uuid.uuid4().hex
            self.run_directory.mkdir(parents=True);(self.run_directory/'config.json').write_text(json.dumps(backend.config,ensure_ascii=False,indent=2))
            self.client=None;self._reserve(True);self.program.start()
            self.terminal_saved=False
        except Exception as exc:self.status.setText('任务未启动：'+str(exc))
        self.update_controls()
    def pause_resume(self):
        try:
            if self.client:
                if self.client.state=='paused':self.client.resume()
                elif self.client.state=='running':self.client.pause()
            elif self.program:
                if self.program.state=='paused':self.program.resume()
                else:self.program.pause()
        except Exception as exc:self.status.setText(str(exc))
    def cancel(self):
        if self.client:self.client.stop('操作员取消装配，保持当前仿真场景')
        if self.program and self.program.state in ('running','paused'):self.program.cancel()
        self.status.setText('正在取消；不自动松手、不自动返回初始位置。')
    def confirm_check(self):
        try:
            if self.backend:self.backend.confirm_verification()
        except Exception as exc:self.status.setText('检查未确认：'+str(exc))
    def change_view(self):
        if not self.client:return
        view=self.view.currentData()
        if self.client.process.poll() is not None:
            path=self.client.directory/f'final_{view}.png'
            if path.exists():self.frame.set_frame(path)
            return
        self.pending_view=view
        if self.client.state in ('idle','running','paused') and self.client.acknowledged>=self.client.sequence:
            self.client._command('view',view=view);self.pending_view=None
    def edit_hardware(self):
        if self.active():return
        path=self.owner.root/'config/phone_assembly_hardware.json'
        dialog=QDialog(self);dialog.setWindowTitle('手机盒装配实测参数；仿真数据不能用于真机');dialog.resize(850,650)
        layout=QVBoxLayout(dialog);editor=QTextEdit(path.read_text());layout.addWidget(editor)
        buttons=QDialogButtonBox(QDialogButtonBox.Save|QDialogButtonBox.Cancel);layout.addWidget(buttons)
        buttons.accepted.connect(dialog.accept);buttons.rejected.connect(dialog.reject)
        if dialog.exec_()!=QDialog.Accepted:return
        try:
            data=json.loads(editor.toPlainText())
            if data.get('hardware_calibrated'):
                from .config import validate_hardware
                validate_hardware(data)
            from grasp_execution_service import atomic_json
            atomic_json(path,data);self.status.setText('任务参数已保存；真机仍须通过场景、反馈、IK和碰撞检查。')
        except Exception as exc:self.status.setText('参数未保存：'+str(exc))
    def inspect_stage(self,index):
        self.inspected_stage=index
        labels=[label for _,label,group in STEPS if group==index]
        self.detail.setText(STAGES[index]+'：'+' → '.join(labels))
    def workflow_text(self):return self.status.text()
    def update_controls(self):
        state=self.client.state if self.client else (self.program.state if self.program else 'empty')
        moving=state in ('running','starting','pausing','resuming','finishing','stopping')
        self.load_button.setEnabled(self.simulation and not self.active())
        self.run_button.setEnabled((self.simulation and state=='idle') or (not self.simulation and not self.active()))
        self.pause_button.setEnabled(state in ('running','paused'));self.pause_button.setText('继续' if state=='paused' else '暂停')
        self.cancel_button.setEnabled(self.active());self.mode.setEnabled(not self.active())
        self.hardware_confirm.setEnabled(not self.active())
        self.arm.setEnabled(not self.active())
        for widget in (self.seed,self.randomize,self.speed):widget.setEnabled(self.simulation and not self.active())
        self.config_button.setEnabled(not self.active())
        if self.external_busy():
            for widget in (self.load_button,self.run_button,self.mode,self.arm,self.config_button,self.hardware_confirm):widget.setEnabled(False)
        self.confirm_button.setVisible(not self.simulation)
        self.confirm_button.setEnabled(bool(self.backend and self.backend.needs_operator and not self.backend.vision_pending
                                             and not self.backend.operator_confirmed and self.program.state=='running'))
    def _display(self,status):
        index=int(status.get('index',0));group=int(status.get('group',0));state=status.get('state','starting')
        self.latest=status;self.status.setText(status.get('message',state));self.progress.setValue(index)
        goal=status.get('goal',{})
        self.criteria.setText('　｜　'.join(label+'：'+('✓' if goal.get(key) else '待检查') for key,label in GOAL_LABELS))
        self.owner._workflow_completed_steps=set(range(min(group,3)))
        self.owner.task_flow.set_completed_steps(self.owner._workflow_completed_steps);self.owner.task_flow.set_current(group)
        self.owner.task_flow.set_runtime_status(self.status.text(),'failed' if state in ('failed','stop_unconfirmed') else 'paused' if state=='paused' else 'active')
        self.owner._workflow_step=min(group,2);self.owner._workflow_finished=state=='complete'
        self.owner._workflow_error=status.get('message','') if state in ('failed','stop_unconfirmed') else ''
        self.owner._workflow_auto_paused=state=='paused'
        self.owner.message=self.status.text()
        if index<len(STEPS) and self.inspected_stage is None:self.detail.setText(f'{index+1}/{len(STEPS)}　'+STEPS[index][1]+'　｜　'+STAGES[min(group,2)])
        elif index>=len(STEPS) and self.inspected_stage is None:self.detail.setText(f'{len(STEPS)}/{len(STEPS)}　全部动作与验收已完成')
        if self.client and self.simulation:
            frame=self.client.directory/'frame.png'
            if frame.exists() and frame.stat().st_mtime_ns!=self.frame_stamp:
                self.frame_stamp=frame.stat().st_mtime_ns;self.frame.set_frame(frame)
            if status.get('render_error') and not frame.exists():self.frame.setText('渲染不可用：'+status['render_error'])
            q=(status.get('feedback') or {}).get('q16')
            if q is not None and time.monotonic()-self.display_stamp>.1:
                sim=self.owner.simulation_backend();sim.hands=status.get('hands',{});sim.publish(q)
                obs=status.get('observation',{});cfg=json.loads((self.client.directory/'config.json').read_text())
                world=[]
                sizes={'phone':cfg['phone_size'],'box':[cfg['box_inner_size'][0]+2*cfg['wall'],cfg['box_inner_size'][1]+2*cfg['wall'],cfg['box_inner_size'][2]],
                       'lid':[cfg['lid_inner_size'][0]+2*cfg['lid_wall'],cfg['lid_inner_size'][1]+2*cfg['lid_wall'],cfg['lid_skirt']+cfg['lid_wall']]}
                for name,T in obs.get('objects',{}).items():
                    T=np.asarray(T,float).copy()
                    if name=='box':
                        T[:3,3]+=T[:3,:3]@np.array([0,0,(cfg['wall']+cfg['box_inner_size'][2])/2])
                        sizes[name][2]=cfg['wall']+cfg['box_inner_size'][2]
                    elif name=='lid':T[:3,3]+=T[:3,:3]@np.array([0,0,-cfg['lid_skirt']/2])
                    world.append({'id':name,'T_base_object':T.tolist(),'size_m':sizes[name]})
                if self.owner.scene_display_callback:self.owner.scene_display_callback({'calibrated':False,'source':'simulation_task','world':world},[])
                self.display_stamp=time.monotonic()
        if state in ('complete','failed','stopped','stop_unconfirmed'):
            try:directory=str(self.run_directory.relative_to(self.owner.root))
            except (ValueError,AttributeError):directory=str(self.run_directory)
            self.note.setText('运行记录：'+directory+'（悬停查看完整目录）')
            self.note.setToolTip(str(self.run_directory)+'\n仿真接触工具为代理；真机参数需独立标定。')
    def tick(self):
        if not self.is_current_task() and not self.active():return
        if not self.simulation and self.owner.assembly_live_frame_callback:
            try:
                live=self.owner.assembly_live_frame_callback()
                if live is not None:
                    image,stamp=live
                    if stamp!=self.frame_stamp:self.frame_stamp=stamp;self.frame.set_image(image)
                else:self.frame.source=None;self.frame.setText('等待新鲜头部画面；不会用历史图像冒充当前现场')
            except Exception:self.frame.source=None;self.frame.setText('实时画面暂不可用')
        if self.terminal_saved:self.update_controls();return
        try:
            if self.client:
                self.client.tick();path=self.client.directory/'status.json'
                if self.pending_view and self.client.state in ('idle','running','paused') and self.client.acknowledged>=self.client.sequence:self.change_view()
                if path.exists():
                    status=json.loads(path.read_text());status['state']=self.client.state
                    self._display(status)
                elif self.client.state in ('failed','stop_unconfirmed'):
                    log=self.client.directory/'service.log'
                    detail=log.read_text()[-600:] if log.exists() else ''
                    self._display({'state':self.client.state,'message':'仿真服务启动失败：'+(detail or self.client.message),
                                   'index':0,'group':0,'goal':{'success':False}})
                if self.client.state in ('complete','failed','stopped','stop_unconfirmed'):
                    self._reserve(False)
                    if self.client.process.poll() is not None:self.change_view();self.pending_view=None
                    self.terminal_saved=True
            elif self.program:
                if self.program.state=='paused' or self.program.state=='cancelled':self.backend.tick()
                else:self.program.tick(.05)
                if self.program.state=='running' and self.backend.pause_required:
                    self.program.pause();self.program.message=self.backend.pause_reason or '执行服务已暂停'
                status=self.program.status();self._display(status)
                if self.program.state in ('complete','failed','cancelled') and not self.backend.busy:
                    from grasp_execution_service import atomic_json
                    atomic_json(self.run_directory/'result.json',status);atomic_json(self.run_directory/'events.json',self.program.events)
                    self.owner._workflow_auto_paused=False;self._reserve(False)
                    self.terminal_saved=True
            self.update_controls()
        except Exception as exc:
            self.status.setText('装配监控异常：'+str(exc))
            if self.active():self.cancel()


def install_phone_console(owner,layout):
    # Retain the existing widgets and callbacks inside a collapsible expert section.
    panel=PhoneAssemblyPanel(owner);owner.phone_program_panel=panel
    panel.legacy_title=layout.itemAt(0).widget();panel.legacy_hint=layout.itemAt(1).widget()
    panel.legacy_title_text=panel.legacy_title.text();panel.legacy_hint_text=panel.legacy_hint.text()
    layout.insertWidget(5,panel)
    advanced=QWidget();advanced_layout=QVBoxLayout(advanced);advanced_layout.setContentsMargins(0,0,0,0)
    while layout.count()>8:
        item=layout.takeAt(8)
        if item.widget():advanced_layout.addWidget(item.widget())
        elif item.layout():advanced_layout.addLayout(item.layout())
        else:advanced_layout.addItem(item)
    owner.assembly_advanced_check=QCheckBox('高级分段调试 / 标定（独立于任务程序）')
    owner.assembly_advanced_body=advanced;layout.addWidget(owner.assembly_advanced_check);layout.addWidget(advanced)
    owner.assembly_advanced_check.toggled.connect(lambda _checked:panel.refresh_visibility())
    panel.mode.blockSignals(True);panel.mode.setCurrentIndex(owner.execution_mode.currentIndex());panel.mode.blockSignals(False)
    owner.execution_mode.currentIndexChanged.connect(lambda index:panel.mode.setCurrentIndex(index) if panel.mode.currentIndex()!=index else None)
    owner.grasp_arm.currentIndexChanged.connect(lambda _index:panel.arm.setCurrentIndex(panel.arm.findData(owner.grasp_arm.currentData())) if not panel.simulation else None)
    owner.profile_path.textChanged.connect(lambda _text:panel.arm.setCurrentIndex(panel.arm.findData(owner.grasp_arm.currentData())) if not panel.simulation else None)
    panel.refresh_visibility()
