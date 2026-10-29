"""Nonblocking UI proxy for the independently clocked execution process."""

from eai_app.paths import PROJECT_ROOT
import os
from pathlib import Path
import subprocess
import time
import uuid
import json
import shutil
import numpy as np
from eai_app.adapters.robot.execution_service import atomic_json


class ProcessRunner:
    def __init__(self, request, directory, display=None):
        from eai_app.development.source_guard import assert_sources_current, source_revision
        assert_sources_current()
        request = dict(request, code_revision=source_revision())
        self.code_revision = request['code_revision']
        self.directory=Path(directory)/uuid.uuid4().hex;self.directory.mkdir(parents=True)
        self.plan=request.get('plan') or {};self.side=request.get('side') or self.plan.get('arm_side')
        self.plan['execution_directory']=str(self.directory)
        self.rows=np.asarray(self.plan.get('joint_waypoints',[]));self.times=np.asarray(self.plan.get('times_s',[]))
        self.state,self.message='starting','独立执行服务启动中'
        self.sequence=0;self.acknowledged=0;self.display=display;self.started=time.monotonic()
        self.startup_timeout=60. if request.get('kind')=='phone_assembly_sim' else 10.
        atomic_json(self.directory/'request.json',request)
        atomic_json(self.directory/'heartbeat.json',{'monotonic':time.monotonic()})
        root=PROJECT_ROOT
        snapshot=root/'.cache/mv6d_last_success.npz'
        if request.get('kind')!='phone_assembly_sim' and snapshot.is_file():
            try:os.link(snapshot,self.directory/'observation_snapshot.npz')
            except OSError:shutil.copyfile(snapshot,self.directory/'observation_snapshot.npz')
        environment=os.environ.copy()
        environment['PYTHONPYCACHEPREFIX']=str(root/'.cache/bytecode'/request['code_revision'])
        prefix=Path(request['profile']['planner_python']).absolute().parent.parent
        for cmeel in prefix.glob('lib/python*/site-packages/cmeel.prefix'):
            environment['LD_LIBRARY_PATH']=str(cmeel/'lib')+':'+environment.get('LD_LIBRARY_PATH','')
        self.log=(self.directory/'service.log').open('w')
        self.process=subprocess.Popen(['bash',str(root/'scripts/run_grasp_executor.sh'),
            request['profile']['planner_python'],str(self.directory)],env=environment,
            stdout=self.log,stderr=subprocess.STDOUT,start_new_session=True)
    def _command(self,action,**extra):
        if action == 'start':
            from eai_app.development.source_guard import assert_sources_current, verify_revision
            assert_sources_current()
            verify_revision(self.code_revision)
        self.sequence+=1
        try:atomic_json(self.directory/'command.json',dict(sequence=self.sequence,action=action,**extra))
        except OSError as exc:
            self.state,self.message='stop_unconfirmed','执行通信写入失败：'+str(exc)
            if self.process.poll() is None:self.process.terminate()
            return False
        return True
    def pause(self):
        if self._command('pause'):self.state='pausing'
    def resume(self):
        if self.state!='paused':raise ValueError('执行服务尚未确认暂停')
        if self._command('resume'):self.state='resuming'
    def stop(self,reason='操作员取消',failed=False):
        if self.state in ('complete','failed','stopped','stop_unconfirmed'):return
        if self._command('stop',reason=reason,failed=failed):self.state='stopping'
    def tick(self):
        try:atomic_json(self.directory/'heartbeat.json',{'monotonic':time.monotonic()})
        except OSError as exc:
            self.state,self.message='stop_unconfirmed','心跳写入失败：'+str(exc)
            if self.process.poll() is None:self.process.terminate()
            return
        try:
            status=json.loads((self.directory/'status.json').read_text())
            self.acknowledged=status.get('acknowledged_command',0)
            if self.acknowledged>=self.sequence or status['state'] in ('failed','stop_unconfirmed','complete','stopped'):
                self.state=status['state'];self.message=status['message']
            self.trajectory_elapsed=status.get('elapsed',0.)
            if self.display and status.get('feedback'):
                self.display(status['feedback']['q16'],status.get('hands',{}))
            if time.time()-status['stamp']>2.:
                self.state,self.message='stop_unconfirmed','执行服务反馈超时，停止未确认'
        except (OSError,ValueError):
            if time.monotonic()-self.started>self.startup_timeout:
                self.state,self.message='stop_unconfirmed','执行服务启动超时，停止未确认'
        if self.process.poll() is not None:
            self.log.close()
            if self.state not in ('complete','stopped','failed','stop_unconfirmed'):
                self.state,self.message='stop_unconfirmed','执行服务退出，停止未确认'
        elif self.state in ('complete','stopped','failed'):
            self.state='finishing'  # Do not release the hardware lease before process exit.
