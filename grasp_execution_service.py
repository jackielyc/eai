"""Independent execution process. GUI supplies goals/heartbeat, never servo ticks."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import numpy as np
from grasp_execution import JointRunner, evaluate_wrench, read_feedback
from grasp_hand_execution import HandRunner


def json_safe(value):
    if isinstance(value,dict):return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_safe(v) for v in value]
    if isinstance(value,(float,np.floating)) and not np.isfinite(value):return None
    return value


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))
    os.replace(temporary,path)


class KinematicBackend:
    """No ROS imports, publishers, sockets or hardware fallback in simulation."""
    def __init__(self, request):
        from host_arm_ik import HostArmIK
        self.ik = HostArmIK(request['profile']['urdf_path'])
        self.q = list(request['seed'])
        self.hands = request.get('hands', {})
        self.digest = hashlib.sha256(Path(request['profile']['urdf_path']).read_bytes()).hexdigest()
    def spin(self): pass
    def enabled(self): return True
    def prepare(self): pass
    def publish(self,q): self.q=list(q)
    def hand(self,side,values): self.hands[side]=list(values)
    def hand_feedback(self,side): return {'stamp':time.time(),'positions':self.hands.get(side,[0.]*6)}
    def wrench(self,side): return None
    def feedback(self):
        pin=self.ik._pin
        pin.forwardKinematics(self.ik.model,self.ik.data,self.ik.q_from_list(self.q))
        pin.updateFramePlacements(self.ik.model,self.ik.data)
        stamp=time.time()
        result={'q16':self.q,'urdf_sha256':self.digest,'arm_received_at':stamp,'waist_received_at':stamp}
        for side,fid in (('left',self.ik._left_fid),('right',self.ik._right_fid)):
            T=self.ik.data.oMf[fid]
            result[side]={'xyz':T.translation.tolist(),'xyzw':pin.Quaternion(T.rotation).coeffs().tolist()}
        return result


class RosBackend:
    def __init__(self,request):
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile,DurabilityPolicy,ReliabilityPolicy
        from sensor_msgs.msg import JointState
        from std_msgs.msg import UInt32
        from genie_msgs.msg import ArmState
        import socket
        self.rclpy,self.JointState,self.UInt32=rclpy,JointState,UInt32
        self.hand_velocity=request.get('hand_velocity') or [2000.]*6
        self.hand_effort=request.get('hand_effort') or [800.]*6
        for vector in (self.hand_velocity,self.hand_effort):
            array=np.asarray(vector,float)
            if array.shape!=(6,) or not np.isfinite(array).all() or np.any(array<0):raise ValueError('手部原生速度/力参数无效')
        self.hand_velocity=[float(v) for v in self.hand_velocity];self.hand_effort=[float(v) for v in self.hand_effort]
        rclpy.init();self.node=Node('eai_grasp_executor_'+str(os.getpid()))
        self.feedback_path=request.get('feedback_path','/tmp/eai_host_fk_state.json')
        self.enables={'arm':False,'hand':False};self.hands={};self.forces={}
        self.control_mode=None;self.expect_model=False;self.mode_grace=0.
        self.joint_pub=self.node.create_publisher(JointState,'/wbc/target_joints',10)
        self.mode_pub=self.node.create_publisher(UInt32,'/control_mode',10)
        self.hand_pubs={side:[self.node.create_publisher(JointState,topic,10) for topic in
            (f'/ry_hand/{side}/set_angles',f'/wbc/{side}_ee_command')] for side in ('left','right')}
        qos=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE)
        for group in ('arm','hand'):
            self.node.create_subscription(UInt32,f'/{group}/enable_state',
                lambda msg,g=group:self.enables.update({g:bool(msg.data)}),qos)
        self.node.create_subscription(UInt32,'/control_mode',lambda msg:setattr(self,'control_mode',int(msg.data)),10)
        for side in ('left','right'):
            sensor_qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT,durability=DurabilityPolicy.VOLATILE)
            self.node.create_subscription(JointState,f'/ry_hand/{side}/joint_states',
                lambda msg,s=side:self._hand_state(s,msg),sensor_qos)
            self.node.create_subscription(ArmState,f'/hal/{side}_arm_data',
                lambda msg,s=side:self._wrench(s,msg),sensor_qos)
        self.sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);self.sock.setblocking(False)
        self.last_mode=0.
    def relay(self,payload):
        if os.environ.get('HAND_CMD_RELAY','1').lower() not in ('0','false','off','no'):
            self.sock.sendto(json.dumps(payload).encode(),
                (os.environ.get('HAND_CMD_RELAY_HOST','127.0.0.1'),int(os.environ.get('HAND_CMD_RELAY_PORT','19283'))))
    def _hand_state(self,side,msg):
        from arm_text_commands import HAND_JOINT_ORDER,canonical_hand_joint
        values={canonical_hand_joint(k):float(v) for k,v in zip(msg.name,msg.position)}
        source_stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
        if source_stamp>0 and source_stamp<=self.hands.get(side,{}).get('source_stamp',0):return
        if all(k in values for k in HAND_JOINT_ORDER):
            self.hands[side]={'positions':[values[k] for k in HAND_JOINT_ORDER],'stamp':time.time(),'source_stamp':source_stamp}
    def _wrench(self,side,msg):
        values=list(msg.force_data)
        source_stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
        if source_stamp>0 and source_stamp<=self.forces.get(side,{}).get('source_stamp',0):return
        self.forces[side]={'force':values[:3],'torque':values[3:6],
                          'force_state':int(msg.force_state),'force_coordinate':int(msg.force_coordinate),
                          'stamp':time.time(),'source_stamp':source_stamp}
    def spin(self):
        deadline=time.monotonic()+.005
        for _ in range(16):
            self.rclpy.spin_once(self.node,timeout_sec=0.)
            if time.monotonic()>deadline:break
    def enabled(self):
        mode_ok=not self.expect_model or time.monotonic()<self.mode_grace or self.control_mode==0
        return all(self.enables.values()) and mode_ok
    def mode(self,value):
        msg=self.UInt32();msg.data=value;self.mode_pub.publish(msg)
        self.relay({'type':'control_mode','mode':value});self.last_mode=time.monotonic()
    def prepare(self):
        self.expect_model=True;self.mode_grace=time.monotonic()+.5
        self.mode(99);self.mode(0)
    def feedback(self): return read_feedback(self.feedback_path)
    def hand_feedback(self,side): return self.hands.get(side,{'stamp':0,'positions':[]})
    def wrench(self,side): return self.forces.get(side)
    def publish(self,q):
        if not self.enabled(): raise ValueError('手臂/手使能丢失')
        msg=self.JointState();msg.header.stamp=self.node.get_clock().now().to_msg()
        msg.header.frame_id='base_link';msg.position=list(q)
        self.joint_pub.publish(msg)
        self.relay({'type':'wbc_joints','frame_id':'base_link','name':[],
                    'position':list(q),'velocity':[],'effort':[]})
    def hand(self,side,values):
        from arm_text_commands import HAND_JOINT_ORDER
        if not self.enabled(): raise ValueError('手部使能丢失')
        msg=self.JointState();msg.header.stamp=self.node.get_clock().now().to_msg()
        msg.header.frame_id=side+'_hand';msg.name=list(HAND_JOINT_ORDER)
        msg.position=list(values);msg.velocity=list(self.hand_velocity);msg.effort=list(self.hand_effort)
        for pub in self.hand_pubs[side]:pub.publish(msg)
        self.relay({'type':'hand','side':side,'name':msg.name,'position':list(values),
                    'velocity':msg.velocity,'effort':msg.effort})


def run(directory):
    directory=Path(directory);request=json.loads((directory/'request.json').read_text())
    if request.get('kind')=='phone_assembly_sim':
        from phone_assembly.worker import run_interactive
        return run_interactive(directory,request)
    simulation=request['execution_mode']=='simulation'
    if request['execution_mode'] not in ('simulation','hardware'):raise ValueError('未知执行方式')
    lock=None
    if not simulation:
        lock=open('/tmp/eai_grasp_execution.lock','w')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    backend=KinematicBackend(request) if simulation else RosBackend(request)
    stop_signal=[False]
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stop_signal.__setitem__(0,True))
    start=time.monotonic();runner=None;last_command=0;last_trace=0.;force_reason='';hand_preflight_done=False
    geometry_verified=False
    import platform
    atomic_json(directory/'environment.json',{'python':platform.python_version(),'numpy':np.__version__,
        'execution_mode':request['execution_mode'],'pid':os.getpid(),
        'io_backend':type(backend).__name__,
        'urdf_sha256':hashlib.sha256(Path(request['profile']['urdf_path']).read_bytes()).hexdigest(),
        'code_sha256':{name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                      for name in ('grasp_execution_service.py','grasp_execution.py','grasp_hand_execution.py','grasp_trajectory.py')}})
    side=request.get('side') or request.get('plan',{}).get('arm_side','right')
    sides=('left','right') if side=='both' or request.get('plan',{}).get('other_tcp_held') else (side,)
    def guard():
        if simulation or not request.get('wrench_guard',True): return None
        for arm in sides:
            violation,*_=evaluate_wrench(backend.wrench(arm),request['force_limit_n'],request['torque_limit_nm'],
                                         policy=request.get('force_policy'))
            if violation:return arm+': '+violation
    while True:
        loop=time.monotonic();backend.spin()
        alive=loop-float(json.loads((directory/'heartbeat.json').read_text())['monotonic']) < 1.5
        if runner is None:
            if stop_signal[0] or not alive:raise ValueError('启动期间界面心跳丢失，未执行')
            if (directory/'command.json').exists():
                early=json.loads((directory/'command.json').read_text())
                if early['action'] in ('stop','pause'):
                    atomic_json(directory/'status.json',{'state':'stopped' if early['action']=='stop' else 'paused',
                        'message':'执行尚未启动','stamp':time.time(),'acknowledged_command':early['sequence']})
                    if early['action']=='stop':return
                    time.sleep(.02);continue
                last_command=early['sequence']
            try:
                if not backend.enabled():raise ValueError('等待手臂/手使能反馈')
                if guard():raise ValueError(guard())
                backend.feedback()
                if not simulation:
                    from grasp_execution import fingerprint
                    from grasp_scene import scene_digest
                    if not request.get('collision_required'):raise ValueError('真机必须通过碰撞预检')
                    if fingerprint(json.loads(Path(request['scene_path']).read_text()))!=request['scene_file_sha256']:
                        raise ValueError('场景文件变化，拒绝旧计划')
                    if request['kind']=='arm':
                        plan=request['plan']
                        if not plan.get('collision_checked') or scene_digest(request['scene'])!=plan.get('scene_sha256'):
                            raise ValueError('计划与碰撞场景不匹配')
                        if not geometry_verified:
                            from grasp_scene import augmented_urdf,geometry_digest
                            accessory=Path(__file__).parent.parent/'psi-policy/assets/A2D_RuiYan_D405/A2D_RuiYan_D405.urdf'
                            if geometry_digest(augmented_urdf(request['profile']['urdf_path'],accessory))!=plan.get('geometry_sha256'):
                                raise ValueError('机器人/手/支架网格发生变化，请重新预检')
                            geometry_verified=True
                        if plan.get('stage')!='restore':
                            from grasp_perception import require_verified
                            require_verified({'observation_id':request['observation_id']},request.get('verification'),max_age=10.)
                if request['kind']=='hand':
                    from grasp_trajectory import exceeds
                    if exceeds(backend.feedback()['q16'],request['seed'],'start'):
                        raise ValueError('手部预检期间手臂位置变化')
                    if request.get('collision_required') and not hand_preflight_done:
                        from grasp_scene import CollisionScene
                        checker=CollisionScene(request['profile']['urdf_path'],request['scene'],
                            request['hands'],request.get('attachments',[]),stage=request.get('hand_operation',''))
                        hands=dict(request['hands']);begin=np.asarray(hands[side]);target=np.asarray(request['target'])
                        for fraction in np.linspace(0,1,max(2,int(np.max(abs(target-begin))/.02)+1)):
                            checker.fraction=float(fraction)
                            hands[side]=(begin*(1-fraction)+target*fraction).tolist()
                            checker.set_hands(hands);checker.check(request['seed'])
                        hand_preflight_done=True
                    backend.spin()
                    if time.monotonic()-json.loads((directory/'heartbeat.json').read_text())['monotonic']>=1.5:
                        raise ValueError('预检期间心跳丢失，未发送手部命令')
                    if exceeds(backend.feedback()['q16'],request['seed'],'start'):
                        raise ValueError('碰撞预检后手臂位置变化')
                    candidate=HandRunner(side,request['target'],lambda q:backend.hand(side,q),
                        lambda:backend.hand_feedback(side),backend.enabled,
                        timeout=request['profile'].get('hand_timeout_s',10.),
                        tolerance=request['profile'].get('hand_feedback_tolerance',.03))
                    measured,_=candidate._sample()
                    if np.max(abs(measured-np.asarray(request['hands'][side])))>.03:
                        raise ValueError('碰撞预检后手型变化，必须重新预检')
                    runner=candidate;runner.start()
                else:
                    if time.monotonic()-json.loads((directory/'heartbeat.json').read_text())['monotonic']>=1.5:
                        raise ValueError('启动运动前心跳丢失')
                    runner=JointRunner(request['plan'],backend.publish,backend.feedback,backend.enabled)
                    backend.prepare()
            except Exception as exc:
                if runner is not None:raise
                if loop-start>5:raise
                atomic_json(directory/'status.json',{'state':'starting','message':str(exc),'stamp':time.time()})
                time.sleep(.02);continue
        if stop_signal[0] or not alive:runner.stop('界面心跳丢失或进程停止请求',failed=True)
        command_path=directory/'command.json'
        if command_path.exists():
            command=json.loads(command_path.read_text())
            if command['sequence']>last_command:
                last_command=command['sequence'];action=command['action']
                if action=='stop':runner.stop(command.get('reason','操作员取消'),failed=command.get('failed',False))
                elif action=='pause':runner.pause()
                elif action=='resume':
                    violation=guard()
                    if request.get('collision_required'):
                        from grasp_execution import fingerprint
                        if fingerprint(json.loads(Path(request['scene_path']).read_text()))!=request['scene_file_sha256']:
                            violation='场景配置变化，请停止并重新预检'
                    if request['kind']=='hand':
                        from grasp_trajectory import exceeds
                        if exceeds(backend.feedback()['q16'],request['seed'],'start'):
                            violation='手部暂停期间手臂移动，请重新预检'
                    if violation:force_reason=violation
                    else:
                        try:runner.resume();force_reason=''
                        except Exception as exc:force_reason=str(exc)
        violation=guard()
        if request['kind']=='hand' and runner.state=='running':
            from grasp_trajectory import exceeds
            if exceeds(backend.feedback()['q16'],request['seed'],'start'):
                runner.stop('手部执行期间手臂发生移动',failed=True)
        if runner.state=='running' and violation:
            runner.pause();force_reason='六维力保护暂停：'+violation
        runner.tick()
        try:feedback=backend.feedback()
        except Exception:feedback=None
        status={'state':runner.state,'message':force_reason or runner.message,'stamp':time.time(),
                'acknowledged_command':last_command,'feedback':feedback,
                'hands':backend.hands,'elapsed':getattr(runner,'trajectory_elapsed',0.)}
        if isinstance(runner,JointRunner):status['command_q16']=runner.last_command.tolist()
        else:
            status['command_hand']={'side':side,'positions':(runner.target if runner.state in ('running','complete') else getattr(runner,'hold',runner.target)).tolist()}
        try:atomic_json(directory/'status.json',json_safe(status))
        except OSError:runner.stop('执行状态无法写入，停止后续轨迹',failed=True)
        if loop-last_trace>=.05 or runner.state in ('complete','stopped','failed','stop_unconfirmed'):
            try:
                with (directory/'telemetry.jsonl').open('a') as stream:
                    stream.write(json.dumps(json_safe(dict(status,wrenches={s:backend.wrench(s) for s in sides})),allow_nan=False)+'\n')
            except OSError:runner.stop('反馈记录无法写入，停止后续轨迹',failed=True)
            last_trace=loop
        if runner.state in ('complete','stopped','failed','stop_unconfirmed'):break
        time.sleep(max(0.,.02-(time.monotonic()-loop)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--directory')
    parser.add_argument('--check-environment',action='store_true')
    parser.add_argument('--probe-feedback',action='store_true')
    args=parser.parse_args()
    if args.check_environment:
        import rclpy
        from sensor_msgs.msg import JointState
        from std_msgs.msg import UInt32
        from genie_msgs.msg import ArmState
        ArmState.__class__.__import_type_support__()
        print(json.dumps({'ok':True,'ros_initialized':False,'hardware_commands':0}))
        raise SystemExit(0)
    if args.probe_feedback:
        backend=RosBackend({'feedback_path':os.environ.get('EAI_HOST_FK_STATE','/tmp/eai_host_fk_state.json')})
        try:
            deadline=time.monotonic()+2
            while time.monotonic()<deadline:backend.spin();time.sleep(.01)
            try:backend.feedback();fk_ready=True
            except Exception:fk_ready=False
            print(json.dumps({'probe_only':True,'hardware_commands':0,'fk_feedback_ready':fk_ready,
                'enables':backend.enables,'hands_received':list(backend.hands),'wrenches_received':list(backend.forces)}))
        finally:backend.node.destroy_node();backend.rclpy.shutdown();backend.sock.close()
        raise SystemExit(0)
    if not args.directory:parser.error('--directory required')
    try:run(args.directory)
    except Exception as exc:
        atomic_json(Path(args.directory)/'status.json',{'state':'stop_unconfirmed',
            'message':'独立执行服务异常：'+str(exc),'stamp':time.time()})
        raise
