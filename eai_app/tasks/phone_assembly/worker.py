"""Isolated contact simulation worker and reproducible headless task runner."""
from __future__ import annotations
import json
import hashlib
import os
from pathlib import Path
import signal
import time
import numpy as np


def _build(config,urdf=None,robot_ik=False):
    from .mujoco_backend import MujocoAssemblyBackend
    from .program import AssemblyProgram
    if robot_ik:config=dict(config,robot_visual_urdf=urdf)
    backend=MujocoAssemblyBackend(config);mapping=None
    if robot_ik:
        from .kinematics import A2DMapping
        mapping=A2DMapping(urdf,backend.pose('tool'),config);backend.update_robot_visual(mapping)
    return backend,AssemblyProgram(backend,config),mapping


def _robot_state(backend,mapping):
    if mapping is None:return None,{}
    q=mapping.update(backend.pose('tool'),{name:backend.pose(name) for name in ('phone','lid')});backend.update_robot_visual(mapping)
    gap=float(np.mean([backend.data.qpos[backend.model.joint(name).qposadr[0]] for name in ('jaw_left_slide','jaw_right_slide')]))
    closed=float(np.clip((.06-gap)/.048,0,1))
    return q,{'left':[0.]*6,'right':[closed]*6}


def _image(backend,path,view='close'):
    from PIL import Image
    image=backend.render(view=view);temporary=Path(path).with_suffix('.tmp.png')
    Image.fromarray(image).save(temporary);os.replace(temporary,path)


def _manifest(directory,backend,mapping):
    import platform
    from eai_app.adapters.robot.execution_service import atomic_json
    data={'version':1,'python':platform.python_version(),'numpy':np.__version__,
          'mujoco':backend.mj.__version__,'hardware_commands':0,'tool_model':backend.config['tool_model'],
          'robot_mapping':'kinematic' if mapping else None,
          'code_sha256':{name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                         for name in ('program.py','scene.py','mujoco_backend.py','kinematics.py')}}
    if mapping:data['robot_geometry_sha256']=mapping.collision.geometry_hash
    atomic_json(Path(directory)/'environment.json',data)


def run_headless(config,directory,urdf=None,robot_ik=False,render=False):
    os.environ.setdefault('MUJOCO_GL','egl')
    from eai_app.adapters.robot.execution_service import atomic_json
    from .scene import xml_scene
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    backend,program,mapping=_build(config,urdf,robot_ik)
    _manifest(directory,backend,mapping)
    (directory/'scene.xml').write_text(xml_scene(backend.config));atomic_json(directory/'config.json',config)
    program.start();start=time.monotonic();q=None;hands={};last_index=-1
    try:
        with (directory/'trajectory.jsonl').open('w') as log:
            for i in range(15000):
                program.tick(.02)
                if i%5==0:
                    try:q,hands=_robot_state(backend,mapping)
                    except Exception as exc:program.state='failed';program.message=str(exc);backend.cancel()
                    log.write(json.dumps(dict(program.status(),robot_q16=q,hands=hands),ensure_ascii=False)+'\n')
                if render and program.index!=last_index:
                    if program.index in (0,7,13,26):_image(backend,directory/f'stage_{program.index:02d}.png')
                    last_index=program.index
                if program.state!='running':break
            result=program.status()
            result.update(physics='MuJoCo',tool_model=config['tool_model'],hardware_commands=0,
                          wall_seconds=time.monotonic()-start,object_pose_writes_after_reset=backend.object_write_count)
            if mapping:result['ik']={'max_position_error_m':mapping.max_position_error,'max_rotation_error_rad':mapping.max_rotation_error,
                'body_collision_samples':mapping.collision_samples,'mapping':'kinematic','sim_proxy_mount':mapping.T_link_tool.tolist()}
            atomic_json(directory/'result.json',result);atomic_json(directory/'events.json',program.events)
            if render:
                _image(backend,directory/'final.png')
                _image(backend,directory/'final_inspect.png','inspect')
                _image(backend,directory/'final_robot.png','robot')
            return result
    finally:backend.close()


def run_interactive(directory,request):
    if request.get('execution_mode')!='simulation':raise ValueError('物理仿真工作进程禁止使用真机模式')
    os.environ.setdefault('MUJOCO_GL','egl')
    from eai_app.adapters.robot.execution_service import atomic_json
    from .scene import scenario,xml_scene
    directory=Path(directory)
    config=scenario(request.get('seed_value',0),request.get('randomize',False))
    backend,program,mapping=_build(config,request['profile']['urdf_path'],True)
    _manifest(directory,backend,mapping)
    atomic_json(directory/'config.json',config);(directory/'scene.xml').write_text(xml_scene(backend.config))
    stop=[False]
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    command_sequence=0;frame_time=0.;view='close';view_dirty=True;ticks=0;q,hands=_robot_state(backend,mapping);last_logged_state=None
    speed=float(request.get('realtime_factor',1.));render_error=''
    try:
        with (directory/'trajectory.jsonl').open('w') as log:
            while True:
                start=time.monotonic()
                heartbeat=json.loads((directory/'heartbeat.json').read_text())['monotonic']
                if start-heartbeat>15 or stop[0]:
                    program.cancel();program.message='界面心跳丢失或进程停止，已停止仿真'
                elif start-heartbeat>1.5 and program.state=='running':
                    program.pause();program.message='界面响应延迟，物理仿真已暂停；恢复后可继续'
                path=directory/'command.json'
                if path.exists():
                    command=json.loads(path.read_text())
                    if command['sequence']>command_sequence:
                        command_sequence=command['sequence'];action=command['action']
                        if action=='start' and program.state=='idle':program.start()
                        elif action=='pause':program.pause()
                        elif action=='resume':program.resume()
                        elif action=='stop':program.cancel()
                        elif action=='view':view=command.get('view','close');view_dirty=True
                program.tick(.02);ticks+=1
                changed=program.state!=last_logged_state
                if ticks%5==0 or changed:
                    try:q,hands=_robot_state(backend,mapping)
                    except Exception as exc:program.state='failed';program.message=str(exc);backend.cancel()
                if view_dirty or (program.state=='running' and start-frame_time>.12) or changed or program.state in ('complete','failed','cancelled'):
                    try:_image(backend,directory/'frame.png',view)
                    except Exception as exc:render_error=str(exc)
                    frame_time=start
                    view_dirty=False
                status=program.status();status.update(state='stopped' if program.state=='cancelled' else program.state,
                    stamp=time.time(),acknowledged_command=command_sequence,
                    feedback={'q16':q} if q is not None else None,hands=hands,render_error=render_error,
                    simulation_kind='contact_physics',tool_model=config['tool_model'],hardware_commands=0)
                atomic_json(directory/'status.json',status)
                if (program.state=='running' and ticks%5==0) or changed or program.state in ('complete','failed','cancelled'):
                    log.write(json.dumps(status,ensure_ascii=False)+'\n');log.flush();last_logged_state=program.state
                if program.state in ('complete','failed','cancelled'):
                    atomic_json(directory/'result.json',dict(status,ik={'max_position_error_m':mapping.max_position_error,
                        'max_rotation_error_rad':mapping.max_rotation_error,'body_collision_samples':mapping.collision_samples,
                        'mapping':'kinematic','sim_proxy_mount':mapping.T_link_tool.tolist()},object_pose_writes_after_reset=backend.object_write_count))
                    atomic_json(directory/'events.json',program.events)
                    try:
                        for name in ('close','inspect','robot'):_image(backend,directory/f'final_{name}.png',name)
                    except Exception:pass
                    break
                time.sleep(max(.05 if program.state in ('idle','paused') else 0.,.02/max(.1,speed)-(time.monotonic()-start)))
    finally:backend.close()
