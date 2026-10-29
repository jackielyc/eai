"""Nominal simulation fixture; these dimensions are never real-robot calibration."""
from __future__ import annotations
import copy
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


DEFAULT_SCENE = {
    'version': 1, 'source': 'simulation_example', 'hardware_calibrated': False,
    'table_z': .72, 'phone_size': [.12, .07, .012],
    'box_inner_size': [.132, .082, .025], 'wall': .002,
    'lid_inner_size': [.138, .088], 'lid_wall': .002, 'lid_skirt': .008,
    'phone_xy': [.55, -.32], 'box_xy': [.55, -.10], 'lid_xy': [.55, .13],
    'tool_speed_m_s': .12, 'approach_speed_m_s': .035,
    'clearance_m': .08, 'settle_s': .5, 'friction': 1.4, 'phone_mass_kg': .18,
    'force_limit_n': 15., 'torque_limit_nm': 1.,
    'release_height_m': .003,
    'tool_model': 'parallel_jaw_contact_proxy',
    'notes': '盒体放在定位工装；双指接触代理用于验证任务逻辑，不是睿研手接触参数标定。',
}


def scenario(seed=0, randomize=False):
    cfg=copy.deepcopy(DEFAULT_SCENE)
    if randomize:
        rng=np.random.default_rng(seed)
        for key in ('phone_xy','box_xy','lid_xy'):
            cfg[key]=(np.asarray(cfg[key])+rng.uniform(-.008,.008,2)).tolist()
        cfg['phone_yaw']=float(rng.uniform(-.04,.04));cfg['lid_yaw']=float(rng.uniform(-.04,.04))
    cfg['seed']=int(seed)
    return cfg


def xml_scene(cfg):
    z=cfg['table_z'];p=np.asarray(cfg['phone_size']);inner=np.asarray(cfg['box_inner_size'])
    wall=cfg['wall'];lip=cfg['lid_wall'];skirt=cfg['lid_skirt'];lid=np.asarray(cfg['lid_inner_size'])
    px,py=cfg['phone_xy'];bx,by=cfg['box_xy'];lx,ly=cfg['lid_xy']
    def yaw_quat(key):
        a=cfg.get(key,0.)/2
        return f'{np.cos(a)} 0 0 {np.sin(a)}'
    def geom(name,size,pos,rgba,mass):
        return f'<geom name="{name}" type="box" size="{" ".join(map(str,size))}" pos="{" ".join(map(str,pos))}" rgba="{rgba}" mass="{mass}"/>'
    box=[geom('box_floor',[inner[0]/2+wall,inner[1]/2+wall,wall/2],[0,0,wall/2],'0.75 0.78 0.84 1',.1)]
    for side in (-1,1):
        box.append(geom(f'box_x{side}',[wall/2,inner[1]/2+wall,inner[2]/2],
                        [side*(inner[0]+wall)/2,0,wall+inner[2]/2],'0.75 0.78 0.84 1',.03))
        box.append(geom(f'box_y{side}',[inner[0]/2,wall/2,inner[2]/2],
                        [0,side*(inner[1]+wall)/2,wall+inner[2]/2],'0.75 0.78 0.84 1',.03))
    cover=[geom('lid_plate',[lid[0]/2+lip,lid[1]/2+lip,lip/2],[0,0,0],'0.2 0.48 0.82 1',.045)]
    for side in (-1,1):
        cover.append(geom(f'lid_x{side}',[lip/2,lid[1]/2+lip,skirt/2],
                          [side*(lid[0]+lip)/2,0,-lip/2-skirt/2],'0.2 0.48 0.82 1',.003))
        cover.append(geom(f'lid_y{side}',[lid[0]/2,lip/2,skirt/2],
                          [0,side*(lid[1]+lip)/2,-lip/2-skirt/2],'0.2 0.48 0.82 1',.003))
    # Only the tool is connected to a mocap actuator. Objects are always free bodies.
    xml=f'''<mujoco model="phone_box_assembly">
      <compiler angle="radian"/><option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast" iterations="80" cone="elliptic" noslip_iterations="10"/>
      <visual><global offwidth="800" offheight="600"/><quality shadowsize="2048"/></visual>
      <default><geom friction="{cfg.get('friction',1.4)} 0.01 0.001" condim="4" solref="0.008 1" solimp="0.95 0.99 0.001"/>
        <joint damping="3" armature="0.001"/></default>
      <worldbody>
        <geom name="ground" type="plane" pos="0 0 0" size="3 3 .05" rgba=".13 .15 .18 1"/>
        <light pos=".4 -.3 2" dir="0 0 -1" diffuse=".8 .8 .8"/>
        <geom name="table" type="box" pos=".55 -.1 {z-.025}" size=".28 .42 .025" rgba=".35 .29 .23 1"/>
        <body name="box" pos="{bx} {by} {z}">{''.join(box)}</body>
        <body name="phone" pos="{px} {py} {z+p[2]/2+.0005}" quat="{yaw_quat('phone_yaw')}"><freejoint name="phone_free"/>
          {geom('phone_shell',p/2,[0,0,0],'.07 .09 .12 1',cfg.get('phone_mass_kg',.18))}
          <geom type="box" size="{p[0]/2-.003} {p[1]/2-.003} .0002" pos="0 0 {p[2]/2+.0001}" rgba=".1 .65 .8 1" contype="0" conaffinity="0" mass="0"/>
        </body>
        <body name="lid" pos="{lx} {ly} {z+skirt+lip/2+.0005}" quat="{yaw_quat('lid_yaw')}"><freejoint name="lid_free"/>{''.join(cover)}</body>
        <body name="tool_target" mocap="true" pos="{px} {py} {z+.16}"/>
        <body name="tool" pos="{px} {py} {z+.16}"><freejoint name="tool_free"/>
          <geom name="palm" type="box" size=".035 .07 .012" pos="0 0 .055" rgba=".8 .82 .84 1" mass=".25"/>
          <body name="jaw_left" pos="0 0 0"><joint name="jaw_left_slide" type="slide" axis="0 1 0" range=".012 .06"/>
            <geom name="jaw_left_pad" type="box" size=".024 .0015 .025" pos="0 0 .005" rgba=".15 .16 .18 1" mass=".035"/>
          </body>
          <body name="jaw_right" pos="0 0 0"><joint name="jaw_right_slide" type="slide" axis="0 -1 0" range=".012 .06"/>
            <geom name="jaw_right_pad" type="box" size=".024 .0015 .025" pos="0 0 .005" rgba=".15 .16 .18 1" mass=".035"/>
          </body>
          <site name="tool_tcp" size=".002" rgba="1 .3 .1 1"/>
        </body>
      </worldbody>
      <equality><weld body1="tool_target" body2="tool" solref=".004 1" solimp=".98 .999 .001"/></equality>
      <actuator>
        <position name="jaw_left_motor" joint="jaw_left_slide" kp="600" kv="12" ctrlrange=".012 .06" forcerange="-8 8"/>
        <position name="jaw_right_motor" joint="jaw_right_slide" kp="600" kv="12" ctrlrange=".012 .06" forcerange="-8 8"/>
      </actuator>
    </mujoco>'''
    if cfg.get('robot_visual_urdf'):xml=add_robot_visuals(xml,cfg['robot_visual_urdf'])
    return xml


def add_robot_visuals(xml,urdf_path):
    root=ET.fromstring(xml);assets=ET.SubElement(root,'asset');world=root.find('worldbody')
    source=ET.parse(urdf_path).getroot();package_root=Path(urdf_path).resolve().parents[3]
    for link in source.findall('link'):
        name=link.get('name')
        if name in ('Link_l_ext','Link_r_ext'):continue
        visuals=link.findall('visual')
        if not visuals:continue
        body=ET.SubElement(world,'body',name='robot_'+name,mocap='true')
        for i,visual in enumerate(visuals):
            mesh=visual.find('geometry/mesh')
            if mesh is None:continue
            file=mesh.get('filename');file=str(package_root/file[len('package://'):]) if file.startswith('package://') else str(Path(urdf_path).parent/file)
            mesh_name=f'robot_mesh_{name}_{i}'
            ET.SubElement(assets,'mesh',name=mesh_name,file=file,scale=mesh.get('scale','1 1 1'))
            origin=visual.find('origin');xyz='0 0 0' if origin is None else origin.get('xyz','0 0 0')
            r,p,y=np.asarray([float(v) for v in ('0 0 0' if origin is None else origin.get('rpy','0 0 0')).split()])/2
            cr,cp,cy=np.cos([r,p,y]);sr,sp,sy=np.sin([r,p,y])
            quat=[cr*cp*cy+sr*sp*sy,sr*cp*cy-cr*sp*sy,cr*sp*cy+sr*cp*sy,cr*cp*sy-sr*sp*cy]
            color=visual.find('material/color');rgba='.62 .66 .72 1' if color is None else color.get('rgba','.62 .66 .72 1')
            ET.SubElement(body,'geom',type='mesh',mesh=mesh_name,pos=xyz,quat=' '.join(map(str,quat)),
                          rgba=rgba,contype='0',conaffinity='0',mass='0',group='1')
    return ET.tostring(root,encoding='unicode')
