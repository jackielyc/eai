"""Separate nominal simulation settings from measured hardware calibration."""
import copy
import json
from pathlib import Path
import numpy as np
from eai_app.domain.geometry.transforms import validate_transform
from eai_app.domain.grasp.execution import hand_positions


def validate_hardware(config):
    missing=[]
    if config.get('source')!='hardware_calibration' or config.get('hardware_calibrated') is not True:
        missing.append('真机标定声明')
    for key,length in (('phone_size',3),('box_inner_size',3),('lid_inner_size',2)):
        try:
            values=np.asarray(config[key],float)
            if values.shape!=(length,) or not np.isfinite(values).all() or np.any(values<=0):raise ValueError()
        except Exception:missing.append(key)
    for key in ('wall','lid_wall','table_z','clearance_m','tool_speed_m_s','approach_speed_m_s'):
        try:
            if not np.isfinite(config[key]) or config[key]<=0:raise ValueError()
        except Exception:missing.append(key)
    try:
        release=float(config['release_height_m'])
        if not np.isfinite(release) or not 0<=release<=.1:raise ValueError()
    except Exception:missing.append('release_height_m')
    for key in ('T_link_tool','T_box_frame_cavity'):
        try:validate_transform(config[key])
        except Exception:missing.append(key)
    for role in ('phone','lid'):
        for key in ('placement_T_reference_object','grasp_T_object_tool'):
            try:validate_transform(config[key][role])
            except Exception:missing.append(key+'.'+role)
        for kind in ('open','close','release'):
            try:hand_positions(config['hand_presets'][role][kind])
            except Exception:missing.append('hand_presets.'+role+'.'+kind)
        for kind in ('velocity','effort'):
            try:
                values=np.asarray(config['hand_presets'][role][kind],float)
                if values.shape!=(6,) or not np.isfinite(values).all() or np.any(values<0) or not np.any(values>0):raise ValueError()
            except Exception:missing.append('hand_presets.'+role+'.'+kind)
    if missing:raise ValueError('待实测标定：'+'、'.join(missing))
    if config.get('tool_model')!='ruiyan_6dof_v1':raise ValueError('真机必须配置实际睿研手模型，禁止使用仿真代理')
    if config.get('pose_method') not in ('pca','foundationpose','any6d'):raise ValueError('需明确标定所用位姿方法')
    return config


def load_hardware(path,assembly_path=None):
    config=json.loads(Path(path).read_text())
    if assembly_path:
        relations=json.loads(Path(assembly_path).read_text()).get('relations',{})
        for role in ('phone','lid'):
            relation=relations.get('phone_box_assembly:'+role+':box',{})
            if relation.get('calibrated') and config.get('placement_T_reference_object',{}).get(role) is None:
                if relation.get('pose_method') and relation['pose_method']!=config.get('pose_method'):
                    raise ValueError('装配关系与任务标定使用了不同位姿方法')
                config.setdefault('placement_T_reference_object',{})[role]=copy.deepcopy(relation['T_reference_object'])
    return validate_hardware(config)
