"""Object-relative manipulation state. A command is not evidence of attachment."""
import copy
import numpy as np
from click_pose import validate_transform
from grasp_execution import rotation_from_quaternion,fingerprint


def tcp_matrix(feedback,side):
    pose=feedback[side];T=np.eye(4)
    T[:3,:3]=rotation_from_quaternion(pose['xyzw']);T[:3,3]=pose['xyz']
    return validate_transform(T)


def object_matrix(observation):
    return validate_transform(observation.get('T_base_object',
        np.asarray(observation['T_base_camera'])@np.asarray(observation['object_pose']['T_camera_object'])))


class ManipulationWorld:
    def __init__(self):self.attached={};self.placed={};self.last_placement={}
    def attach_confirmed(self,side,role,target_id,observation,feedback,replace=False):
        if side in self.attached and not replace:raise ValueError('该手已有附着物，不能覆盖夹持状态')
        T=np.linalg.inv(tcp_matrix(feedback,side))@object_matrix(observation)
        size=np.asarray(observation['object_pose']['extents_m'],float)
        if size.shape!=(3,) or not np.isfinite(size).all() or np.any(size<=0):
            raise ValueError('附着物尺寸无效')
        self.attached[side]={'id':target_id,'role':role,'verified':True,'link':'Link7_l' if side=='left' else 'Link7_r',
            'T_link_object':T.tolist(),'size_m':size.tolist(),
            'touch_links':['hand'+('1' if side=='left' else '2')+'_link_'+suffix for suffix in
                ('0','1_1','1_2','1_3','2_1','2_2','3_1','3_2','4_1','4_2','5_1','5_2')]}
    def detach(self,side,feedback):
        item=self.attached.pop(side,None)
        if item:
            item=copy.deepcopy(item)
            item['T_base_object']=(tcp_matrix(feedback,side)@np.asarray(item['T_link_object'])).tolist()
            self.placed[item['id']]=item
        return item
    def placement_preview(self,preview,task,side,reference_role,reference_observation,settings,hardware=False):
        held=self.attached.get(side)
        if not held:
            cached=self.last_placement.get((side,reference_role))
            return copy.deepcopy(cached) if cached else preview
        if held['role']==reference_role:return preview
        key=f'{task}:{held["role"]}:{reference_role}'
        relation=settings.get('relations',{}).get(key)
        if not relation or not relation.get('calibrated'):
            raise ValueError('装配关系待标定：'+key+'；需要物体相对目标的放置位置和方向')
        expected=relation.get('pose_method')
        actual=reference_observation.get('pose_method',reference_observation.get('object_pose',{}).get('method'))
        if expected and expected!=actual:raise ValueError('装配关系的位姿方法与当前输入不一致，请重新标定关系')
        T_ref_object=validate_transform(relation['T_reference_object'])
        contact=object_matrix(reference_observation)@T_ref_object@np.linalg.inv(np.asarray(held['T_link_object']))
        direction=np.asarray(relation.get('approach_axis_reference',[0,0,1]),float)
        if direction.shape!=(3,) or not np.isclose(np.linalg.norm(direction),1):raise ValueError('装配接近方向必须是单位向量')
        direction=object_matrix(reference_observation)[:3,:3]@direction
        distance=float(relation.get('clearance_m',.05))
        if not .01<=distance<=.3:raise ValueError('装配撤离距离需在0.01~0.3m')
        above=contact.copy();above[:3,3]+=direction*distance
        result=copy.deepcopy(preview)
        result['goals']={'pregrasp':above.tolist(),'approach':contact.tolist(),'lift':above.tolist()}
        result['operation']='place';result['assembly_relation']=key
        result['assembly_sha256']=fingerprint(relation);result['id']=fingerprint(result)
        self.last_placement[(side,reference_role)]=copy.deepcopy(result)
        return result


def expand_skill(step):
    """Compile explicit pick/place/align/insert/verify skills to existing actions."""
    kind=step['kind'];role=step.get('role')
    if kind=='verify':return dict(step,kind='confirm')
    if kind not in ('pick','place','align','insert'):return step
    motion=lambda label,stage:dict(label=label,kind='motion',role=role,stage=stage,operation=kind)
    if kind=='pick':
        actions=[{'label':'张开手','kind':'hand','role':role,'open':True},motion('预抓取','pregrasp'),
            motion('接近','approach'),{'label':'闭合','kind':'hand','role':role,'open':False},
            {'label':'确认夹持','kind':'confirm'},motion('抬起','lift')]
    elif kind=='place':
        actions=[motion('移动到放置上方','pregrasp'),motion('放置','approach'),
            {'label':'释放','kind':'hand','role':role,'open':True},motion('撤离','lift'),
            {'label':'确认放置结果','kind':'confirm'}]
    else:
        actions=[motion('对齐上方','pregrasp'),motion('沿标定方向接近','approach'),
                 {'label':'确认'+step['label'],'kind':'confirm'}]
    return dict(step,kind='sequence',skill=kind,actions=actions)


def alignment_preview(preview,task,role,observation,settings):
    key=f'{task}:{role}:align';entry=settings.get('tcp_targets',{}).get(key)
    if not entry or not entry.get('calibrated'):raise ValueError('对齐关系待标定：'+key)
    target=object_matrix(observation)@validate_transform(entry['T_reference_tcp'])
    direction=np.asarray(entry.get('approach_axis_reference',[0,0,1]),float)
    distance=float(entry.get('clearance_m',.05))
    if direction.shape!=(3,) or not np.isclose(np.linalg.norm(direction),1) or not .01<=distance<=.3:
        raise ValueError('对齐接近方向/撤离距离无效')
    above=target.copy();above[:3,3]+=object_matrix(observation)[:3,:3]@direction*distance
    result=copy.deepcopy(preview);result.update(operation='align',assembly_relation=key,
        goals={'pregrasp':above.tolist(),'approach':target.tolist(),'lift':above.tolist()})
    result['id']=fingerprint(result);return result
