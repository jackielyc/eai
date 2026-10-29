"""Explicit orientation proposals and endpoint ranking; never robot execution."""
import copy
import time
import numpy as np


def orientation_proposals(candidate, angles=(-15.,0.,15.)):
    result=[]
    for degrees in angles:
        if not np.isfinite(degrees) or abs(degrees)>180:raise ValueError('候选角度无效')
        angle=np.deg2rad(degrees);R=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
        item=copy.deepcopy(candidate);T=np.asarray(item['T_camera_grasp'],float).copy()
        T[:3,:3]=T[:3,:3]@R;item['T_camera_grasp']=T.tolist()
        item['orientation_offset_deg']=float(degrees)
        item['note']='人工朝向附近采样；未证明抓取接触有效，需预览核对'
        result.append(item)
    return result


def rank(request):
    from host_arm_ik import HostArmIK,_JOINT_NAMES
    from click_pose import validate_transform
    ik=HostArmIK(request['profile']['urdf_path']);pin=ik._pin
    seed=np.asarray(request['start_joints']);side=request['profile']['arm_side']
    indices=[ik._q_idx[name] for name in _JOINT_NAMES]
    lower,upper=ik.model.lowerPositionLimit[indices],ik.model.upperPositionLimit[indices]
    span=np.maximum(upper-lower,1e-6);rows=[];deadline=time.monotonic()+30
    for i,goal in enumerate(request['candidate_goals']):
        if time.monotonic()>deadline:break
        T=validate_transform(goal)
        q=ik.solve_coordinated(seed,moving_side=side,target=pin.SE3(T[:3,:3],T[:3,3]),
            allow_waist=request.get('allow_waist',False),max_iters=250,attempts=3,deadline=deadline)
        if q is None:
            rows.append({'index':i,'ok':False,'reason':'终点IK未收敛'});continue
        q=np.asarray(q);margin=float(np.min(np.minimum(q-lower,upper-q)/span))
        fid=ik._left_fid if side=='left' else ik._right_fid
        J=pin.computeFrameJacobian(ik.model,ik.data,ik.q_from_list(q),fid,pin.LOCAL)
        variable=list(range(16)) if request.get('allow_waist') else (list(range(2,9)) if side=='left' else list(range(9,16)))
        cols=[ik.model.joints[ik.model.getJointId(_JOINT_NAMES[k])].idx_v for k in variable]
        singular=float(np.linalg.svd(J[:,cols],compute_uv=False)[-1])
        travel=float(np.linalg.norm((q-seed)/span))
        rows.append({'index':i,'ok':True,'score':float(travel-.5*margin-.1*singular),
                     'joint_margin':margin,'min_singular_value':singular,'travel':travel})
    rows.sort(key=lambda row:(not row['ok'],row.get('score',float('inf'))))
    return {'ok':True,'observation_id':request['observation_id'],'rankings':rows,
            'note':'仅终点可达性排序，不代表全路径/碰撞/抓取质量通过'}
