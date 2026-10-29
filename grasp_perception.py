"""Observation provenance and live-scene revalidation, separate from GUI state."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from grasp_execution import fingerprint


def calibration_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def thumbnail(image):
    a=np.asarray(image)
    if a.ndim!=3 or a.shape[2]<3:raise ValueError('复核需要彩色图像')
    # Fixed sample coordinates work without a cv2 dependency.
    y=np.linspace(0,a.shape[0]-1,64).astype(int);x=np.linspace(0,a.shape[1]-1,64).astype(int)
    return a[y[:,None],x[None,:],:3].astype(float)


def capture_provenance(buffer, views, calibration_path, depth_source, tf_buffer=None, selected_depths=None):
    result={'version':1,'depth_source':depth_source,'calibration_sha256':calibration_digest(calibration_path),
            'frames':{},'captured_at':time.time(),'issues':[],'origin':'live'}
    for label,(color_topic,depth_topic,image) in views.items():
        if image is None:continue
        try:
            with buffer.lock:
                frames=list(buffer.frames.get(color_topic,()))
            frame=next((f for f in reversed(frames) if f.displayed is image or f.image is image),None)
            if frame is None or time.monotonic()-frame.received>.8 or frame.stamp<=0:
                raise ValueError('缺少匹配的新鲜彩色采集帧')
            item={'topic':color_topic,'frame_id':frame.frame_id,'stamp':frame.stamp,
                  'received':frame.received,'thumbnail':thumbnail(image).tolist()}
            if depth_source!='lingbot':
                calibration=json.loads(Path(calibration_path).read_text()).get('cameras',{}).get(color_topic,{})
                if calibration.get('depth_registered_to_color') is not True:
                    raise ValueError('传感器深度与彩色图配准尚未确认，请到相机标定页核对')
                if selected_depths and label in selected_depths:
                    depth=selected_depths[label];skew=abs(depth.stamp-frame.stamp)
                    if skew>.08 or time.monotonic()-depth.received>.8:raise ValueError('实际使用深度帧过期或不同步')
                else:_,depth,skew=buffer.pair(color_topic,depth_topic,frame.displayed)
                item.update(depth_stamp=depth.stamp,skew_s=skew)
            result['frames'][label]=item
        except Exception as exc:result['issues'].append(label+': '+str(exc))
    stamps=[f['stamp'] for f in result['frames'].values()]
    if not stamps or max(stamps)-min(stamps)>.12:
        result['issues'].append('多相机采集时间差超过120ms或无有效帧')
    return result


class CaptureTimeTF:
    """All estimation queries use the frozen acquisition timestamp."""
    def __init__(self,buffer,stamp):self.buffer,self.stamp=buffer,stamp
    def lookup_transform(self,target,source,_time,**kwargs):
        from rclpy.time import Time
        return self.buffer.lookup_transform(target,source,Time(seconds=self.stamp),**kwargs)


def verify_live(observation, buffer, calibration_path, now=None):
    now=time.time() if now is None else now
    provenance=observation.get('provenance') or {}
    if provenance.get('version')!=1 or provenance.get('issues'):
        raise ValueError('观测缺少同步采集/标定证据，请重新估计：'+'；'.join(provenance.get('issues',[])))
    digest=calibration_digest(calibration_path)
    if provenance.get('calibration_sha256')!=digest:
        raise ValueError('相机标定已变化，必须重新估计')
    # Prefer head; wrist viewpoints change during grasp motion.
    item=provenance.get('frames',{}).get('head')
    if not item:raise ValueError('缺少头部基准视图，不能自动复核历史场景')
    with buffer.lock:frames=list(buffer.frames.get(item['topic'],()))
    if not frames or time.monotonic()-frames[-1].received>.8:
        raise ValueError('现场复核没有新鲜头部画面')
    frame=frames[-1]
    if frame.stamp<=item['stamp']:
        raise ValueError('现场复核尚未收到晚于估计的新帧')
    if provenance.get('robot_q16') is not None:
        import os
        from grasp_execution import read_feedback
        current_q=read_feedback(os.environ.get('EAI_HOST_FK_STATE','/tmp/eai_host_fk_state.json'))['q16']
        if np.any(abs(np.asarray(current_q)[:2]-np.asarray(provenance['robot_q16'])[:2])>np.array([.002,.005])):
            raise ValueError('腰部/头部相机基座已变化，请重新估计')
    reference=np.asarray(item['thumbnail'],float);current=thumbnail(frame.image)
    difference=np.abs(reference-current)
    mean=float(difference.mean());changed=float(np.mean(np.max(difference,axis=2)>35))
    if mean>12 or changed>.08:
        raise ValueError(f'现场图像变化（平均差{mean:.1f}，变化区域{changed:.0%}），请重新估计目标')
    target=provenance.get('target_patch')
    if target:
        h,w=frame.image.shape[:2];x0,y0,x1,y1=target['roi_normalized']
        crop=frame.image[int(y0*h):max(int(y0*h)+1,int(y1*h)),int(x0*w):max(int(x0*w)+1,int(x1*w))]
        error=float(np.mean(abs(thumbnail(crop)-np.asarray(target['thumbnail']))))
        if error>15:raise ValueError(f'目标局部图像变化（差值{error:.1f}），请重新估计')
    return {'observation_id':observation['observation_id'],'verified_at':now,
            'frame_stamp':frame.stamp,'calibration_sha256':digest,
            'mean_pixel_difference':mean,'changed_fraction':changed,
            'kind':'live_appearance_check','pose_reestimated':False}


def require_verified(observation, verification, max_age=5., now=None):
    now=time.time() if now is None else now
    if not verification or verification.get('observation_id')!=observation.get('observation_id'):
        raise ValueError('当前目标尚未现场复核；历史结果只能预览/离线规划')
    if not 0<=now-float(verification.get('verified_at',0))<=max_age:
        raise ValueError('现场复核已过期，请再次复核')
    if observation.get('execution_blockers'):
        raise ValueError('；'.join(observation['execution_blockers']))


def mark_cached(observations):
    for observation in observations.values():
        observation.pop('verification',None)
        observation.setdefault('provenance',{})['origin']='cache'
