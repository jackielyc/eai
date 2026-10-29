"""Validated RGB-D I/O and result framing shared by both LingBot entry points."""
from __future__ import annotations

import codecs
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from eai_app.perception.geometry import sanitize_depth, valid_depth_mask


def align_rgb_to_depth(rgb, depth):
    import cv2
    rgb, depth = np.asarray(rgb), np.asarray(depth)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.size:
        raise ValueError('RGB 必须是非空三通道图像')
    if depth.ndim != 2 or not depth.size:
        raise ValueError('深度必须是非空二维图像')
    h,w=depth.shape
    if rgb.shape[:2] != (h,w):
        rgb=cv2.resize(rgb,(w,h),interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(rgb), np.ascontiguousarray(depth)


def validate_intrinsics(k):
    k=np.asarray(k,dtype=np.float64)
    if k.shape!=(3,3) or not np.isfinite(k).all() or k[0,0]<=0 or k[1,1]<=0 or not np.allclose(k[2],[0,0,1]):
        raise ValueError('内参必须是有效的 3×3 像素内参矩阵，与输入深度分辨率一致')
    return k


def write_camera_inputs(out_dir, rgb, depth, k):
    import cv2
    """Keep the sensor pixel grid and validate writes before starting inference."""
    rgb,depth=align_rgb_to_depth(rgb,depth)
    k=validate_intrinsics(k)
    meters=sanitize_depth(depth)
    if not np.any(meters):
        raise ValueError('原始深度在 0.08–2.5m 范围内没有有效测量，无法精修')
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    rgb_path,depth_path,k_path=out/'rgb.png',out/'raw_depth.png',out/'intrinsics.txt'
    if not cv2.imwrite(str(rgb_path),rgb):raise OSError(f'无法写入 RGB: {rgb_path}')
    if not cv2.imwrite(str(depth_path),np.rint(meters*1000).astype(np.uint16)):
        raise OSError(f'无法写入深度: {depth_path}')
    np.savetxt(k_path,k,fmt='%.8f')
    return str(rgb_path),str(depth_path),str(k_path)


def load_depth_input(path, scale=1000.):
    import cv2
    if not np.isfinite(scale) or scale<=0:raise ValueError('depth-scale 必须为正数')
    path=Path(path)
    if path.suffix.lower()=='.npy':
        depth=np.load(path,allow_pickle=False)
        if not np.issubdtype(depth.dtype,np.floating):raise ValueError('NPY 深度必须是以米为单位的浮点数组')
        depth=depth.astype(np.float32)
    else:
        raw=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
        if raw is None:raise ValueError(f'无法读取深度: {path}')
        if raw.dtype!=np.uint16:raise ValueError('深度 PNG 必须为 uint16 原始深度，不能使用彩色预览图或 8 位灰度图')
        depth=raw.astype(np.float32)/scale
    if depth.ndim!=2 or not depth.size:raise ValueError('深度必须是非空二维图像')
    return depth


def prediction_to_cloud(prediction, k, expected_shape):
    depth=np.asarray(prediction)
    if depth.shape!=tuple(expected_shape):raise ValueError(f'模型深度输出形状异常: {depth.shape}')
    depth=sanitize_depth(depth)
    valid=valid_depth_mask(depth)
    if not np.any(valid):raise ValueError('LingBot-Depth 没有有效深度输出')
    k=validate_intrinsics(k)
    v,u=np.indices(depth.shape,dtype=np.float32)
    points=np.stack(((u-k[0,2])*depth/k[0,0],(v-k[1,2])*depth/k[1,1],depth),axis=-1)
    return depth,points.astype(np.float32),valid


class ResultStream:
    """QProcess chunks may split a UTF-8 character or an entire JSON result."""
    def __init__(self):
        self.decoder=codecs.getincrementaldecoder('utf-8')('replace')
        self.buffer=''
        self.result=None

    def feed(self,data=b'',*,final=False):
        self.buffer+=self.decoder.decode(data,final=final)
        lines=self.buffer.split('\n')
        self.buffer=lines.pop()
        if final and self.buffer:
            lines.append(self.buffer);self.buffer=''
        logs=[]
        for line in lines:
            line=line.strip()
            if not line:continue
            if line.startswith('[RESULT] '):
                try:
                    result=json.loads(line[9:])
                    if isinstance(result,dict):self.result=result
                    else:logs.append(line)
                except json.JSONDecodeError:logs.append(line)
            else:logs.append(line)
        return logs


def run_cancelable(cmd, *, cancel_event=None, timeout=600, **kwargs):
    if cancel_event is None:
        return subprocess.run(cmd,capture_output=True,text=True,timeout=timeout,**kwargs)
    if cancel_event.is_set():raise RuntimeError('LingBot-Depth 已停止')
    start=time.monotonic()
    with subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,**kwargs) as proc:
        while True:
            reason=''
            if cancel_event.is_set():reason='LingBot-Depth 已停止'
            elif time.monotonic()-start>timeout:reason='LingBot-Depth 推理超时'
            if reason:
                proc.terminate()
                try:proc.communicate(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill();proc.communicate()
                raise RuntimeError(reason)
            try:
                stdout,stderr=proc.communicate(timeout=.2)
                return subprocess.CompletedProcess(cmd,proc.returncode,stdout,stderr)
            except subprocess.TimeoutExpired:pass
