"""LingBot input, transport and lifecycle checks; no model or robot needed."""
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QProcess
import show_camera_topics as m
import lingbot_depth_worker as worker
from lingbot_depth_io import (
    ResultStream,align_rgb_to_depth,write_camera_inputs,load_depth_input,
    prediction_to_cloud,validate_intrinsics,run_cancelable,
)
m._ensure_cv2_numpy()


class InputTests(unittest.TestCase):
    def test_manual_capture_preserves_depth_grid_and_source_camera(self):
        rgb=np.full((8,12,3),100,np.uint8)
        depth=np.full((4,6),1000,np.uint16);depth[0,0]=64000
        calls=[]
        owner=NS(lingbot_depth_source_combo=NS(currentData=lambda:'camera'),
                 _refresh_lingbot_depth_topic_combo=lambda:None,
                 _pick_lingbot_depth_color_source=lambda:('/camera/head_color',rgb,None),
                 _get_paired_depth_topic_name=lambda _: '/camera/head_depth',
                 _ensure_topic_enabled_for_capture=lambda _:False,
                 topic_checks={'/camera/head_depth':True},
                 _get_paired_depth_frame=lambda _:depth,
                 node=NS(get_intrinsics=lambda *args:(calls.append(args) or (200.,210.,3.,2.))),
                 _append_lingbot_depth_log=lambda _:None,_on_lingbot_depth_status=lambda _:None)
        with tempfile.TemporaryDirectory() as tmp,patch.object(m,'LINGBOT_DEPTH_CACHE_DIR',tmp):
            result=m.CameraTopicWindow._resolve_lingbot_depth_inputs(owner)
            _,r,d,k=result
            self.assertEqual(cv2.imread(r).shape[:2],(4,6))
            self.assertEqual(cv2.imread(d,-1)[0,0],0)
            self.assertEqual(np.loadtxt(k)[0,0],200.)
        self.assertEqual(calls,[('/camera/head_depth',6,4)])
        self.assertEqual(owner._lingbot_depth_capture_topic,'/camera/head_color')

    def test_png_millimeters_and_npy_meters_have_same_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            png=Path(tmp)/'d.png';npy=Path(tmp)/'d.npy'
            cv2.imwrite(str(png),np.full((3,4),1250,np.uint16))
            np.save(npy,np.full((3,4),1.25,np.float32))
            np.testing.assert_array_equal(load_depth_input(png),load_depth_input(npy))
            cv2.imwrite(str(png),np.ones((3,4),np.uint8))
            with self.assertRaisesRegex(ValueError,'uint16'):load_depth_input(png)
            with self.assertRaises(ValueError):load_depth_input(npy,scale=0)

    def test_prediction_validation_and_pointcloud_use_same_depth(self):
        k=np.array([[2.,0,1.],[0,4.,1.],[0,0,1.]])
        pred=np.array([[1.,np.inf],[50.,.5]],np.float32)
        depth,points,valid=prediction_to_cloud(pred,k,(2,2))
        np.testing.assert_array_equal(valid,[[True,False],[False,True]])
        np.testing.assert_allclose(points[0,0],[-.5,-.25,1.])
        np.testing.assert_array_equal(points[:,:,2],depth)
        for wrong in (np.zeros((2,2)),np.full((2,2),np.nan),np.ones((4,4))):
            with self.assertRaises(ValueError):prediction_to_cloud(wrong,k,(2,2))
        with self.assertRaises(ValueError):validate_intrinsics(np.zeros((3,3)))

    def test_missing_explicit_model_or_depth_does_not_fall_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rgb=root/'rgb.png';k=root/'K.txt'
            cv2.imwrite(str(rgb),np.zeros((4,6,3),np.uint8));np.savetxt(k,np.eye(3))
            with self.assertRaises(FileNotFoundError):worker._resolve_model_checkpoint(str(root/'missing.pt'))
            args=NS(example='',rgb=str(rgb),depth=str(root/'missing.png'),intrinsics=str(k),allow_empty_depth=True)
            with self.assertRaises(FileNotFoundError):worker._resolve_inputs(args)
            with self.assertRaises(ValueError):worker._synthesize_sparse_depth_png(root/'s.png',height=4,width=6,sparse_z_m=.01)


class TransportTests(unittest.TestCase):
    def test_utf8_and_json_can_arrive_one_byte_at_a_time(self):
        result={'ok':True,'point_cloud':'/tmp/相机/点云.ply'}
        data=('加载中\n[RESULT] '+json.dumps(result,ensure_ascii=False)).encode()
        stream=ResultStream();logs=[]
        for byte in data:logs.extend(stream.feed(bytes([byte])))
        logs.extend(stream.feed(final=True))
        self.assertEqual(stream.result,result);self.assertEqual(logs,['加载中'])

    def test_cancel_terminates_the_owned_subprocess(self):
        event=threading.Event();timer=threading.Timer(.15,event.set);timer.start();start=time.monotonic()
        try:
            with self.assertRaisesRegex(RuntimeError,'已停止'):
                run_cancelable([sys.executable,'-c','import time;time.sleep(30)'],cancel_event=event,timeout=5)
        finally:timer.cancel()
        self.assertLess(time.monotonic()-start,3.)


class LauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])

    def test_failed_or_cancelled_process_cannot_publish_success(self):
        for code,cancelled in ((1,False),(0,True)):
            launcher=m.LingbotDepthLauncher();results=[]
            launcher.result_ready.connect(results.append)
            launcher._output.feed(b'[RESULT] {"ok":true,"points":1}')
            launcher._cancelled=cancelled
            launcher._on_process_finished(code,QProcess.NormalExit)
            self.assertFalse(results[-1]['ok'])

    def test_final_unterminated_line_keeps_captured_camera(self):
        launcher=m.LingbotDepthLauncher();results=[]
        launcher.result_ready.connect(results.append)
        launcher._result_context={'color_topic':'/camera/head_color'}
        launcher._output.feed(b'[RESULT] {"ok":true,"points":3}')
        launcher._on_process_finished(0,QProcess.NormalExit)
        self.assertTrue(results[-1]['ok'])
        self.assertEqual(results[-1]['color_topic'],'/camera/head_color')

    def test_delayed_kill_cannot_kill_a_new_process(self):
        killed=[]
        old=NS(state=lambda:QProcess.Running,kill=lambda:killed.append('old'))
        new=NS(state=lambda:QProcess.Running,kill=lambda:killed.append('new'))
        launcher=m.LingbotDepthLauncher();launcher._process=new
        launcher._force_kill(old);self.assertEqual(killed,['old'])
        launcher._process=None

    def test_unregistered_file_cloud_is_not_placed_at_robot_origin(self):
        clears=[]
        owner=NS(_robot_view=NS(set_lingbot_depth=clears.append),
                 _lingbot_depth_last_color_topic='/camera/head_color',_robot_lingbot_depth={'old':True})
        with tempfile.NamedTemporaryFile(suffix='.ply') as f:
            m.CameraTopicWindow._show_lingbot_depth_on_robot_view(owner,{'point_cloud':f.name,'color_topic':''},silent=True)
        self.assertIsNone(owner._robot_lingbot_depth);self.assertEqual(clears,[None])

    def test_stop_cancels_pipeline_and_disables_realtime_restart(self):
        event=threading.Event();checked=[]
        owner=NS(_append_lingbot_depth_log=lambda _:None,_mv6d_lingbot_active=True,
                 _mv6d_lingbot_cancel=event,mv6d_enable_check=NS(setChecked=checked.append),
                 _on_lingbot_depth_status=lambda _:None,
                 _lingbot_depth_launcher=NS(stop=lambda:self.fail('not the standalone job')))
        m.CameraTopicWindow._on_lingbot_depth_stop_clicked(owner)
        self.assertTrue(event.is_set());self.assertEqual(checked,[False])

    def test_error_clears_stale_refinement(self):
        calls=[]
        owner=NS(_clear_lingbot_depth_result=lambda:calls.append('clear'))
        m.CameraTopicWindow._on_lingbot_depth_result(owner,{'ok':False,'error':'failed'})
        self.assertEqual(calls,['clear'])


if __name__=='__main__':unittest.main()
