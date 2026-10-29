import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QPushButton, QTextEdit
import show_camera_topics as m
from mv6d_cache import load_cache, save_cache

m._ensure_cv2_numpy()


class CacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)/'result.npz'
        self.snap = m.MultiView6DPoseSnapshot(
            frame_id='base_link', position_xyz=(.3,.2,.8), euler_rpy_rad=(0,0,0),
            quaternion_xyzw=(0,0,0,1), rotation_matrix=np.eye(3), obb_extents=(.1,.2,.3),
            obb_corners=np.zeros((8,3)), tf_ok=True)
        self.data = dict(snap=self.snap, captured_at=123., method_poses={'pca':
            (np.array([.3,.2,.8]), np.eye(3), None, (.1,.2,.3))},
            observations={'pca': {'source_label':'PCA', 'observation_id':'original', 'captured_at':123.}},
            sam3_points=np.zeros((4,3)), image=np.ones((8,8,3), dtype=np.uint8))
        self.classes = {'MultiView6DPoseSnapshot': m.MultiView6DPoseSnapshot,
                        'Object6DPoseResult': m.Object6DPoseResult}

    def tearDown(self): self.temp.cleanup()

    def test_roundtrip_and_failed_write_preserves_previous(self):
        save_cache(self.path, self.data)
        restored = load_cache(self.path, self.classes)
        self.assertIsInstance(restored['snap'], m.MultiView6DPoseSnapshot)
        np.testing.assert_array_equal(restored['image'], self.data['image'])
        np.testing.assert_array_equal(restored['snap'].rotation_matrix, np.eye(3))
        self.assertEqual(restored['observations'], self.data['observations'])
        with self.assertRaises(ValueError): save_cache(self.path, {'invalid':object()})
        self.assertEqual(load_cache(self.path,self.classes)['captured_at'],123.)

    def test_startup_restores_without_inference_or_refreshing_timestamp(self):
        self.snap.head_result=m.Object6DPoseResult(
            mask=np.ones((2,2),dtype=np.uint8), points_3d=np.zeros((1,3)), centroid_uv=(1,1),
            contact_uv=(1,1), contact_xyz=(0,0,1), position_xyz=(0,0,1), euler_rpy_rad=(0,0,0),
            quaternion_xyzw=(0,0,0,1), rotation_matrix=np.eye(3), obb_center=(0,0,1),
            obb_extents=(.1,.1,.1), obb_corners=np.zeros((8,3)), depth_m=1., pixel_count=4,
            point_count=1, method='pca', source_bgr=np.full((4,6,3),17,dtype=np.uint8))
        save_cache(self.path,self.data)
        received=[]
        messages=[]
        shown=[]
        h=NS(_mv6d_cache_path=lambda:str(self.path), _mv6d_busy=False,
             grasp_execution_panel=NS(set_method_observations=received.append),
             _refresh_mv6d_result_text=lambda snap:None,
             _apply_mv6d_overlays=lambda snap:(_ for _ in ()).throw(RuntimeError('UI not ready')),
             _set_mv6d_preview_label=lambda *args,**kwargs:shown.append((args,kwargs)),
             mv6d_head_preview=NS(),
             mv6d_result_edit=QTextEdit(),status_bar=NS(showMessage=messages.append))
        m.CameraTopicWindow._restore_mv6d_cache(h)
        self.assertEqual(received[0]['pca']['captured_at'],123.)
        self.assertIn('历史缓存',received[0]['pca']['source_label'])
        self.assertEqual(shown[0][1]['source_wh'],(6,4))
        np.testing.assert_array_equal(shown[0][0][1],np.full((4,6,3),17,dtype=np.uint8))
        self.assertIn('已恢复',messages[-1])
        np.testing.assert_allclose(h._robot_method_poses['pca'][0],[.3,.2,.8])

    def test_failed_estimation_and_timer_do_not_replace_saved_result(self):
        h=NS(_mv6d_last_snap=self.snap,mv6d_once_btn=QPushButton(),
             mv6d_result_edit=QTextEdit('上次结果'),status_bar=NS(showMessage=lambda text:None))
        m.CameraTopicWindow._on_mv6d_estimate_finished(h, {'ok':False,'error':'failed','depth_source':'lingbot'})
        self.assertEqual(h.mv6d_result_edit.toPlainText(),'上次结果')
        self.assertIs(h._mv6d_last_snap,self.snap)
        m.CameraTopicWindow._on_mv6d_timer_tick(h)

    def test_live_frame_refreshes_picker_even_with_cached_overlay(self):
        calls=[]
        h=NS(_arm_panel_selector_key=lambda:'mv6d',
             _control_tab_title_is_current=lambda title:True,
             _mv6d_preview_refresh_at=0., _mv6d_last_snap=self.snap,
             mv6d_overlay_check=NS(isChecked=lambda:True),
             mv6d_head_preview=NS(pixmap=lambda:QPixmap()),
             _refresh_mv6d_live_previews_for_pick=lambda:calls.append(True))
        m.CameraTopicWindow._maybe_refresh_mv6d_previews_for_topics(h,['/camera/head_color'])
        self.assertEqual(calls,[True])

    def test_cached_snapshot_does_not_block_live_wrist_frames(self):
        class Label:
            def setVisible(self, value): self.visible=value
        head,left,right=Label(),Label(),Label()
        images={'/camera/head_color':np.full((4,6,3),1,np.uint8),
                '/camera/hand_left_color':np.full((4,6,3),2,np.uint8),
                '/camera/hand_right_color':np.full((4,6,3),3,np.uint8)}
        shown=[]
        def pick(key):
            topic={'left':'/camera/hand_left_color','right':'/camera/hand_right_color'}.get(key,key)
            return (topic,images[topic])
        h=NS(mv6d_head_preview=head,mv6d_left_wrist_preview=left,mv6d_right_wrist_preview=right,
             _mv6d_left_wrist_pix=None,_mv6d_right_wrist_pix=None,
             _arm_panel_selector_key=lambda:'mv6d',_control_tab_title_is_current=lambda title:True,
             _pick_mv6d_color_frame=pick,_mv6d_object_picked=False,_mv6d_last_snap=self.snap,
             _draw_mv6d_target_results=lambda image:image,
             _mv6d_overlay_hand_skeleton=lambda pending:dict(pending),
             _set_mv6d_preview_label=lambda label,image,*args,**kwargs:shown.append((label,image)),
             _sync_mv6d_lightbox=lambda:None)
        m.CameraTopicWindow._refresh_mv6d_live_previews_for_pick(h)
        by_label={id(label):image for label,image in shown}
        np.testing.assert_array_equal(by_label[id(left)],images['/camera/hand_left_color'])
        np.testing.assert_array_equal(by_label[id(right)],images['/camera/hand_right_color'])

    def test_success_callback_updates_cache_and_failed_attempt_preserves_it(self):
        poses=self.data['method_poses']
        received=[]
        h=NS(_mv6d_last_snap=None, mv6d_once_btn=QPushButton(),
             mv6d_result_edit=QTextEdit(), status_bar=NS(showMessage=lambda text:None),
             _mv6d_cache_path=lambda:str(self.path),
             _mv6d_pose_in_base=lambda snap:poses['pca'],
             grasp_execution_panel=NS(set_method_observations=received.append),
             _refresh_mv6d_result_text=lambda snap:None, _apply_mv6d_overlays=lambda snap:None,
             _adopt_mv6d_segment_target=lambda snap:None, _mv6d_cmd_arm_side=lambda:'right')
        def show(snap, **kwargs): h._robot_method_poses=kwargs['method_poses']
        h._show_mv6d_on_robot_view=show
        m.CameraTopicWindow._on_mv6d_estimate_finished(h,dict(
            ok=True,silent=False,snap=self.snap,captured_at=234.,method_poses=poses,depth_source='lingbot'))
        self.assertEqual(len(received),1)
        self.assertEqual(load_cache(self.path,self.classes)['captured_at'],234.)
        saved=self.path.read_bytes()
        m.CameraTopicWindow._on_mv6d_estimate_finished(h,dict(ok=False,error='failed'))
        self.assertEqual(self.path.read_bytes(),saved)
        self.assertEqual(len(received),1)


if __name__ == '__main__': unittest.main()
