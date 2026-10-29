import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace as NS
import numpy as np
from PyQt5.QtWidgets import QApplication, QComboBox
import show_camera_topics as m


class StoreHarness:
    _grasp_task_path=m.CameraTopicWindow._grasp_task_path
    _save_grasp_task_data=m.CameraTopicWindow._save_grasp_task_data
    _load_grasp_task_data=m.CameraTopicWindow._load_grasp_task_data
    def __init__(self,snap):
        self._mv6d_targets=[{'id':'target-1','u':12,'v':34}]
        self._mv6d_target_results={'target-1':{'ok':True,'target':dict(self._mv6d_targets[0]),
            'snap':snap,'observations':{'pca':{'observation_id':'saved'}}}}
        self._mv6d_last_snap=snap;self._robot_method_poses={};self._robot_sam3_points=np.ones((3,3))
        self._robot_lingbot_depth=None;self._robot_view=None;self._mv6d_active_target=None
        self._mv6d_object_picked=False;self._mv6d_target_serial=0;self.mv6d_target_combo=QComboBox()
    def _refresh_mv6d_target_combo(self,*_):
        self.mv6d_target_combo.clear()
        for t in self._mv6d_targets:self.mv6d_target_combo.addItem(t['id'],t['id'])
    def _sync_mv6d_targets_to_grasp(self): pass
    def _set_mv6d_active_target(self,target,**_): self._mv6d_active_target=target


class TaskStoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.app=QApplication.instance() or QApplication([])
    def test_task_snapshot_roundtrip_includes_image_pose_and_panel_state(self):
        result=m.Object6DPoseResult(mask=np.ones((2,2),np.uint8),points_3d=np.zeros((1,3)),
            centroid_uv=(1,1),contact_uv=(1,1),contact_xyz=(0,0,1),position_xyz=(0,0,1),
            euler_rpy_rad=(0,0,0),quaternion_xyzw=(0,0,0,1),rotation_matrix=np.eye(3),
            obb_center=(0,0,1),obb_extents=(.1,.2,.3),obb_corners=np.zeros((8,3)),depth_m=1,
            pixel_count=4,point_count=1,method='pca',source_bgr=np.full((4,6,3),73,np.uint8))
        snap=m.MultiView6DPoseSnapshot('base_link',(0,0,1),(0,0,0),(0,0,0,1),np.eye(3),
                                      (.1,.2,.3),np.zeros((8,3)),head_result=result,tf_ok=True)
        h=StoreHarness(snap);state={'roles':{'phone':'target-1'},'workflow_step':4,
                                   'selected_target':'target-1','arm_side':'right'}
        with tempfile.TemporaryDirectory() as root, patch.object(m,'EAI_DIR',root):
            message=h._save_grasp_task_data('phone_box_assembly',state)
            self.assertIn('已保存',message)
            h._mv6d_targets=[];h._mv6d_target_results={}
            loaded=h._load_grasp_task_data('phone_box_assembly')
        self.assertEqual(loaded['panel_state']['workflow_step'],4)
        self.assertEqual(loaded['task_definition']['name'],'手机盒装配')
        self.assertEqual(loaded['target_observations']['target-1']['methods']['pca']['observation_id'],'saved')
        restored=h._mv6d_target_results['target-1']['snap'].head_result.source_bgr
        np.testing.assert_array_equal(restored,np.full((4,6,3),73,np.uint8))


if __name__=='__main__':unittest.main()
