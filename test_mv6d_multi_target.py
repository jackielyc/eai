import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import unittest
from types import SimpleNamespace as NS
import numpy as np

from PyQt5.QtWidgets import QApplication, QCheckBox, QComboBox, QLabel, QPushButton, QSpinBox
import show_camera_topics as m

m._ensure_cv2_numpy()


class Status:
    def __init__(self): self.messages=[]
    def showMessage(self, text, *_): self.messages.append(text)


class Harness:
    _mv6d_update_seed_status_label=m.CameraTopicWindow._mv6d_update_seed_status_label
    _set_mv6d_active_target=m.CameraTopicWindow._set_mv6d_active_target
    _refresh_mv6d_target_combo=m.CameraTopicWindow._refresh_mv6d_target_combo
    _update_mv6d_batch_summary=m.CameraTopicWindow._update_mv6d_batch_summary
    _mv6d_mark_object_picked=m.CameraTopicWindow._mv6d_mark_object_picked
    _on_mv6d_target_changed=m.CameraTopicWindow._on_mv6d_target_changed
    _remove_mv6d_target=m.CameraTopicWindow._remove_mv6d_target
    _clear_mv6d_targets=m.CameraTopicWindow._clear_mv6d_targets
    _on_mv6d_seed_spin_changed=m.CameraTopicWindow._on_mv6d_seed_spin_changed
    _on_mv6d_estimate_once=m.CameraTopicWindow._on_mv6d_estimate_once
    _start_next_mv6d_target=m.CameraTopicWindow._start_next_mv6d_target
    _finish_mv6d_target=m.CameraTopicWindow._finish_mv6d_target

    def __init__(self):
        self.mv6d_head_u_spin=QSpinBox();self.mv6d_head_v_spin=QSpinBox()
        self.mv6d_gt_enable_check=QCheckBox();self.mv6d_seed_status_label=QLabel()
        self.mv6d_target_combo=QComboBox();self.mv6d_remove_target_btn=QPushButton()
        self.mv6d_clear_targets_btn=QPushButton();self.mv6d_once_btn=QPushButton()
        self.mv6d_batch_summary_label=QLabel()
        self.status_bar=Status();self._mv6d_busy=False;self._mv6d_object_picked=False
        self._mv6d_targets=[];self._mv6d_target_results={};self._mv6d_target_serial=0
        self._mv6d_estimate_queue=[];self._mv6d_active_target=None;self._mv6d_batch_active=False
        self.started=[];self.shown=[]
        self.mv6d_target_combo.currentIndexChanged.connect(self._on_mv6d_target_changed)
    def _refresh_mv6d_live_previews_for_pick(self): pass
    def _sync_mv6d_targets_to_grasp(self): pass
    def _show_mv6d_target_result(self,result): self.shown.append(result)
    def _mv6d_ensure_object_selected(self,**_): return self._mv6d_object_picked
    def _start_mv6d_estimate(self,**_): self.started.append(self._mv6d_active_target['id'])


class MultiTargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.app=QApplication.instance() or QApplication([])

    def test_clicks_add_select_edit_delete_and_clear_targets(self):
        h=Harness()
        for uv in ((10,20),(30,40),(50,60)): h._mv6d_mark_object_picked(*uv)
        self.assertEqual([(t['u'],t['v']) for t in h._mv6d_targets],[(10,20),(30,40),(50,60)])
        self.assertEqual(h.mv6d_target_combo.count(),3)
        h.mv6d_target_combo.setCurrentIndex(0)
        self.assertEqual((h.mv6d_head_u_spin.value(),h.mv6d_head_v_spin.value()),(10,20))
        h._mv6d_target_results['target-1']={'ok':True}
        h.mv6d_head_u_spin.setValue(11);h._on_mv6d_seed_spin_changed()
        self.assertEqual(h._mv6d_targets[0]['u'],11)
        self.assertNotIn('target-1',h._mv6d_target_results)
        h._remove_mv6d_target();self.assertEqual(len(h._mv6d_targets),2)
        h._clear_mv6d_targets();self.assertFalse(h._mv6d_targets);self.assertFalse(h._mv6d_object_picked)

    def test_estimate_once_runs_every_target_sequentially(self):
        h=Harness()
        for uv in ((10,20),(30,40),(50,60)): h._mv6d_mark_object_picked(*uv)
        h._on_mv6d_estimate_once();self.assertEqual(h.started,['target-1'])
        for target_id in ('target-1','target-2','target-3'):
            h._finish_mv6d_target({'ok':True,'value':target_id})
            self.app.processEvents()
        self.assertEqual(h.started,['target-1','target-2','target-3'])
        self.assertFalse(h._mv6d_batch_active)
        self.assertEqual(set(h._mv6d_target_results),{'target-1','target-2','target-3'})
        self.assertIn('成功 3/3',h.status_bar.messages[-1])

    def test_all_successful_targets_are_drawn_on_head_image(self):
        def result(x,y):
            view=m.Object6DPoseResult(
                mask=np.ones((100,100),np.uint8),points_3d=np.zeros((1,3)),centroid_uv=(x,y),
                contact_uv=(x,y),contact_xyz=(0,0,1),position_xyz=(0,0,1),euler_rpy_rad=(0,0,0),
                quaternion_xyzw=(0,0,0,1),rotation_matrix=np.eye(3),obb_center=(0,0,1),
                obb_extents=(.1,.1,.1),obb_corners=np.array([[-.05,-.05,1],[.05,-.05,1],
                    [-.05,.05,1],[.05,.05,1],[-.05,-.05,1.1],[.05,-.05,1.1],
                    [-.05,.05,1.1],[.05,.05,1.1]]),depth_m=1,pixel_count=1,point_count=1,
                method='pca',intrinsics=(100,100,50,50))
            snap=m.MultiView6DPoseSnapshot('camera',(0,0,1),(0,0,0),(0,0,0,1),np.eye(3),
                (.1,.1,.1),view.obb_corners,head_result=view)
            return {'ok':True,'snap':snap}
        h=NS(mv6d_overlay_check=NS(isChecked=lambda:True),
             _mv6d_targets=[{'id':'a'},{'id':'b'}],
             _mv6d_target_results={'a':result(30,30),'b':result(70,70)})
        out=m.CameraTopicWindow._draw_mv6d_target_results(h,np.zeros((100,100,3),np.uint8))
        self.assertGreater(np.count_nonzero(out),100)

    def test_target_observations_are_sent_to_grasp_and_robot_layers(self):
        received=[];visuals=[]
        observations={}
        for key,x in (('a',.2),('b',.5)):
            T=np.eye(4);T[0,3]=x
            observations[key]={'pca':{'T_base_object':T.tolist(),
                'object_pose':{'extents_m':[.1,.2,.3]}}}
        h=NS(_mv6d_targets=[{'id':'a','u':1,'v':2},{'id':'b','u':3,'v':4}],
             _mv6d_target_results={key:{'observations':obs} for key,obs in observations.items()},
             grasp_execution_panel=NS(set_target_observations=received.append),
             mv6d_method_combo=NS(currentData=lambda:'pca'),
             _robot_view=NS(set_target_poses=visuals.append))
        m.CameraTopicWindow._sync_mv6d_targets_to_grasp(h)
        self.assertEqual(set(received[0]),{'a','b'})
        self.assertEqual(len(visuals[0]),2)
        self.assertEqual([v['xyz'][0] for v in visuals[0]],[.2,.5])


if __name__=='__main__': unittest.main()
