"""Calibration/range/segmentation regressions. Source ROS Humble before running."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import cv2
import numpy as np
import perception_geometry as g
import show_camera_topics as m
from depth_cloud_node import _intrinsics
from robot_view3d import _pack_depth_image

m._ensure_cv2_numpy()


class IntrinsicsTests(unittest.TestCase):
    def test_saved_head_intrinsics_are_scaled_by_native_image_size(self):
        store = {'/camera/head_color': {'intrinsics': dict(fx=645.1859,fy=644.9612,cx=639.5412,cy=389.899)}}
        k, source = g.resolve_intrinsics('/camera/head_depth',480,300,cameras=store)
        np.testing.assert_allclose(k,[645.1859*480/1280,644.9612*300/800,639.5412*480/1280,389.899*300/800])
        self.assertEqual(source,'标定内参')
        self.assertLess(k[0],250.)

    def test_live_camera_info_takes_priority_and_scales_each_axis(self):
        live={'/camera/head_depth':(700.,710.,630.,390.)}
        k,source=g.resolve_intrinsics('/camera/head_depth',640,200,live=live,live_sizes={'/camera/head_depth':(1280,800)})
        self.assertEqual(k,(350.,177.5,315.,97.5))
        self.assertEqual(source,'CameraInfo')

    def test_wrist_alias_and_renderer_use_identical_intrinsics(self):
        store={spec['topic']:{'intrinsics':{k:spec[k] for k in ('fx','fy','cx','cy')}} for spec in g.CAMERAS}
        for side in ('left','right'):
            color=f'/camera/hand_{side}_color'
            a,_=g.resolve_intrinsics(f'/camera/{side}_wrist_depth',480,271,cameras=store)
            b=_intrinsics(dict(store[color],color_topic=color),480,271)
            np.testing.assert_allclose(a,b)
            self.assertLess(a[0],250.)

    def test_center_offset_is_preserved_not_used_as_image_width(self):
        entry={'intrinsics':{'fx':400,'fy':450,'cx':300,'cy':200,'width':800,'height':600}}
        self.assertEqual(g.calibrated_intrinsics(entry,400,150),(200.,112.5,150.,50.))

    def test_camera_info_callback_keeps_native_dimensions(self):
        node=NS(_camera_intrinsics={},_camera_intrinsics_sizes={},_camera_frame_ids={})
        msg=NS(k=[600.,0,640.,0,620.,400.,0,0,1],width=1280,height=800,header=NS(frame_id='optical'))
        m.CameraTopicNode._camera_info_callback(node,msg,'/camera/head_depth')
        self.assertEqual(m.CameraTopicNode.get_intrinsics(node,'/camera/head_depth',480,300),(225.,232.5,240.,150.))
        self.assertEqual(node._camera_frame_ids['/camera/head_depth'],'optical')


class ExtrinsicsTests(unittest.TestCase):
    def test_exact_calibration_is_valid_and_poor_calibration_is_not(self):
        from depth_cloud_node import _placement
        entry={'model':'eye_to_hand','rms_m':0.,'T_base_camera':np.eye(4).tolist()}
        self.assertIsNotNone(g.validated_camera_transform(entry)[0])
        entry['rms_m']=.334
        self.assertIsNone(g.validated_camera_transform(entry)[0])
        self.assertIn('未标定示意',_placement(entry,{'fallback_parent':'base_link','frame':'camera'})[-1])

    def test_nonrigid_and_implausible_wrist_transforms_rejected(self):
        t=np.eye(4);t[0,3]=.6
        entry={'model':'eye_in_hand','rms_m':.001,'T_tcp_camera':t.tolist()}
        self.assertIsNone(g.validated_camera_transform(entry)[0])
        t[0,3]=.1;t[0,0]=2.
        entry['T_tcp_camera']=t.tolist()
        self.assertIsNone(g.validated_camera_transform(entry)[0])


class DepthAndMaskTests(unittest.TestCase):
    def test_range_filter_handles_invalid_and_mm_inputs(self):
        np.testing.assert_array_equal(g.sanitize_depth(np.array([0,70,500,2500,64000],np.uint16)),[0,0,.5,0,0])
        np.testing.assert_array_equal(g.sanitize_depth(np.array([np.nan,np.inf,-1,.5,50])),[0,0,0,.5,0])

    def test_raw_display_and_segmentation_reproject_same_pixels(self):
        depth=np.full((8,8),1.,np.float32);depth[0,0]=50.;depth[4,4]=np.nan
        k=(240.,245.,3.,4.)
        raw=_pack_depth_image({},depth,'32FC1',intrinsics=k)
        mask=np.zeros_like(depth,dtype=bool);mask[::4,::4]=True
        pts,_=m.mask_to_points_3d(mask,depth,*k)
        np.testing.assert_allclose(raw[:,:3],pts,atol=1e-7)

    def test_tilted_object_not_clipped_at_seed_plus_22mm(self):
        depth=np.tile(.5+np.arange(50,dtype=np.float32)*.002,(40,1))
        mask=np.ones(depth.shape,bool)
        filtered,points=m.refine_stereo_segment(mask,depth,0,20,800.,800.,25.,20.)
        self.assertGreater(len(points),100)
        self.assertGreater(float(points[:,2].max()),.57)
        self.assertGreater(float(points[:,2].max()-depth[20,0]),.022)

    def test_raw_sam3_mask_and_captured_rgb_survive_depth_filtering(self):
        rgb=np.zeros((20,30,3),np.uint8);rgb[...,1]=70
        raw=np.zeros((20,30),bool);raw[3:17,5:25]=True
        depth=np.full((10,15),.5,np.float32);depth[:,:5]=0
        with tempfile.TemporaryDirectory() as root,patch.object(m,'EAI_DIR',root),patch.object(m,'run_sam3_segmentation',return_value=(raw,'sam3-test')):
            result=m.CameraTopicWindow._estimate_view_6d(rgb,depth,15,10,240.,240.,7.5,5.,method_tag='head')
            self.assertIsNotNone(result)
            np.testing.assert_array_equal(result.raw_mask,raw)
            np.testing.assert_array_equal(result.source_bgr,rgb)
            self.assertIsNot(result.source_bgr,rgb)
            self.assertEqual(result.intrinsics,(240.,240.,7.5,5.))
            saved=cv2.imread(str(Path(result.diagnostics_dir)/'mask_raw.png'),cv2.IMREAD_GRAYSCALE)
            np.testing.assert_array_equal(saved,raw.astype(np.uint8)*255)
            self.assertTrue((Path(result.diagnostics_dir)/'mask_depth_filtered.png').is_file())


class ReferenceTests(unittest.TestCase):
    def snapshot(self,frame='base_link'):
        return m.MultiView6DPoseSnapshot(frame,(0,0,.5),(0,0,0),(0,0,0,1),np.eye(3),(.1,.1,.1),np.zeros((8,3)))

    def test_reference_difference_not_claimed_as_segmentation_error(self):
        text=m.format_multiview_6d_text(self.snapshot(),(.1,0,.5),np.eye(3),True)
        self.assertIn('位姿参考差值（非 SAM3 分割误差）',text)
        self.assertIn('100.0 mm',text)
        self.assertNotIn('真值 XYZ',text)

    def test_camera_frame_cannot_be_compared_to_base_reference(self):
        snap=self.snapshot('camera')
        text=m.format_multiview_6d_text(snap,(0,0,0),np.eye(3),True)
        self.assertNotIn('|Δt|',text)
        owner=NS(_mv6d_gt_pose=lambda:((0,0,0),np.eye(3)),mv6d_gt_enable_check=NS(isChecked=lambda:True))
        self.assertEqual(m.CameraTopicWindow._mv6d_error_pair(owner,snap),(None,None))

    def test_reselecting_target_disables_old_reference(self):
        checked=[]
        spin=NS(blockSignals=lambda _:None,setValue=lambda _:None)
        owner=NS(mv6d_head_u_spin=spin,mv6d_head_v_spin=spin,
                 mv6d_gt_enable_check=NS(setChecked=checked.append),
                 _mv6d_update_seed_status_label=lambda:None,
                 _refresh_mv6d_live_previews_for_pick=lambda:None,
                 status_bar=NS(showMessage=lambda _:None))
        m.CameraTopicWindow._mv6d_mark_object_picked(owner,10,20)
        self.assertEqual(checked,[False])

    def test_stale_simulator_reference_cannot_override_real_camera(self):
        owner=NS(_mv6d_sim_reference_is_current=lambda:False,
                 _read_sim_robot_state=lambda: self.fail('must not read stale objects'))
        self.assertEqual(m.CameraTopicWindow._mv6d_sim_objects_list(owner),[])
        m.CameraTopicWindow._mv6d_auto_gt_from_sim(owner,(0,0,0))


if __name__=='__main__':unittest.main()
