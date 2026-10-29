import unittest
from types import SimpleNamespace as NS
import numpy as np
from mv6d_engine import compute_snapshot


class EngineTests(unittest.TestCase):
    def test_learned_pose_keeps_its_orientation_and_skips_pca_fusion(self):
        R=np.array([[0,-1,0],[1,0,0],[0,0,1]],float)
        result=NS(position_xyz=[.1,.2,.3],rotation_matrix=R,obb_extents=[.1,.2,.3],
                  point_count=100,method='foundationpose')
        T=np.eye(4);T[0,3]=.5
        def forbidden(*args,**kwargs):raise AssertionError('learned pose must not be replaced by PCA')
        services=NS(_estimate_view_6d=lambda *a,**kw:result,
            _mv6d_sim_extrinsic_matrix=lambda *a:None,_mv6d_manual_extrinsic_matrix=lambda *a:T,
            _mv6d_refine_with_sim_gt=lambda *a,**kw:(None,None,''),
            _mv6d_try_wrist_view_if_visible=forbidden,_get_paired_depth_topic_name=lambda t:'depth')
        deps=dict(BASE_LINK_FRAME='base_link',MV6D_EST_GT_SNAP_MAX_DIST_M=.1,MV6D_FUSE_WEIGHT_HEAD=3,
            MV6D_FUSE_WEIGHT_WRIST=1,MultiView6DPoseSnapshot=NS,Object6DPoseResult=NS,
            POSE_BACKEND_ANY6D='any6d',POSE_BACKEND_FOUNDATIONPOSE='foundationpose',POSE_BACKEND_PCA='pca',
            estimate_6d_pose_pca=forbidden,lookup_transform_matrix=lambda *a:None,
            obb_corners=lambda p,r,e:np.zeros((8,3)),quat_to_rotation_matrix=lambda *a:np.eye(3),
            refine_fused_object_points=forbidden,rotation_matrix_to_euler_xyz=lambda r:(0,0,1.57),
            rotation_matrix_to_quaternion=lambda r:(0,0,.707,.707),
            transform_points_by_matrix=lambda p,t:p@t[:3,:3].T+t[:3,3])
        snap=compute_snapshot(services,dependencies=deps,head_topic='head',head_depth_topic='depth',
            head_bgr=np.zeros((4,4,3)),head_depth=np.ones((4,4)),seed_u=1,seed_v=1,
            tf_buffer=None,get_intrinsics=lambda *a:(1,1,0,0),get_camera_frame=lambda *a:'camera',
            pose_backend='foundationpose')
        np.testing.assert_allclose(snap.position_xyz,[.6,.2,.3],atol=1e-7)
        np.testing.assert_allclose(snap.rotation_matrix,R)
        self.assertTrue(snap.tf_ok)


if __name__=='__main__':unittest.main()
