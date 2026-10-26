import unittest
from object_hand_shape import suggest_hand_shape, calibrated_preset

class ShapeTests(unittest.TestCase):
    def test_geometries(self):
        for sizes, key in [([.005,.06,.09],'edge_pinch'),([.02,.02,.04],'precision_pinch'),([.03,.04,.2],'wrap'),([.04,.05,.06],'tripod'),([.08,.09,.1],'power_wrap')]:
            with self.subTest(key=key):
                result=suggest_hand_shape({'ok':True,'object_pose':{'extents_m':sizes}})
                self.assertEqual(result['key'],key)
                self.assertFalse(result['execution_ready'])
    def test_no_fake_joint_mapping(self):
        item=suggest_hand_shape({'ok':True,'object_pose':{'extents_m':[.04,.05,.06]}})
        with self.assertRaises(ValueError): calibrated_preset({},item)
        p={'hand_shape_presets':{'tripod':{'calibrated':True,'aperture_range_m':[.03,.07],'open_positions':[0]*6,'close_positions':[.3]*6}}}
        self.assertEqual(calibrated_preset(p,item)[1],[.3]*6)
        item['suggested_aperture_m']=.2
        with self.assertRaises(ValueError): calibrated_preset(p,item)
    def test_invalid_depth_or_dimensions(self):
        for obs in [{'ok':True,'execution_blockers':['RGB-only']},{'ok':True,'object_pose':{'extents_m':[0,1,1]}}]:
            with self.assertRaises(ValueError): suggest_hand_shape(obs)

if __name__=='__main__': unittest.main()
