import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from grasp_scene import CollisionScene,load_scene,scene_digest
from host_arm_ik import _resolve_urdf


class SceneTests(unittest.TestCase):
    def test_uncalibrated_scene_is_not_a_hardware_scene(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'scene.json'
            path.write_text(json.dumps({'version':1,'base_frame':'base_link','calibrated':False,'world':[]}))
            load_scene(path)
            with self.assertRaisesRegex(ValueError,'待标定'):load_scene(path,hardware=True)

    def test_full_geometry_includes_hands_cameras_and_detects_world_collision(self):
        scene={'version':1,'base_frame':'base_link','calibrated':False,'world':[
            {'id':'test_obstacle','size_m':[10,10,10],'T_base_object':np.eye(4).tolist()}]}
        checker=CollisionScene(_resolve_urdf(),scene,hands={'left':[0]*6,'right':[0]*6})
        names=[obj.name for obj in checker.geometry.geometryObjects]
        self.assertTrue(any(n.startswith('hand1_') for n in names))
        self.assertTrue(any(n.startswith('wrist_camera_') for n in names))
        self.assertTrue(any(n.startswith('wrist_connector_') for n in names))
        with self.assertRaisesRegex(ValueError,'world:test_obstacle'):checker.check([0]*16)
        before=scene_digest(scene);scene['world'][0]['size_m'][0]=9
        self.assertNotEqual(before,scene_digest(scene))

    def test_path_samples_between_endpoints(self):
        checker=object.__new__(CollisionScene)
        seen=[]
        def check(q):
            seen.append(q[2])
            if .045<q[2]<.055:raise ValueError('middle collision')
        checker.check=check
        a=np.zeros(16);b=a.copy();b[2]=.1
        with self.assertRaisesRegex(ValueError,'middle collision'):checker.validate_path([a,b])
        self.assertGreater(len(seen),2)


if __name__=='__main__':unittest.main()
