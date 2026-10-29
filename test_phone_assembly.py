import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from phone_assembly.scene import scenario,xml_scene
from phone_assembly.program import AssemblyProgram,evaluate_goal
from phone_assembly.mujoco_backend import MujocoAssemblyBackend
from phone_assembly.config import validate_hardware


def run(config):
    backend=MujocoAssemblyBackend(config);program=AssemblyProgram(backend,config);program.start()
    for _ in range(10000):
        program.tick()
        if program.state!='running':break
    return backend,program


class AssemblyTests(unittest.TestCase):
    def test_a2d_idle_arm_margin_keeps_perturbed_assembly_feasible(self):
        from phone_assembly.worker import run_headless
        from host_arm_ik import _resolve_urdf
        with tempfile.TemporaryDirectory() as directory:
            result=run_headless(scenario(1,True),directory,_resolve_urdf(),True,False)
        self.assertEqual(result['state'],'complete',result['message'])
        self.assertTrue(result['goal']['success'])
        self.assertGreater(result['ik']['body_collision_samples'],0)
        self.assertLessEqual(result['ik']['max_position_error_m'],.0011)
    def test_numerical_reset_is_failure_not_a_new_valid_scene(self):
        backend=MujocoAssemblyBackend(scenario());program=AssemblyProgram(backend,backend.config);program.start()
        backend.data.qpos[backend.model.joint('phone_free').qposadr[0]]=float('nan')
        program.tick()
        self.assertEqual(program.state,'failed')
        self.assertFalse(backend.simulation_valid)
        self.assertFalse(program.status()['goal']['success'])
        backend.close()
    def test_physical_assembly_succeeds_across_placement_and_yaw_changes(self):
        for seed in range(10):
            with self.subTest(seed=seed):
                backend,program=run(scenario(seed,True))
                self.assertEqual(program.state,'complete',program.message)
                goal=evaluate_goal(backend.observation(),backend.config)
                self.assertTrue(goal['success']);self.assertTrue(goal['released']);self.assertTrue(goal['tool_clear'])
                self.assertEqual(backend.object_write_count,0)
                self.assertEqual(backend.model.neq,1)  # Only the tool actuator weld; no object attachment cheats.
                object_ids={backend.model.body(name).id for name in ('phone','lid')}
                self.assertFalse(object_ids.intersection(set(backend.model.eq_obj1id)|set(backend.model.eq_obj2id)))
                self.assertTrue(all(backend.model.body_mocapid[i]<0 for i in object_ids))
                backend.close()
    def test_low_friction_cannot_be_reported_as_success(self):
        config=scenario();config['friction']=.0001;config['phone_mass_kg']=.3
        backend,program=run(config)
        self.assertEqual(program.state,'failed')
        self.assertFalse(program.status()['goal']['success'])
        self.assertLess(program.index,13)
        backend.close()
    def test_phone_cannot_fit_in_an_undersized_box(self):
        config=scenario();config['box_inner_size'][1]=.06
        backend,program=run(config)
        self.assertEqual(program.state,'failed')
        self.assertFalse(program.status()['goal']['phone_inside'])
        backend.close()
    def test_pause_and_cancel_preserve_scene_instead_of_resetting_objects(self):
        backend=MujocoAssemblyBackend(scenario());program=AssemblyProgram(backend,backend.config);program.start()
        for _ in range(250):program.tick()
        before=backend.pose('phone').copy();clock=backend.time
        program.pause()
        for _ in range(50):program.tick()
        np.testing.assert_allclose(backend.pose('phone'),before);self.assertEqual(backend.time,clock)
        program.resume();program.tick();self.assertGreater(backend.time,clock)
        position=backend.pose('phone').copy();program.cancel();program.tick()
        np.testing.assert_allclose(backend.pose('phone'),position);self.assertEqual(program.state,'cancelled')
        backend.close()
    def test_simulation_config_cannot_be_reused_as_hardware_calibration(self):
        config=scenario();config['hardware_calibrated']=True
        with self.assertRaisesRegex(ValueError,'实测标定'):validate_hardware(config)
    def test_simulation_worker_rejects_hardware_mode_without_constructing_robot_adapter(self):
        from unittest.mock import patch
        from grasp_execution_service import run as run_service
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/'request.json').write_text(json.dumps({'kind':'phone_assembly_sim','execution_mode':'hardware'}))
            with patch('grasp_execution_service.RosBackend',side_effect=AssertionError('must never create hardware')):
                with self.assertRaisesRegex(ValueError,'禁止使用真机'):run_service(directory)


if __name__=='__main__':unittest.main()
