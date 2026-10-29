"""Actual subprocess tests in kinematic mode. Never initialize ROS or hardware."""
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
import numpy as np
from grasp_execution_client import ProcessRunner
from host_arm_ik import HostArmIK,_resolve_urdf


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.runners=[]
        python=Path(__file__).parent/'.cache/grasp_ik_env/bin/python'
        self.profile={'urdf_path':_resolve_urdf(),'planner_python':str(python)}
    def tearDown(self):
        for runner in self.runners:
            if runner.process.poll() is None:
                runner.process.terminate()
                try:runner.process.wait(timeout=4)
                except Exception:runner.process.kill();runner.process.wait()
            runner.log.close()
        self.tmp.cleanup()
    def create(self,kind='arm'):
        request={'kind':kind,'execution_mode':'simulation','profile':self.profile,
                 'seed':[0.]*16,'hands':{'left':[0.]*6,'right':[0.]*6}}
        if kind=='hand':request.update(side='right',target=[.1]*6)
        else:
            ik=HostArmIK(self.profile['urdf_path']);end=np.zeros(16);end[9]=.03
            ik.ee_pose(end,'right');T=ik.data.oMf[ik._right_fid].homogeneous.tolist()
            request['plan']={'arm_side':'right','joint_waypoints':[[0.]*16,end.tolist()],
                'times_s':[0.,3.],'target':T,'urdf_sha256':hashlib.sha256(Path(self.profile['urdf_path']).read_bytes()).hexdigest()}
        runner=ProcessRunner(request,self.tmp.name);self.runners.append(runner);return runner
    def poll(self,runner,states,timeout=6):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            runner.tick()
            if runner.state in states:return
            time.sleep(.02)
        self.fail('executor timeout: '+runner.state+' '+runner.message+' '+(runner.directory/'service.log').read_text()[-500:])
    def test_hand_completes_in_separate_process_with_feedback_trace(self):
        runner=self.create('hand');self.poll(runner,{'complete','stop_unconfirmed'})
        self.assertEqual(runner.state,'complete',runner.message)
        self.assertTrue((runner.directory/'telemetry.jsonl').is_file())
        env=json.loads((runner.directory/'environment.json').read_text())
        self.assertEqual(env['execution_mode'],'simulation')
        self.assertEqual(env['io_backend'],'KinematicBackend')
        from tools.export_grasp_run import export
        metadata=export(runner.directory,Path(self.tmp.name)/'export')
        self.assertEqual(metadata['final_state'],'complete')
        self.assertIsNone(metadata['physical_task_success'])
        self.assertIn('right.command_hand0',(Path(self.tmp.name)/'export/episode.csv').read_text())
    def test_pause_resume_and_cancel_are_acknowledged(self):
        runner=self.create();self.poll(runner,{'running','stop_unconfirmed'})
        self.assertEqual(runner.state,'running',runner.message)
        runner.pause();self.poll(runner,{'paused','stop_unconfirmed'})
        self.assertEqual(runner.state,'paused',runner.message)
        status=json.loads((runner.directory/'status.json').read_text());q=status['feedback']['q16']
        for _ in range(10):runner.tick();time.sleep(.02)
        np.testing.assert_allclose(json.loads((runner.directory/'status.json').read_text())['feedback']['q16'],q)
        runner.resume();self.poll(runner,{'running','stop_unconfirmed'})
        self.assertEqual(runner.state,'running',runner.message)
        runner.stop();self.poll(runner,{'stopped','stop_unconfirmed'})
        self.assertEqual(runner.state,'stopped',runner.message)
    def test_gui_stall_stops_without_gui_timer(self):
        runner=self.create();self.poll(runner,{'running','stop_unconfirmed'})
        self.assertEqual(runner.state,'running',runner.message)
        time.sleep(1.9)  # Deliberately withhold GUI polling/heartbeat.
        self.poll(runner,{'failed','stop_unconfirmed','complete'})
        self.assertEqual(runner.state,'failed',runner.message)
        self.assertIn('心跳',runner.message)
    def test_cancel_during_startup_sends_no_simulated_hand_motion(self):
        runner=self.create('hand');runner.stop()
        self.poll(runner,{'stopped','stop_unconfirmed'})
        self.assertEqual(runner.state,'stopped',runner.message)


if __name__=='__main__':unittest.main()
