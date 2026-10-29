"""Offscreen UI integration. The only spawned executor is the contact simulator."""
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PyQt5.QtWidgets import QApplication
from grasp_execution_ui import GraspExecutionPanel
from test_grasp_execution_ui import FakeNode
from test_grasp_execution import example
from host_arm_ik import _resolve_urdf


class AssemblyUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.node=FakeNode();self.node._grasp_use_process_runner=True
        self.frames=[]
        self.owner=GraspExecutionPanel(self.node,lambda:False,simulation_display=lambda q,h:self.frames.append(q))
        profile,_=example();profile['urdf_path']=_resolve_urdf()
        profile['planner_python']=str(Path(__file__).parent/'.cache/grasp_ik_env/bin/python')
        path=Path(self.tmp.name)/'profile.json';path.write_text(json.dumps(profile));self.owner.profile_path.setText(str(path))
        self.panel=self.owner.phone_program_panel
        self.owner.resize(1280,1000);self.owner.show();self.app.processEvents()
    def tearDown(self):
        self.panel.timer.stop();self.owner.watchdog.stop()
        client=self.panel.client
        if client:
            if client.process.poll() is None:
                client.process.terminate()
                try:client.process.wait(timeout=5)
                except Exception:client.process.kill();client.process.wait()
            client.log.close()
        self.owner.close();self.owner.deleteLater();self.app.processEvents();self.tmp.cleanup()
    def wait(self,predicate,timeout=60):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            self.app.processEvents()
            if predicate():return
            time.sleep(.02)
        self.fail('UI timeout: '+self.panel.status.text())
    def test_primary_view_inspects_stages_without_skipping_execution(self):
        self.assertTrue(self.panel.isVisible());self.assertFalse(self.owner.assembly_advanced_body.isVisible())
        self.owner.task_flow._labels[2].clicked.emit(2)
        self.assertEqual(self.owner._workflow_step,0)
        self.assertIn('盒盖与盒体对齐',self.panel.detail.text())
        self.assertFalse(self.node.arm_commands);self.assertFalse(self.node.hand_commands)
    def test_hardware_start_is_blocked_by_unmeasured_parameters(self):
        self.panel.mode.setCurrentIndex(1);self.panel.hardware_confirm.setChecked(True);self.panel.start()
        self.assertIn('待实测标定',self.panel.status.text())
        self.assertIsNone(self.panel.program);self.assertFalse(self.node.arm_commands);self.assertFalse(self.node.hand_commands)
    def test_one_click_program_completes_contact_simulation_and_updates_flow(self):
        self.panel.speed.setValue(8);self.panel.load_scene()
        self.wait(lambda:self.panel.client.state in ('idle','failed','stop_unconfirmed'),timeout=45)
        self.assertEqual(self.panel.client.state,'idle',self.panel.status.text())
        self.panel.start()
        self.wait(lambda:self.panel.terminal_saved,timeout=60)
        self.assertEqual(self.panel.client.state,'complete',self.panel.status.text())
        self.assertTrue(self.panel.latest['goal']['success'])
        self.assertEqual(self.panel.progress.value(),26)
        self.assertEqual(self.owner.task_flow._current,3)
        self.assertEqual(self.owner._workflow_completed_steps,{0,1,2})
        self.assertIsNotNone(self.panel.frame.source)
        self.assertFalse(self.node.arm_commands);self.assertFalse(self.node.hand_commands)
        self.assertFalse(self.owner._program_active)
        result=json.loads((self.panel.client.directory/'result.json').read_text())
        self.assertEqual(result['hardware_commands'],0)
        self.assertGreater(result['ik']['body_collision_samples'],0)
        self.owner.grab().save('/tmp/phone_assembly_ui_completed.png')


if __name__=='__main__':unittest.main()
