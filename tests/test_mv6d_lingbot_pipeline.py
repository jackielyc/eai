"""Run with ROS Humble sourced; all inference and robot I/O are mocked."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import cv2
import numpy as np
import show_camera_topics as m

m._ensure_cv2_numpy()


class InlineThread:
    def __init__(self, target, **_):
        self.target = target
    def start(self):
        self.target()


class Harness:
    _mv6d_enabled_methods = m.CameraTopicWindow._mv6d_enabled_methods
    _start_mv6d_estimate = m.CameraTopicWindow._start_mv6d_estimate
    _estimate_view_6d = staticmethod(m.CameraTopicWindow._estimate_view_6d)
    _pose_on_mask = staticmethod(m.CameraTopicWindow._pose_on_mask)
    _collect_method_poses = m.CameraTopicWindow._collect_method_poses
    _mv6d_lingbot_model_id = m.CameraTopicWindow._mv6d_lingbot_model_id

    def __init__(self, enabled=True, backend='pca', missing_depth=False, fail=None):
        self.events, self.outputs, self.inputs, self.model_depths = [], [], [], []
        self.backend, self.fail = backend, fail
        self.rgb = np.full((4, 6, 3), 37, dtype=np.uint8)
        self.raw = None if missing_depth else np.full((2, 3), 1000, dtype=np.uint16)
        self.refined = {tag: np.full((4, 6), n, np.float32)
                        for tag, n in [('head', 2), ('left_wrist', 3), ('right_wrist', 4)]}
        self.mv6d_wrist_combo = NS(currentData=lambda: 'left')
        self.mv6d_method_combo = NS(currentData=lambda: backend)
        self.mv6d_lingbot_depth_check = NS(isChecked=lambda: enabled)
        self.lingbot_depth_model_edit = NS(text=lambda: '/local/model.pt')
        self.lingbot_depth_model_combo = NS(currentData=lambda: 'ignored-default')
        self.lingbot_depth_device_combo = NS(currentData=lambda: 'cuda')
        self.mv6d_head_u_spin = NS(value=lambda: 2)
        self.mv6d_head_v_spin = NS(value=lambda: 1)
        self._mv6d_busy = False
        self._mv6d_bridge = NS(finished=NS(emit=self.outputs.append))
        self.node = NS(get_tf_buffer=lambda: None,
                       get_intrinsics=lambda *args: (100., 100., 3., 2.),
                       resolve_segment_camera_frame=lambda *args: 'camera')

    def _ensure_mv6d_camera_feeds(self): pass
    def _resolve_wrist_color_topic(self, side): return '/camera/' + side + '_color'
    def _pick_mv6d_color_frame(self, topic):
        return (topic if topic.startswith('/') else self._resolve_wrist_color_topic(topic), self.rgb)
    def _get_paired_depth_frame(self, topic): return self.raw
    def _get_paired_depth_topic_name(self, topic): return topic.replace('color', 'depth')
    def _refine_depth_with_lingbot_sync(self, rgb, **kw):
        self.events.append('lingbot:' + kw['tag'])
        self.inputs.append((rgb, kw))
        if kw['tag'] == self.fail: raise RuntimeError('inference failed')
        kw['result_callback']({'ok': True, 'tag': kw['tag'], 'color_topic': kw['topic']})
        return self.refined[kw['tag']]
    def _compute_mv6d_snapshot(self, **kw):
        self.snapshot_inputs = kw
        result = self._estimate_view_6d(kw['head_bgr'], kw['head_depth'], 2, 1,
                                       100., 100., 3., 2., method_tag='head', pose_backend=kw['pose_backend'])
        return NS(head_result=result, notes='')
    def _mv6d_pose_in_base(self, snap): return ('selected-pose',)
    def _camera_pose_to_base(self, snap, result): return ('other-pose',)
    def segment(self, rgb, *args, **kw):
        self.events.append('sam3')
        np.testing.assert_array_equal(rgb, self.rgb)
        return np.ones(rgb.shape[:2], dtype=bool), 'mock-sam3'
    def record(self, name, depth, mask):
        self.model_depths.append((name, depth))
        return NS(mask=mask)
    def run(self):
        with tempfile.TemporaryDirectory() as root, patch.object(m, 'EAI_DIR', root), \
             patch.object(m.threading, 'Thread', InlineThread), \
             patch.object(m, 'run_sam3_segmentation', self.segment), \
             patch.object(m, 'compute_stereo_segment_6d_pose',
                          lambda mask, depth, *a, **kw: self.record('pca', depth, mask)), \
             patch.object(m, 'run_foundationpose_estimation',
                          lambda rgb, depth, mask, *a, **kw: self.record('foundationpose', depth, mask)), \
             patch.object(m, 'run_any6d_estimation',
                          lambda rgb, depth, mask, *a, **kw: self.record('any6d', depth, mask)), \
             patch.object(m, 'object6d_from_foundationpose', lambda *a, **kw: a[8]):
            self._start_mv6d_estimate(silent=True)
        return self.outputs[-1]


class PipelineTests(unittest.TestCase):
    def test_multi_target_batch_reuses_capture_and_lingbot_depth(self):
        h = Harness()
        h._mv6d_batch_active = True
        h._mv6d_batch_capture = None
        h._mv6d_batch_depth_cache = None
        self.assertTrue(h.run()['ok'])
        first_events = list(h.events)
        h._mv6d_busy = False  # Real UI callback clears this before advancing the batch.
        self.assertTrue(h.run()['ok'])
        self.assertEqual(first_events,
                         ['lingbot:head','lingbot:left_wrist','lingbot:right_wrist','sam3'])
        self.assertEqual(h.events, first_events + ['sam3'])

    def test_all_backends_share_refined_depth_even_with_sensor_depth(self):
        for backend in ('pca', 'foundationpose', 'any6d'):
            with self.subTest(backend=backend):
                h = Harness(backend=backend)
                result = h.run()
                self.assertTrue(result['ok'], result.get('error'))
                self.assertEqual(h.events, ['lingbot:head', 'lingbot:left_wrist', 'lingbot:right_wrist', 'sam3'])
                self.assertEqual({name for name, _ in h.model_depths}, {'pca', 'foundationpose', 'any6d'})
                for _, depth in h.model_depths: np.testing.assert_array_equal(depth, h.refined['head'])
                for key, tag in [('head_depth','head'),('left_wrist_depth','left_wrist'),('right_wrist_depth','right_wrist')]:
                    self.assertIs(h.snapshot_inputs[key], h.refined[tag])
                for rgb, kw in h.inputs:
                    np.testing.assert_array_equal(kw['depth'], h.raw)
                    self.assertIsNot(kw['depth'], h.raw)
                    self.assertIsNot(rgb, h.rgb)
                    self.assertEqual(kw['model'], '/local/model.pt')
                self.assertEqual(result['depth_source'], 'lingbot')
                self.assertEqual(len(result['lingbot_results']), 3)

    def test_unchecked_preserves_sensor_pipeline(self):
        h = Harness(enabled=False)
        self.assertTrue(h.run()['ok'])
        self.assertEqual(h.events, ['sam3'])
        for _, depth in h.model_depths: np.testing.assert_array_equal(depth, h.raw.astype(np.float32) / 1000.)

    def test_missing_depth_still_uses_lingbot_output(self):
        h = Harness(missing_depth=True)
        self.assertTrue(h.run()['ok'])
        for _, kw in h.inputs: self.assertIsNone(kw['depth'])
        for _, depth in h.model_depths: np.testing.assert_array_equal(depth, h.refined['head'])

    def test_cancel_after_refinement_stops_before_sam3(self):
        h = Harness()
        refine = h._refine_depth_with_lingbot_sync
        def cancelled(*args, **kwargs):
            result = refine(*args, **kwargs)
            kwargs['cancel_event'].set()
            return result
        h._refine_depth_with_lingbot_sync = cancelled
        result = h.run()
        self.assertFalse(result['ok'])
        self.assertIn('停止', result['error'])
        self.assertNotIn('sam3', h.events)

    def test_failure_stops_before_sam3_and_pose(self):
        for fail in ('head', 'left_wrist', 'right_wrist'):
            h = Harness(fail=fail)
            result = h.run()
            self.assertFalse(result['ok'])
            self.assertIn('本轮已停止', result['error'])
            self.assertNotIn('sam3', h.events)
            self.assertEqual(h.model_depths, [])


class DepthTransportTests(unittest.TestCase):
    def invoke(self, raw, fake_output=None, status=0):
        self.commands, self.saved_depth, self.results, self.intrinsic_calls = [], None, [], []
        with tempfile.TemporaryDirectory() as tmp:
            def run(cmd, **kw):
                self.commands = cmd
                out = Path(cmd[cmd.index('--out')+1])
                if '--depth' in cmd:
                    self.saved_depth = cv2.imread(cmd[cmd.index('--depth')+1], cv2.IMREAD_UNCHANGED)
                shape = cv2.imread(cmd[cmd.index('--rgb')+1]).shape[:2]
                np.save(out/'depth_refined.npy', np.full(shape,2.0,np.float32) if fake_output is None else fake_output)
                return NS(returncode=status, stdout='[RESULT] '+json.dumps({'ok':status==0,'point_cloud':str(out/'point_cloud.ply')}),stderr='worker failure')
            def intr(*args):
                self.intrinsic_calls.append(args);return 100.,100.,3.,2.
            with patch.object(m, 'LINGBOT_DEPTH_CACHE_DIR', tmp), patch.object(m.subprocess, 'run', run):
                return m.CameraTopicWindow._refine_depth_with_lingbot_sync(
                    NS(), np.zeros((4,6,3),np.uint8), topic='/head/color',depth_topic='/head/depth',
                    depth=raw, seed_u=2,seed_v=1,sparse_z_m=.45,get_intrinsics=intr,
                    model='/local/model.pt',device='cuda',result_callback=self.results.append)

    def test_sensor_mm_and_meter_inputs_are_passed_to_worker(self):
        for raw in (np.full((2,3),1250,np.uint16),np.full((2,3),1.25,np.float32)):
            out=self.invoke(raw)
            self.assertIn('--depth', self.commands)
            self.assertNotIn('--allow-empty-depth',self.commands)
            self.assertNotIn('--no-mask',self.commands)
            np.testing.assert_array_equal(self.saved_depth,np.full((2,3),1250,np.uint16))
            np.testing.assert_array_equal(out,np.full((2,3),2.0,np.float32))
            self.assertEqual(self.intrinsic_calls,[('/head/depth',3,2)])
            self.assertEqual(self.results[0]['color_topic'],'/head/color')

    def test_missing_depth_uses_explicit_sparse_prior(self):
        self.invoke(None)
        self.assertIn('--allow-empty-depth',self.commands)
        self.assertNotIn('--depth',self.commands)

    def test_out_of_range_sensor_depth_is_rejected_before_inference(self):
        with self.assertRaisesRegex(RuntimeError, '没有有效测量'):
            self.invoke(np.full((4,6),50.,np.float32))
        self.assertEqual(self.commands, [])

    def test_invalid_output_and_failed_worker_do_not_fallback(self):
        for out in (np.zeros((4,6)),np.full((4,6),np.nan),np.ones((2,3))):
            with self.assertRaises(RuntimeError):self.invoke(np.ones((4,6)),out)
        with self.assertRaisesRegex(RuntimeError,'LingBot-Depth 失败'):self.invoke(None,status=1)


class ResultRoutingTests(unittest.TestCase):
    def test_cloud_uses_capture_topic_not_last_manual_selection(self):
        calls = []
        view = NS(set_lingbot_depth_from_ply=lambda path, **kw: calls.append((path, kw)))
        owner = NS(_robot_view=view, _lingbot_depth_last_color_topic='/camera/left_color')
        with tempfile.NamedTemporaryFile(suffix='.ply') as f:
            m.CameraTopicWindow._show_lingbot_depth_on_robot_view(
                owner, {'point_cloud': f.name, 'color_topic': '/camera/head_color'}, silent=True)
        self.assertEqual(calls[0][1]['color_topic'], '/camera/head_color')

    def test_async_result_prefers_head_and_reports_lingbot_failure(self):
        displayed, errors, cleared = [], [], []
        owner = NS(_mv6d_last_snap=None, _clear_lingbot_depth_result=lambda: cleared.append(True),mv6d_once_btn=NS(setEnabled=lambda _: None, setText=lambda _: None),
                   _on_lingbot_depth_result=displayed.append,
                   mv6d_result_edit=NS(setPlainText=errors.append),
                   status_bar=NS(showMessage=lambda _: None))
        head = {'tag': 'head', 'ok': True}
        m.CameraTopicWindow._on_mv6d_estimate_finished(owner, {
            'ok': False, 'silent': False, 'depth_source': 'lingbot', 'error': 'refinement failed',
            'lingbot_results': [{'tag': 'left_wrist', 'ok': True}, head],
        })
        self.assertEqual(displayed, [])
        self.assertEqual(cleared, [])  # Failed attempts preserve previous cached cloud.
        self.assertIn('refinement failed', errors[0])
        m.CameraTopicWindow._on_mv6d_estimate_finished(owner, {
            'ok': True, 'silent': False,
            'lingbot_results': [{'tag': 'left_wrist', 'ok': True}, head],
        })
        self.assertEqual(displayed, [head])


if __name__ == '__main__':
    unittest.main()
