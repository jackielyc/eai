import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from click_pose import (Frame, FrameBuffer, geometric_preview, load_calibration,
                        metric_depth, object_points, validate_transform)


class ClickPoseTests(unittest.TestCase):
    def test_display_frame_selects_full_resolution_and_nearest_depth(self):
        buffer = FrameBuffer()
        full = np.zeros((80, 120, 3), dtype=np.uint8)
        preview = full[::2, ::2].copy()
        depth = np.full((80, 120), 700, dtype=np.uint16)
        buffer.add('rgb', full, 10, 'optical', 'bgr8', preview)
        buffer.add('depth', depth, 10.02, 'optical', '16UC1')
        buffer.add('depth', depth * 2, 10.2, 'optical', '16UC1')
        color, frame, skew = buffer.pair('rgb', 'depth', preview)
        self.assertIs(color.image, full)
        self.assertAlmostEqual(skew, 0.02)
        np.testing.assert_allclose(metric_depth(frame, full.shape), 0.7)

    def test_no_depth_no_timestamp_and_stale_frames_rejected(self):
        buffer = FrameBuffer()
        image = np.zeros((20, 20, 3))
        buffer.add('rgb', image, 10, 'optical', 'bgr8')
        with self.assertRaisesRegex(ValueError, '未收到'):
            buffer.pair('rgb', 'depth', image)
        buffer.add('depth', image[:, :, 0], 0, 'optical', '32FC1')
        with self.assertRaisesRegex(ValueError, '时间戳'):
            buffer.pair('rgb', 'depth', image)
        buffer.add('depth', image[:, :, 0], 10.5, 'optical', '32FC1')
        with self.assertRaisesRegex(ValueError, '不同步'):
            buffer.pair('rgb', 'depth', image)
        with self.assertRaisesRegex(ValueError, '过期'):
            buffer.pair('rgb', 'depth', image, max_age=-1)

    def test_real_depth_projection_and_invalid_fraction(self):
        depth = np.full((20, 20), 0.8, np.float32)
        K = np.array([[100, 0, 10], [0, 100, 10], [0, 0, 1]])
        points, _, quality = object_points(np.ones((20, 20), bool), depth, K, (10, 10))
        np.testing.assert_allclose(points[0], [-0.08, -0.08, 0.8], atol=1e-6)
        self.assertEqual(quality['point_count'], 400)
        depth[:15] = np.nan
        with self.assertRaisesRegex(ValueError, '深度不足'):
            object_points(np.ones((20, 20), bool), depth, K, (10, 10))

    def test_depth_alignment_and_encoding_not_guessed(self):
        frame = Frame(np.ones((10, 10), np.uint16), 1, 1, 'optical', 'mono16')
        with self.assertRaisesRegex(ValueError, '编码'):
            metric_depth(frame, (10, 10, 3))
        with self.assertRaisesRegex(ValueError, '尺寸'):
            metric_depth(frame, (20, 20, 3))

    def test_calibration_requires_registration_and_exact_resolution(self):
        config = {'cameras': {'rgb': {'width': 20, 'height': 10,
                  'K': [[100, 0, 10], [0, 100, 5], [0, 0, 1]],
                  'rectified': True, 'depth_registered_to_color': True,
                  'optical_frame': 'optical', 'depth_topic': 'depth'}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'calibration.json'
            path.write_text(json.dumps(config))
            _, K = load_calibration(path, 'rgb', (10, 20, 3))
            self.assertEqual(K[0, 0], 100)
            with self.assertRaisesRegex(ValueError, '分辨率'):
                load_calibration(path, 'rgb', (5, 10, 3))
            config['cameras']['rgb']['depth_registered_to_color'] = False
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, '注册'):
                load_calibration(path, 'rgb', (10, 20, 3))

    def test_geometric_preview_is_not_a_grasp_and_transform_must_be_rigid(self):
        points = np.random.default_rng(42).normal(size=(400, 3)) * [0.1, 0.05, 0.02]
        points += [0.1, 0.2, 0.8]
        preview = geometric_preview(points)
        self.assertTrue(preview['orientation_ambiguous'])
        self.assertEqual(preview['pose_kind'], 'geometry_preview')
        validate_transform(preview['T_camera_object'])
        invalid = np.eye(4)
        invalid[0, 0] = -1
        with self.assertRaises(ValueError):
            validate_transform(invalid)


class ClickIntegrationTests(unittest.TestCase):
    def test_ui_snapshot_runs_without_robot_commands(self):
        # Execute the actual UI orchestration method with a fake display and
        # deterministic segmenter; no ROS node, Qt window or robot is created.
        import ast
        import os
        import threading
        import time
        from types import SimpleNamespace as NS
        import cv2
        import click_pose
        from tests.source_helpers import method_tree
        source = '<CameraTopicWindow._run_click_pose>'
        method = method_tree('CameraTopicWindow', '_run_click_pose', unqualify=True)
        calls = [n.func.attr for n in ast.walk(method) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute)]
        self.assertNotIn('_store_segment_target', calls)
        self.assertNotIn('publish', calls)
        class Panel:
            pass
        class Field:
            def __init__(self, text=''): self.value = text
            def text(self): return self.value
            def setPlainText(self, text): self.value = text
        class ImmediateThread:
            def __init__(self, target, **kw): self.target = target
            def start(self): self.target()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'calibration.json'
            path.write_text(json.dumps({'cameras': {'rgb': {
                'width': 40, 'height': 30, 'K': [[100, 0, 20], [0, 100, 15], [0, 0, 1]],
                'rectified': True, 'depth_registered_to_color': True,
                'optical_frame': 'optical', 'depth_topic': 'depth'}}}))
            rgb = np.zeros((30, 40, 3), np.uint8)
            preview = rgb[::2, ::2].copy()
            buffer = FrameBuffer()
            buffer.add('rgb', rgb, 100, 'optical', 'bgr8', preview)
            buffer.add('depth', np.full((30, 40), 800, np.uint16), 100.01, 'optical', '16UC1')
            panel = Panel()
            panel._latest_image = preview
            results = []
            owner = NS(_click_pose_busy=False, panels={'rgb': panel},
                       node=NS(click_frames=buffer, enabled_topics={'rgb', 'depth'}),
                       click_pose_calibration_edit=Field(str(path)), click_pose_result_edit=Field(),
                       _click_pose_bridge=NS(finished=NS(emit=results.append)),
                       status_bar=NS(showMessage=lambda _: None))
            context = dict(np=np, os=os, time=time, json=json, click_pose=click_pose, cv2=cv2,
                           CameraPanel=Panel, threading=NS(Thread=ImmediateThread), EAI_DIR=directory,
                           SAM3_BACKEND_GEOMETRY='geometry', POSE_BACKEND_PCA='pca',
                           get_segment_settings=lambda: NS(backend='geometry'),
                           get_pose_settings=lambda: NS(backend='pca'),
                           segment_object_at_click=lambda *args, **kw: (np.ones((30, 40), bool), 'test'))
            module = ast.Module(body=[method], type_ignores=[])
            exec(compile(ast.fix_missing_locations(module), str(source), 'exec'), context)
            context['_run_click_pose'](owner, 'rgb', 5, 6, preview)
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0]['ok'], results[0])
            self.assertEqual(results[0]['seed_uv'], [11, 13])
            self.assertFalse(results[0]['execution_ready'])
            self.assertEqual(results[0]['object_pose']['pose_kind'], 'geometry_preview')
            archives = list(Path(directory).glob('.cache/click_pose/*/result.json'))
            self.assertEqual(len(archives), 1)
            saved = json.loads(archives[0].read_text())
            self.assertNotIn('image', saved)
            self.assertAlmostEqual(saved['object_pose']['T_camera_object'][2][3], .8, places=5)


class GraspAdapterTests(unittest.TestCase):
    def test_official_api_adapter_orders_candidates_and_rejects_wrong_hand(self):
        from types import SimpleNamespace as NS, ModuleType
        from unittest.mock import patch
        import graspgen_worker
        class Tensor:
            def __init__(self, data): self.data = np.asarray(data)
            def detach(self): return self
            def cpu(self): return self
            def numpy(self): return self.data
        class Sampler:
            def __init__(self, cfg): pass
            @staticmethod
            def run_inference(points, sampler, **options):
                poses = np.repeat(np.eye(4)[None], 2, axis=0)
                poses[:, 2, 3] = [.6, .7]
                return Tensor(poses), Tensor([.2, .8])
        module = ModuleType('grasp_gen.grasp_server')
        module.GraspGenSampler = Sampler
        module.load_grasp_cfg = lambda _: NS(data=NS(gripper_name='test_gripper'))
        with patch.dict('sys.modules', {'grasp_gen': ModuleType('grasp_gen'),
                                       'grasp_gen.grasp_server': module}):
            result = graspgen_worker.infer(np.zeros((100, 3)), 'test.yaml', 'test_gripper')
            self.assertEqual(result['candidates'][0]['model_score'], .8)
            self.assertFalse(result['collision_checked'])
            self.assertFalse(result['ik_checked'])
            with self.assertRaisesRegex(ValueError, '手型不匹配'):
                graspgen_worker.infer(np.zeros((100, 3)), 'test.yaml', 'dexterous_hand')


if __name__ == '__main__':
    unittest.main()
