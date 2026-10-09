# Copyright (c) 2026-2027 zh
"""
内容：
    验证深度工具的单位换算、输入检查、误差计算和后端失败处理。
用法：
    python -m unittest discover -s legged_gym/tests -p test_depth_tools.py
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from tool import depth_tools as depth


class DepthToolsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def input_args(self, *extra):
        return depth.parse_args(["compare", "--root", str(self.root), *extra])

    def save_pair(self, left, right):
        for backend, values in (("isaac", left), ("warp", right)):
            directory = self.root / backend
            directory.mkdir()
            np.save(directory / "frame.npy", values)

    def test_known_metrics(self):
        difference, metrics = depth.depth_metrics(np.array([[1, 2]]), np.array([[1, 1]]))
        self.assertEqual(metrics["pixels"], 2)
        self.assertEqual(metrics["mae_m"], 0.5)
        self.assertAlmostEqual(metrics["rmse_m"], np.sqrt(0.5))
        np.testing.assert_array_equal(difference, [[0, 1]])

    def test_nonzero_near_png(self):
        path = self.root / "depth.png"
        cv2.imwrite(str(path), np.array([[0, 255]], dtype=np.uint8))
        values = depth.read_depth(path, "auto", depth.CameraSpec(near_clip=0.2, far_clip=2.0))
        np.testing.assert_allclose(values, [[0.2, 2.0]])

    def test_normalized_input(self):
        path = self.root / "depth.npy"
        np.save(path, np.array([[-0.5, 0.0, 0.5]], dtype=np.float32))
        values = depth.read_depth(path, "normalized", depth.CameraSpec(near_clip=0.2, far_clip=2.0))
        np.testing.assert_allclose(values, [[0.2, 1.1, 2.0]])

    def test_unknown_npy_units(self):
        self.save_pair(np.ones((2, 2)), np.ones((2, 2)))
        with self.assertRaisesRegex(ValueError, "units are unknown"):
            depth.load_depths(self.input_args())

    def test_raw_npy_preferred_and_csv(self):
        self.save_pair(np.ones((2, 2)), np.full((2, 2), 1.1))
        for backend in ("isaac", "warp"):
            cv2.imwrite(str(self.root / backend / "frame.png"), np.zeros((2, 2), dtype=np.uint8))
        values, spec, source = depth.load_depths(self.input_args("--encoding", "meters"))
        rows = depth.compare(values, spec, source, self.root)
        self.assertAlmostEqual(rows[0]["mae_m"], 0.1)
        self.assertTrue((self.root / "compare" / "metrics.csv").exists())
        self.assertTrue(all(path.endswith(".npy") for path in source["sources"].values()))

    def test_shape_mismatch(self):
        self.save_pair(np.ones((2, 2)), np.ones((3, 2)))
        with self.assertRaisesRegex(ValueError, "Shape mismatch"):
            depth.load_depths(self.input_args("--encoding", "meters"))

    def test_missing_input(self):
        with self.assertRaisesRegex(ValueError, "No backend directories"):
            depth.load_depths(self.input_args())

    def test_nonfinite_input(self):
        path = self.root / "depth.npy"
        np.save(path, np.array([[np.nan]]))
        with self.assertRaises(ValueError):
            depth.read_depth(path, "meters", depth.CameraSpec())

    def test_invalid_camera(self):
        for spec in (depth.CameraSpec(width=0), depth.CameraSpec(near_clip=2, far_clip=2), depth.CameraSpec(far_clip=float("nan"))):
            with self.assertRaises(ValueError):
                spec.validate()

    def test_metadata_conflict(self):
        depth.save_json(self.root / "summary.json", {"camera": {"far_clip": 3.0}})
        with self.assertRaisesRegex(ValueError, "conflicts"):
            depth.load_depths(self.input_args("--far_clip", "2"))

    def test_legacy_summary(self):
        self.save_pair(np.zeros((2, 2)), np.zeros((2, 2)))
        depth.save_json(self.root / "summary.json", {"camera": {"far_clip": 2.0}, "yaw_degrees": [0]})
        values, _, _ = depth.load_depths(self.input_args())
        np.testing.assert_array_equal(values["isaac"]["frame"], np.ones((2, 2)))

    def test_generation_failure_is_not_success(self):
        args = depth.parse_args(["generate", "--backend", "warp", "--out", str(self.root)])
        with patch.object(depth.subprocess, "run", return_value=SimpleNamespace(returncode=-11)):
            with self.assertRaisesRegex(RuntimeError, "Generation failed"):
                depth.generate(args)
        metadata = json.loads((self.root / "summary.json").read_text())
        self.assertEqual(metadata["backends"]["warp"]["status"], "failed")

    def test_generation_does_not_overwrite(self):
        (self.root / "existing.txt").touch()
        args = depth.parse_args(["generate", "--out", str(self.root)])
        with self.assertRaises(FileExistsError):
            depth.generate(args)

    def test_yaw_order(self):
        self.assertEqual(sorted(["wall_yaw_p30", "wall_yaw_m30", "wall_yaw_p0"], key=depth.frame_key), ["wall_yaw_m30", "wall_yaw_p0", "wall_yaw_p30"])

    def test_plot_panels_only_pdf(self):
        values = np.array([[0.2, 0.4], [0.6, 0.8]])
        panels = {"Isaac Gym": {"frame": (values, "2 x 2 pixels")}}
        args = SimpleNamespace(rows_per_page=3, dpi=72)
        paths = depth.plot_panels(panels, ["frame"], "Depth", "Depth (m)", "gray", 0, 2, self.root, "depth", args)
        self.assertEqual(paths, [str(self.root / "depth_01.pdf")])
        self.assertTrue((self.root / "depth_01.pdf").read_bytes().startswith(b"%PDF"))
        self.assertFalse(list(self.root.glob("*.png")))

    def test_plot_uses_raw_depth_and_gray(self):
        values = np.array([[0.25, 1.75]])
        args = SimpleNamespace(font_file=None, dpi=300)
        provenance = {"frames": ["frame"], "sources": {}}
        with patch.object(depth, "configure_plot", return_value="Times New Roman"), patch.object(depth, "plot_panels", return_value=[]) as panels:
            depth.plot({"isaac": {"frame": values}}, depth.CameraSpec(), provenance, self.root, args)
        arguments = panels.call_args.args
        self.assertIs(arguments[0]["Isaac Gym"]["frame"][0], values)
        self.assertEqual(arguments[4], "gray")
        self.assertEqual(arguments[5:7], (0.0, 2.0))


if __name__ == "__main__":
    unittest.main()
