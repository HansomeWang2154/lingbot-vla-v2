import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

PATH = Path(__file__).resolve().parents[1] / "experiment/robotwin/lingbot_rollout.py"
SPEC = importlib.util.spec_from_file_location("rollout", PATH)
rollout = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rollout)


def observation(value=0):
    return {"observation": {name: {"rgb": np.full((4, 5, 3), value, dtype=np.uint8)}
                            for name in ("head_camera", "left_camera", "right_camera")},
            "joint_action": {"vector": np.zeros(14, dtype=np.float32)}}


class RolloutTests(unittest.TestCase):
    def test_round_trip_and_no_aliasing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "episode"
            writer = rollout.RolloutRecorder(root, {"task": "grab_roller", "seed": 2100000})
            obs = observation(7)
            for i in range(3):
                writer.record(obs, np.ones(14), 0, i, i)
            obs["observation"]["head_camera"]["rgb"][:] = 255
            writer.finish(obs, True, 3, 400)
            metadata = rollout.validate_episode(root)
            self.assertEqual(metadata["frames"], 3)
            self.assertTrue(metadata["terminated"])
            with np.load(root / "step_000000.npz", allow_pickle=False) as frame:
                self.assertTrue((frame["rgb_head"] == 7).all())
                self.assertEqual(frame["action"].shape, (14,))
            (root / "step_000001.npz").unlink()
            with self.assertRaises(ValueError):
                rollout.validate_episode(root)

    def test_nonfinite_rejected_and_incomplete_marked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "episode"
            writer = rollout.RolloutRecorder(root, {})
            with self.assertRaises(ValueError):
                writer.record(observation(), np.full(14, np.nan), 0, 0, 0)
            writer.pool.shutdown(wait=True)
            self.assertEqual(json.loads((root / "episode.json").read_text())["status"], "incomplete")

    def test_timeout_is_truncated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "episode"
            writer = rollout.RolloutRecorder(root, {})
            writer.record(observation(), np.zeros(14), 0, 0, 0)
            writer.finish(observation(), False, 1, 1)
            self.assertTrue(rollout.validate_episode(root)["truncated"])


if __name__ == "__main__":
    unittest.main()
