"""Bounded, lossless raw trajectory writer. No dataset conversion or training."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time

import numpy as np


class RolloutRecorder:
    def __init__(self, root, metadata):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.metadata = dict(metadata, format="lingbot-raw-rollout-v1",
                             purpose="diagnostic_collection_not_approved_training_data",
                             status="incomplete", observation_timing="before_executed_action",
                             created_unix=time.time(), frames=0)
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.pending = deque()
        self.steps = 0
        self._save_metadata()

    def _save_metadata(self):
        temporary = self.root / "episode.json.tmp"
        temporary.write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.root / "episode.json")

    @staticmethod
    def _write(path, arrays):
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as output:
            np.savez_compressed(output, **arrays)
        os.replace(temporary, path)

    def record(self, observation, action, chunk, offset, env_step):
        # At most two in-flight frames; errors propagate, no silent dropping.
        if len(self.pending) >= 2:
            self.pending.popleft().result()
        arrays = {}
        for name, camera in (("rgb_head", "head_camera"), ("rgb_left", "left_camera"),
                             ("rgb_right", "right_camera")):
            image = np.asarray(observation["observation"][camera]["rgb"])
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"Invalid RGB schema: {camera}")
            arrays[name] = image.copy()
        state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        if state.ndim != 1 or action.ndim != 1 or not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError("State/action must be finite raw vectors")
        if self.steps == 0:
            self.metadata.update(state_dim=int(state.size), action_dim=int(action.size))
        arrays.update(state=state.copy(), action=action.copy(), chunk=np.int64(chunk),
                      offset=np.int64(offset), env_step=np.int64(env_step),
                      timestamp_unix=np.float64(time.time()))
        self.pending.append(self.pool.submit(self._write, self.root / f"step_{self.steps:06d}.npz", arrays))
        self.steps += 1

    def finish(self, final_observation, success, env_step, step_limit):
        try:
            while self.pending:
                self.pending.popleft().result()
            state = np.asarray(final_observation["joint_action"]["vector"], dtype=np.float32)
            final_arrays = {"state": state, **{
                key: np.asarray(final_observation["observation"][camera]["rgb"])
                for key, camera in (("rgb_head", "head_camera"), ("rgb_left", "left_camera"), ("rgb_right", "right_camera"))}}
            self._write(self.root / "final_observation.npz", final_arrays)
            self.metadata.update(status="complete", success=bool(success), reward=float(bool(success)),
                                 terminated=bool(success), truncated=bool(not success and env_step >= step_limit),
                                 final_env_step=int(env_step), step_limit=int(step_limit), frames=self.steps,
                                 finished_unix=time.time())
            self._save_metadata()
        finally:
            self.pool.shutdown(wait=True)


def validate_episode(root):
    root = Path(root)
    metadata = json.loads((root / "episode.json").read_text(encoding="utf-8"))
    frames = metadata["frames"]
    if metadata["status"] != "complete" or frames < 1:
        raise ValueError("Incomplete/empty raw rollout")
    paths = sorted(root.glob("step_*.npz"))
    if [p.name for p in paths] != [f"step_{i:06d}.npz" for i in range(frames)]:
        raise ValueError("Missing/duplicate rollout frames")
    for i, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as data:
            if int(data["env_step"]) != i:
                raise ValueError("Noncontiguous executed-action steps")
            for key, dimension in (("state", metadata["state_dim"]), ("action", metadata["action_dim"])):
                if data[key].shape != (dimension,) or not np.isfinite(data[key]).all():
                    raise ValueError("Invalid state/action shape or values")
            if not {"rgb_head", "rgb_left", "rgb_right"}.issubset(data.files):
                raise ValueError("Missing RGB camera")
    if metadata["final_env_step"] != frames or not (root / "final_observation.npz").is_file():
        raise ValueError("Missing terminal observation or executed-step mismatch")
    return metadata
