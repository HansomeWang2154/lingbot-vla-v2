import ast
import typing
import unittest
from pathlib import Path


def _load_alignment_key_predicate():
    source_path = (
        Path(__file__).resolve().parents[1]
        / "lingbotvla"
        / "models"
        / "module_utils.py"
    )
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    selected = []
    for node in module.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name)
            and target.id == "_ALIGNMENT_CHECKPOINT_PREFIXES"
            for target in node.targets
        ):
            selected.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in {
            "_is_alignment_checkpoint_key",
            "_should_skip_disabled_alignment_checkpoint_key",
        }:
            selected.append(node)
    namespace = {
        "Optional": typing.Optional,
        "Dict": typing.Dict,
        "Any": typing.Any,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source_path), "exec"), namespace)
    return (
        namespace["_is_alignment_checkpoint_key"],
        namespace["_should_skip_disabled_alignment_checkpoint_key"],
    )


(
    _is_alignment_checkpoint_key,
    _should_skip_disabled_alignment_checkpoint_key,
) = _load_alignment_key_predicate()


class AlignmentCheckpointKeyTest(unittest.TestCase):
    def test_recognizes_optional_alignment_parameters(self):
        keys = (
            "model.depth_align_embs",
            "model.depth_align_head.projector.proj_in1.weight",
            "model.future_depth_align_head.projector.norm_out.bias",
            "model.current_video_align_embs",
            "model.future_video_align_head.projector.layers.0.0.to_q.weight",
            "model.current_shared_task_proj.weight",
            "model.future_shared_task_proj.bias",
            "model.future_video_cls_head.1.weight",
        )
        for key in keys:
            with self.subTest(key=key):
                self.assertTrue(_is_alignment_checkpoint_key(key))

    def test_keeps_strict_loading_for_action_and_backbone_parameters(self):
        keys = (
            "model.action_in_proj.weight",
            "model.qwenvl_with_expert.qwenvl.model.layers.0.self_attn.q_proj.weight",
            "model.depth_align_head_extra.weight",
        )
        for key in keys:
            with self.subTest(key=key):
                self.assertFalse(_is_alignment_checkpoint_key(key))

    def test_only_skips_alignment_keys_when_alignment_is_disabled(self):
        key = "model.current_video_align_embs"
        self.assertTrue(_should_skip_disabled_alignment_checkpoint_key(key, {}))
        self.assertTrue(_should_skip_disabled_alignment_checkpoint_key(key, None))
        self.assertFalse(
            _should_skip_disabled_alignment_checkpoint_key(
                key, {"use_future_video": True}
            )
        )
        self.assertFalse(
            _should_skip_disabled_alignment_checkpoint_key(
                "model.action_in_proj.weight", {}
            )
        )


if __name__ == "__main__":
    unittest.main()
