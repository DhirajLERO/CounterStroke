"""Lightweight regression coverage for repaired counterfactual baselines."""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
CIFAR_DIR = ROOT / "Cifar10"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CIFAR_DIR))

from common.cfrl_targets import extract_intended_targets  # noqa: E402
from explainers import cfproto_pt  # noqa: E402


class TinyTwoClassModel(nn.Module):
    """A one-dimensional, differentiable classifier with a boundary at zero."""

    def __init__(self) -> None:
        super().__init__()
        self.inputs: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.inputs.append(x.detach().cpu().clone())
        score = x.flatten(start_dim=1).mean(dim=1)
        return torch.stack((-2.0 * score, 2.0 * score), dim=1)


class CFProtoRepairTests(unittest.TestCase):
    def test_targeted_optimization_returns_nearest_clipped_success(self) -> None:
        model = TinyTwoClassModel()
        explainer = cfproto_pt.CounterfactualProtoPT(
            model,
            num_classes=2,
            gamma=0.0,
            theta=0.0,
            beta=0.0,
            c_init=10000.0,
            c_steps=1,
            learning_rate=0.05,
            max_iterations=12,
            feature_range=(-0.5, 0.5),
            ae_train_epochs=0,
        )
        fit_x = np.stack(
            (
                np.full((3, 32, 32), -0.4, dtype=np.float32),
                np.full((3, 32, 32), 0.4, dtype=np.float32),
            ),
            axis=0,
        )
        explainer.fit(fit_x, np.asarray([0, 1], dtype=np.int64))

        result = explainer.explain(
            np.full((1, 3, 32, 32), -0.8, dtype=np.float32),
            target_class=1,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.adv_class, result.target_class)
        self.assertEqual(result.adv_class, 1)
        self.assertGreaterEqual(float(result.adv.min()), -0.5)
        self.assertLessEqual(float(result.adv.max()), 0.5)

        # Calls are: original, then (pre-update, candidate) per iteration,
        # followed by one prediction of the selected candidate.
        visited_candidates = model.inputs[2:-1:2]
        successful = [
            candidate
            for candidate in visited_candidates
            if float(candidate.flatten().mean()) > 0.0
        ]
        self.assertGreater(len(successful), 1)
        original = torch.full((1, 3, 32, 32), -0.8, dtype=torch.float32)
        expected = min(
            successful,
            key=lambda candidate: float(((candidate - original) ** 2).sum()),
        )
        np.testing.assert_allclose(result.adv, expected.numpy(), rtol=0.0, atol=1e-6)

    def test_old_checkpoint_versions_are_rejected(self) -> None:
        model = TinyTwoClassModel()
        for old_version in (0, 1, 2):
            with self.subTest(version=old_version):
                with mock.patch.object(
                    cfproto_pt.torch,
                    "load",
                    return_value={"version": old_version},
                ):
                    with self.assertRaisesRegex(
                        ValueError,
                        r"expected version 3.*latent KDTree prototypes",
                    ):
                        cfproto_pt.CounterfactualProtoPT.load_checkpoint(
                            "old_cfproto.pt",
                            model,
                        )

    def test_noncanonical_checkpoint_range_is_rejected(self) -> None:
        model = TinyTwoClassModel()
        payload = {
            "version": 3,
            "init": {"feature_range": (-1e10, 1e10)},
        }
        with mock.patch.object(cfproto_pt.torch, "load", return_value=payload):
            with self.assertRaisesRegex(ValueError, r"expected \(-0.5, 0.5\)"):
                cfproto_pt.CounterfactualProtoPT.load_checkpoint(
                    "unbounded_cfproto.pt",
                    model,
                )


class CFRLTargetValidationTests(unittest.TestCase):
    def test_returns_validated_int64_intended_targets(self) -> None:
        manifest = [
            {"y_true": 0, "y_intended_target": 2},
            {"y_true": 2, "y_intended_target": 1},
        ]

        targets = extract_intended_targets(manifest, num_classes=3)

        np.testing.assert_array_equal(targets, np.asarray([2, 1], dtype=np.int64))
        self.assertEqual(targets.dtype, np.dtype(np.int64))

    def test_rejects_invalid_intended_targets(self) -> None:
        invalid_cases = (
            ([], "at least one"),
            ([{"y_true": 0}], "missing required field"),
            ([{"y_true": 0, "y_intended_target": True}], "non-integer"),
            ([{"y_true": 0, "y_intended_target": "1"}], "non-integer"),
            ([{"y_true": 0, "y_intended_target": -1}], r"in \[0, 3\)"),
            ([{"y_true": 0, "y_intended_target": 3}], r"in \[0, 3\)"),
            ([{"y_true": 1, "y_intended_target": 1}], "equal to y_true"),
        )
        for manifest, message in invalid_cases:
            with self.subTest(manifest=manifest):
                with self.assertRaisesRegex(ValueError, message):
                    extract_intended_targets(manifest, num_classes=3)


class StaticBaselineWiringTests(unittest.TestCase):
    @staticmethod
    def _source(relative_path: str) -> str:
        return (ROOT / relative_path).read_text(encoding="utf-8")

    @classmethod
    def _tree(cls, relative_path: str) -> ast.AST:
        return ast.parse(cls._source(relative_path), filename=relative_path)

    def test_imagenet_generators_use_manifest_intended_targets(self) -> None:
        contracts = {
            "Imagenet/generate_cfproto.py": (
                'y_tgt = int(entry["y_intended_target"])',
                "proto.explain(x_chw, target_class=y_tgt)",
            ),
            "Imagenet/generate_leapfactual.py": (
                'y_tgt = int(entry["y_intended_target"])',
                "leap.explain_nchw(x_chw, y_tgt)",
            ),
            "Imagenet/generate_cfrl.py": (
                "tgt_arr = extract_intended_targets(manifest, num_classes=100)",
                "cfrl_explain_batch_nhwc(explainer, x_batch, tgt_arr, CFRL_BATCH_SIZE)",
            ),
        }
        for relative_path, required_fragments in contracts.items():
            compact_source = " ".join(self._source(relative_path).split())
            with self.subTest(path=relative_path):
                for fragment in required_fragments:
                    self.assertIn(" ".join(fragment.split()), compact_source)

    def test_mnist_training_uses_train_pool_not_test_split(self) -> None:
        relative_path = "MNIST/train_explainer_models.py"
        source = " ".join(self._source(relative_path).split())
        self.assertIn(
            "x_fit = np.asarray(x_train[pool], dtype=np.float32)",
            source,
        )
        self.assertIn(
            "y_fit = np.asarray(y_train[pool], dtype=np.int64)",
            source,
        )
        self.assertIn("proto.fit(x_fit, y_fit)", source)
        self.assertIn(
            "train_cfrl_pytorch_explainer(x_fit, args.cfrl_steps, device, clf)",
            source,
        )

        loaded_names = {
            node.id
            for node in ast.walk(self._tree(relative_path))
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        self.assertTrue({"x_train", "y_train"}.issubset(loaded_names))
        self.assertTrue({"_x_te", "_y_te", "_ohe_te"}.isdisjoint(loaded_names))

    def test_mnist_generation_scripts_do_not_fit_on_eval_data(self) -> None:
        generation_scripts = (
            "MNIST/generate_cfproto.py",
            "MNIST/generate_cfrl.py",
            "MNIST/generate_leapfactual.py",
            "MNIST/generate_counterfactual_pt.py",
        )
        for relative_path in generation_scripts:
            fit_calls = [
                node
                for node in ast.walk(self._tree(relative_path))
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"fit", "fit_dataloader"}
            ]
            with self.subTest(path=relative_path):
                self.assertEqual(fit_calls, [])

    def test_all_cfproto_loaders_guard_the_repaired_checkpoint_format(self) -> None:
        contracts = {
            "MNIST/explainers/cfproto_pt.py": (
                "if version != CHECKPOINT_VERSION:",
                "with latent k-d-tree prototypes",
            ),
            "Cifar10/explainers/cfproto_pt.py": (
                "if version != 3:",
                "expected version 3 with latent KDTree prototypes",
            ),
            "Imagenet/explainers/cfproto_pt_rgb.py": (
                "if version != 4:",
                "expected version 4 with latent KDTree prototypes",
            ),
        }
        for relative_path, required_fragments in contracts.items():
            source = self._source(relative_path)
            with self.subTest(path=relative_path):
                for fragment in required_fragments:
                    self.assertIn(fragment, source)


if __name__ == "__main__":
    unittest.main()
