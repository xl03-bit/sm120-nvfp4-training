"""Calibration policy on CPU and native mixed-precision gradients on SM120."""
import unittest
from importlib.util import find_spec
from unittest.mock import patch

import torch
from torch import nn

from precision_lora import PackedNVFP4Linear, calibrate_packed_layers


def frozen_stack(size):
    return nn.Sequential(*[
        nn.Linear(4, 4, dtype=torch.bfloat16) for _ in range(size)
    ]).requires_grad_(False)


class CalibrationTests(unittest.TestCase):
    @unittest.skipUnless(find_spec("nvfp4_training"), "full training package only")
    def test_training_package_export(self):
        from nvfp4_training import calibrate_packed_layers as exported
        self.assertIs(exported, calibrate_packed_layers)

    def test_cumulative_budget_and_rejected_identity(self):
        model = frozen_stack(3)
        original = list(model)
        trials = []

        def evaluate(candidate):
            packed = tuple(i for i, layer in enumerate(candidate) if isinstance(layer, nn.Identity))
            trials.append(packed)
            return 10 - sum((1, 1, 0)[i] for i in packed)

        with patch.object(PackedNVFP4Linear, "from_linear", side_effect=lambda *a, **kw: nn.Identity()):
            report = calibrate_packed_layers(
                model, groups={"first": ["0"], "second": ["1"], "last": ["2"]},
                evaluate=evaluate, max_drop=1,
            )

        self.assertEqual(trials, [(), (0,), (0, 1), (0, 2)])
        self.assertEqual((report["baseline"], report["final_score"]), (10, 9))
        self.assertEqual([(g["name"], g["score"], g["accepted"]) for g in report["groups"]],
                         [("first", 9, True), ("second", 8, False), ("last", 9, True)])
        self.assertIs(model[1], original[1])
        self.assertIsInstance(model[0], nn.Identity)
        self.assertIsInstance(model[2], nn.Identity)

    def test_exception_restores_all_groups(self):
        for failure in ("pack", "evaluate", "nonfinite"):
            with self.subTest(failure=failure):
                model = frozen_stack(3)
                model[1].eval()
                original = list(model)

                def pack(layer, **kwargs):
                    if failure == "pack" and layer is original[2]:
                        raise RuntimeError("synthetic failure")
                    return nn.Identity()

                def evaluate(candidate):
                    if failure == "evaluate" and isinstance(candidate[2], nn.Identity):
                        raise RuntimeError("synthetic failure")
                    if failure == "nonfinite" and isinstance(candidate[2], nn.Identity):
                        return float("nan")
                    return 1

                with patch.object(PackedNVFP4Linear, "from_linear", side_effect=pack):
                    error = ValueError if failure == "nonfinite" else RuntimeError
                    with self.assertRaises(error):
                        calibrate_packed_layers(model, groups={"first": ["0"], "rest": ["1", "2"]},
                                                evaluate=evaluate, max_drop=0)
                for layer, before in zip(model, original):
                    self.assertIs(layer, before)
                self.assertEqual([layer.training for layer in model], [True, False, True])

    def test_keep_protects_subtree_and_skips_empty_trial(self):
        model = nn.Sequential(frozen_stack(2), frozen_stack(2))
        untouched = tuple(model[0])
        scores = []

        def evaluate(candidate):
            scores.append(sum(isinstance(layer, nn.Identity) for layer in candidate.modules()))
            return 1

        with patch.object(PackedNVFP4Linear, "from_linear", side_effect=lambda *a, **kw: nn.Identity()):
            report = calibrate_packed_layers(model, groups={"protected": ["0.*"], "rest": ["1"]},
                                             keep=["0", "1.1"], evaluate=evaluate, max_drop=0)
        self.assertEqual(scores, [0, 1])
        self.assertEqual([g["modules"] for g in report["groups"]], [[], ["1.0"]])
        self.assertIsNone(report["groups"][0]["score"])
        for layer, before in zip(model[0], untouched):
            self.assertIs(layer, before)
        self.assertIsInstance(model[1][0], nn.Identity)
        self.assertIs(type(model[1][1]), nn.Linear)

    def test_shared_module_and_tied_weight(self):
        for alias in ("module", "weight", "view"):
            with self.subTest(alias=alias):
                model = frozen_stack(2)
                if alias == "module":
                    model[1] = model[0]
                elif alias == "view":
                    model[1].weight = nn.Parameter(model[0].weight.T, requires_grad=False)
                else:
                    model[1].weight = model[0].weight
                original = list(model)
                with patch.object(PackedNVFP4Linear, "from_linear") as pack:
                    with self.assertRaises(ValueError):
                        calibrate_packed_layers(model, groups={"one": ["0"]},
                                                evaluate=lambda _: 1, max_drop=0)
                    pack.assert_not_called()
                self.assertIs(model[0], original[0])
                self.assertIs(model[1], original[1])
                self.assertEqual(model[0].weight.data_ptr(), model[1].weight.data_ptr())

    def test_disjoint_views_can_use_different_precisions(self):
        model = frozen_stack(2)
        weights = torch.zeros(8, 4, dtype=torch.bfloat16)
        for layer, weight in zip(model, weights.chunk(2)):
            layer.weight = nn.Parameter(weight, requires_grad=False)
        original = model[1]
        with patch.object(PackedNVFP4Linear, "from_linear", return_value=nn.Identity()):
            calibrate_packed_layers(model, groups={"first": ["0"]}, evaluate=lambda _: 1, max_drop=0)
        self.assertIsInstance(model[0], nn.Identity)
        self.assertIs(model[1], original)

    def test_shared_unselected_child_recovers_its_training_mode(self):
        shared = nn.Dropout()
        model = nn.Sequential(frozen_stack(1), nn.Sequential(shared), nn.Sequential(shared))
        model[1].eval()
        model[2].eval()
        shared.train()
        with patch.object(PackedNVFP4Linear, "from_linear", return_value=nn.Identity()):
            calibrate_packed_layers(model, groups={"linear": ["0"]}, evaluate=lambda _: 1, max_drop=0)
        self.assertFalse(model[1].training or model[2].training)
        self.assertTrue(shared.training)


class NativeCalibrationTests(unittest.TestCase):
    def test_mixed_forward_and_backward_after_inference_calibration(self):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
            self.skipTest("native packed NVFP4 requires an SM120 GPU")
        torch.manual_seed(31)
        options = dict(device="cuda", dtype=torch.bfloat16)
        model = nn.Sequential(nn.Linear(13, 17, **options), nn.Linear(17, 37, **options),
                              nn.Linear(37, 11, **options))
        model[1:].requires_grad_(False)
        x = torch.randn(2, 5, 13, **options, requires_grad=True)
        upstream = torch.randn(2, 5, 11, **options)
        dense_gradient, = torch.autograd.grad(model(x), x, upstream)
        with torch.inference_mode():
            expected = model(x)
            bias = model[1].bias.clone()
            protected = model[2]

            def evaluate(candidate):
                return -float((candidate(x).float() - expected.float()).square().mean())

            default_dtype = torch.get_default_dtype()
            try:
                torch.set_default_dtype(torch.float64)
                report = calibrate_packed_layers(model, groups={"frozen": ["1", "2"]}, keep=["2"],
                                                 evaluate=evaluate, max_drop=1)
            finally:
                torch.set_default_dtype(default_dtype)
        self.assertTrue(report["groups"][0]["accepted"])
        self.assertIsInstance(model[1], PackedNVFP4Linear)
        self.assertEqual(model[1].tensor_scale.dtype, torch.float32)
        self.assertEqual(model[1].backward_scale.dtype, torch.float32)
        self.assertIs(model[2], protected)
        zeros = torch.zeros(2, 17, **options)
        torch.testing.assert_close(model[1](zeros), bias.expand(2, -1), rtol=0, atol=0)
        output = model(x)
        self.assertLess((output.float() - expected.float()).norm() / expected.float().norm(), 0.3)
        output.backward(upstream)
        for grad in (x.grad, model[0].weight.grad):
            self.assertIsNotNone(grad)
            self.assertTrue(grad.isfinite().all() and grad.abs().max() > 0)
        self.assertIsNone(protected.weight.grad)
        cosine = torch.nn.functional.cosine_similarity(x.grad.flatten().float(),
                                                        dense_gradient.flatten().float(), dim=0)
        self.assertGreater(cosine, 0.9)


if __name__ == "__main__":
    unittest.main()
