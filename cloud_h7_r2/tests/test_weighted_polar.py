import unittest

import torch

from weighted_polar import (
    DiagonalFactorCollector,
    HiddenUpdateCommitter,
    calibrate_scale_strict,
    match_global_rms,
    polar_direction,
    weighted_polar_direction,
)
from muon_compat import Muon


class WeightedPolarTest(unittest.TestCase):
    def test_muon_backport_matches_torch_2_10_golden_tall_and_wide(self):
        cases = [
            (
                torch.tensor([[1.0, -2.0], [3.0, 0.5], [-1.5, 2.5]]),
                "match_rms_adamw",
                torch.tensor([
                    [-0.10148735344409943, 0.19079621136188507],
                    [-0.2801050841808319, -0.036197155714035034],
                    [0.1515544354915619, -0.24086330831050873],
                ]),
            ),
            (
                torch.tensor([[1.0, -2.0, 0.25], [3.0, 0.5, -1.0]]),
                None,
                torch.tensor([
                    [-0.333984375, 0.65234375, -0.07958984375],
                    [-0.98828125, -0.15625, 0.328125],
                ]),
            ),
        ]
        for gradient, adjust_lr_fn, expected in cases:
            parameter = torch.nn.Parameter(torch.zeros_like(gradient))
            optimizer = Muon(
                [parameter],
                lr=1.0,
                momentum=0.0,
                weight_decay=0.0,
                adjust_lr_fn=adjust_lr_fn,
            )
            parameter.grad = gradient
            optimizer.step()
            self.assertTrue(torch.equal(parameter.detach(), expected))

    def test_muon_backport_matches_torch_2_10_two_step_momentum(self):
        cases = [
            (
                torch.tensor([[1.0, -2.0], [3.0, 0.5], [-1.5, 2.5]]),
                torch.tensor([[0.5, 1.0], [-2.0, 3.0], [1.25, -0.75]]),
                torch.tensor([
                    [-0.4242171347141266, 0.13903766870498657],
                    [-0.23105287551879883, -0.284333735704422],
                    [-0.04803735017776489, -0.29769623279571533],
                ]),
                torch.tensor([
                    [0.07249999791383743, -0.044999998062849045],
                    [0.042500000447034836, 0.17374999821186066],
                    [-0.008750000037252903, 0.08124999701976776],
                ]),
            ),
            (
                torch.tensor([[1.0, -2.0, 0.25], [3.0, 0.5, -1.0]]),
                torch.tensor([[0.5, 1.0, -2.0], [-2.0, 3.0, 0.75]]),
                torch.tensor([
                    [-0.22530192136764526, 0.21409602463245392, 0.18419954180717468],
                    [-0.28078165650367737, -0.4208342134952545, 0.08524937927722931],
                ]),
                torch.tensor([
                    [0.07249999791383743, -0.044999998062849045, -0.08812500536441803],
                    [0.042500000447034836, 0.17374999821186066, -0.009999999776482582],
                ]),
            ),
        ]
        for first, second, expected_parameter, expected_buffer in cases:
            parameter = torch.nn.Parameter(torch.zeros_like(first))
            optimizer = Muon(
                [parameter],
                lr=1.0,
                momentum=0.95,
                weight_decay=0.0,
                adjust_lr_fn="match_rms_adamw",
            )
            for gradient in (first, second):
                parameter.grad = gradient
                optimizer.step()
            self.assertTrue(
                torch.allclose(
                    parameter.detach(), expected_parameter, rtol=2e-2, atol=1.5e-2
                )
            )
            actual_flat = parameter.detach().float().flatten()
            expected_flat = expected_parameter.float().flatten()
            cosine = torch.dot(actual_flat, expected_flat) / (
                actual_flat.norm() * expected_flat.norm()
            )
            norm_ratio = actual_flat.norm() / expected_flat.norm()
            self.assertGreaterEqual(float(cosine), 0.999)
            self.assertLessEqual(abs(float(norm_ratio) - 1.0), 0.02)
            self.assertTrue(
                torch.allclose(
                    optimizer.state[parameter]["momentum_buffer"],
                    expected_buffer,
                    rtol=1e-6,
                    atol=1e-6,
                )
            )

    def test_identity_geometry_recovers_polar_direction(self):
        gradient = torch.tensor([[3.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        factors = {"weight": {"left": torch.ones(2), "right": torch.ones(3)}}

        actual = weighted_polar_direction(
            {"weight": gradient}, factors, damping=0.0
        )["weight"]
        expected = polar_direction({"weight": gradient})["weight"]

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_polar_direction_is_transpose_symmetric_without_shape_scaling(self):
        tall = torch.tensor([[3.0, 0.0], [0.0, 1.0], [1.0, 2.0]])
        directions = polar_direction({"tall": tall, "wide": tall.T.contiguous()})

        self.assertTrue(torch.allclose(directions["tall"], directions["wide"].T, atol=1e-6))
        self.assertAlmostEqual(
            float(directions["tall"].square().sum()),
            float(directions["wide"].square().sum()),
            places=6,
        )

    def test_factors_are_available_only_after_backward_and_stay_on_device(self):
        model = torch.nn.Linear(2, 2, bias=False)
        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        with DiagonalFactorCollector(model, {"weight"}) as collector:
            loss = model(inputs).square().sum()
            with self.assertRaisesRegex(RuntimeError, "backward"):
                collector.factors()
            loss.backward()
        factors = collector.factors()

        self.assertEqual(factors["weight"]["left"].device, inputs.device)
        self.assertEqual(factors["weight"]["right"].device, inputs.device)

    def test_calibration_failure_does_not_mutate_parameters(self):
        model = torch.nn.Linear(2, 1, bias=False)
        before = model.weight.detach().clone()

        with self.assertRaisesRegex(RuntimeError, "not bracketed"):
            calibrate_scale_strict(
                lambda scale: min(scale, 1.0),
                budget=2.0,
                initial_max_scale=4.0,
                max_scale=16.0,
                tolerance=0.02,
            )

        self.assertTrue(torch.equal(model.weight, before))

    def test_calibration_expands_and_matches_budget(self):
        scale, measured, diagnostics = calibrate_scale_strict(
            lambda value: value,
            budget=6.0,
            initial_max_scale=4.0,
            max_scale=16.0,
            tolerance=0.02,
        )

        self.assertAlmostEqual(scale, 6.0, places=3)
        self.assertAlmostEqual(measured / 6.0, 1.0, places=3)
        self.assertEqual(diagnostics["max_scale"], 16.0)

    def test_global_rms_counts_unmodified_auxiliary_parameters(self):
        matched, _ = match_global_rms(
            {"hidden.weight": torch.tensor([[3.0, 4.0]])},
            target_rms=2e-3,
            total_parameter_count=4,
        )

        realized = float(matched["hidden.weight"].square().sum().div(4).sqrt())
        self.assertAlmostEqual(realized, 2e-3, places=9)

    def test_hidden_update_is_committed_exactly_once(self):
        model = torch.nn.Linear(2, 1, bias=False)
        before = model.weight.detach().clone()
        update = match_global_rms(
            {"weight": torch.ones_like(model.weight)},
            target_rms=1e-3,
            total_parameter_count=model.weight.numel(),
        )[0]
        committer = HiddenUpdateCommitter(model)

        realized_square_sum = committer.commit(update, scale=1.0)
        self.assertTrue(torch.allclose(model.weight, before + update["weight"]))
        self.assertAlmostEqual(
            realized_square_sum,
            float(update["weight"].float().square().sum()),
            places=8,
        )
        with self.assertRaisesRegex(RuntimeError, "already committed"):
            committer.commit(update, scale=1.0)


if __name__ == "__main__":
    unittest.main()
