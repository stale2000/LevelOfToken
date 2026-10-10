"""CPU contracts for the bounded pilot sampler; no quality or GPU claims."""

import unittest
from types import SimpleNamespace

import torch
from budget_layout import EntropyCalibration, choose_budget_layout
from h3 import make_h3_adapter, patchify
from render_h3_pilot import (
    CONTROLS,
    LATENT_T,
    audio_sigma,
    comfy_sigmas,
    configure_control,
    initial_noise,
    make_tiling,
    sample_euler,
)


class OracleHead:
    """Perfect coordinate-aware flow head, with a known audio clean target."""

    def __init__(self, target):
        self.target = target
        self._lot = None
        self.inputs = []

    def __call__(self, state, timestep, _text, *, audio_rows, return_audio):
        self.inputs.append((state.clone(), audio_rows.clone(), self._lot))
        assert return_audio
        sigma = 1 - float(timestep)
        clean = self._lot.to_y(self.target) if self._lot else self.target
        return (clean - state) / sigma, (2 - audio_rows) / audio_sigma(sigma)


class PilotContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = make_h3_adapter(hidden_size=8, shape_hidden=4)
        for extent in cls.adapter.bank.extents:
            if extent != (1, 1, 1):
                cls.adapter.bank.scale(extent).fill_(2)
        cls.calibration = EntropyCalibration(torch.zeros(96), torch.ones(96))

    def test_all_controls_share_both_initial_noise_streams(self):
        video, audio = initial_noise(123, (74, 32))
        original_video, original_audio = video.clone(), audio.clone()
        network = SimpleNamespace(unet_loras=[SimpleNamespace(multiplier=1)])
        for variant in CONTROLS:
            model = OracleHead(torch.ones_like(video))
            controller = configure_control(model, network, self.adapter, variant,
                                           calibration=self.calibration, warmup_steps=2,
                                           oracle_latent=video)
            sample_euler(model, None, video, audio, comfy_sigmas(4), controller=controller, dtype=torch.float32)
            first_video, first_audio, _splice = model.inputs[0]
            self.assertTrue(torch.equal(first_video, original_video))
            self.assertTrue(torch.equal(first_audio, original_audio))
            self.assertEqual(network.unet_loras[0].multiplier, 0 if variant == "base_dense" else 1)
        self.assertTrue(torch.equal(video, original_video))
        self.assertTrue(torch.equal(audio, original_audio))

    def test_exact_video_and_carried_audio_targets(self):
        video, audio = initial_noise(21, (74, 32))
        network = SimpleNamespace(unet_loras=[])
        target = video.sin()
        for variant in CONTROLS:
            model = OracleHead(target)
            controller = configure_control(model, network, self.adapter, variant,
                                           calibration=self.calibration, warmup_steps=2,
                                           oracle_latent=target)
            result, audio_result, _trace = sample_euler(
                model, None, video, audio, comfy_sigmas(6), controller=controller, dtype=torch.float32)
            self.assertTrue(torch.allclose(result, target, atol=2e-6), variant)
            self.assertTrue(torch.allclose(audio_result, torch.full_like(audio_result, 2), atol=2e-6), variant)

    def test_inference_layout_starts_dense_and_switches_once(self):
        video, audio = initial_noise(71, (74, 32))
        model = OracleHead(video.sin())
        network = SimpleNamespace(unet_loras=[])
        controller = configure_control(model, network, self.adapter, "entropy_inference",
                                       calibration=self.calibration, warmup_steps=2)
        _result, _audio, trace = sample_euler(model, None, video, audio, comfy_sigmas(6),
                                            controller=controller, dtype=torch.float32)
        self.assertEqual([step["step"] for step in trace if step["layout_selected"]], [2])
        self.assertIsNone(model.inputs[0][2])
        self.assertIsNone(model.inputs[1][2])
        selected = model.inputs[2][2]
        self.assertIsNotNone(selected)
        self.assertLess(selected.layout.count, selected.layout.dense_count)
        self.assertTrue(all(item[2] is selected for item in model.inputs[2:]))

    def test_fixed_half_budget_switches_at_declared_sigma_boundary(self):
        video, audio = initial_noise(171, (74, 32))
        schedule = comfy_sigmas(8)
        expected_step = next(index for index, next_sigma in enumerate(schedule[1:], 1)
                             if next_sigma <= 0.90)
        model = OracleHead(video.cos())
        controller = configure_control(
            model, SimpleNamespace(unet_loras=[]), self.adapter, "uniform50_1x2",
            warmup_steps=3, warmup_sigma=0.90,
        )
        _result, _audio, trace = sample_euler(
            model, None, video, audio, schedule, controller=controller, dtype=torch.float32)
        self.assertEqual([row["step"] for row in trace if row["layout_selected"]], [expected_step])
        self.assertEqual(controller.last_selection.layout.count / controller.last_selection.layout.dense_count, 0.5)
        self.assertEqual(controller.last_selection.layout, make_tiling(1, 2))
        self.assertIsNone(model.inputs[expected_step - 1][2])
        self.assertIsNotNone(model.inputs[expected_step][2])

    def test_warmup_selector_shuffles_preserve_selected_extent_budgets(self):
        video, audio = initial_noise(91, (74, 32))
        network = SimpleNamespace(unet_loras=[])
        def extents(layout):
            counts = {}
            for rect in layout.rects:
                counts[rect.extent] = counts.get(rect.extent, 0) + rect.et * rect.eh * rect.ew
            return counts
        for policy in ("detail", "entropy"):
            outputs = {}
            variants = (f"{policy}_inference", f"{policy}_inference_shuffle")
            for variant in variants:
                model = OracleHead(video.sin())
                controller = configure_control(
                    model, network, self.adapter, variant,
                    calibration=self.calibration if policy == "entropy" else None,
                    warmup_steps=2, budget_fraction=0.25,
                )
                sample_euler(model, None, video, audio, comfy_sigmas(4),
                             controller=controller, dtype=torch.float32)
                outputs[variant] = controller.last_selection.layout
            self.assertEqual(outputs[variants[0]].count, outputs[variants[1]].count)
            self.assertEqual(extents(outputs[variants[0]]), extents(outputs[variants[1]]))
            self.assertEqual(outputs[variants[0]].count / outputs[variants[0]].dense_count, 0.25)

    def test_oracle_detail_selects_from_reference_and_trace_records_the_map(self):
        video, audio = initial_noise(119, (74, 32))
        reference = torch.zeros_like(video)
        reference[:, :, :, :, 20:] = 1
        network = SimpleNamespace(unet_loras=[])
        model = OracleHead(video.sin())
        controller = configure_control(model, network, self.adapter, "oracle_detail",
                                       warmup_steps=2, budget_fraction=0.25,
                                       seed=19, oracle_latent=reference)
        _result, _audio, trace = sample_euler(
            model, None, video, audio, comfy_sigmas(4), controller=controller, dtype=torch.float32)
        expected = choose_budget_layout(
            patchify(reference)[0], 60, policy="detail", seed=21
        )
        self.assertEqual(controller.last_selection.layout, expected.layout)
        selected = next(row for row in trace if row["layout_selected"])
        self.assertEqual(selected["selection_policy"], "detail")
        self.assertEqual(selected["selected_tokens"], 60 * LATENT_T)
        self.assertTrue(selected["selected_rectangles"])
        self.assertEqual(sum(selected["selected_extent_rectangle_histogram"].values()),
                         selected["selected_tokens"])
        self.assertEqual(sum(selected["selected_extent_token_histogram"].values()),
                         selected["selected_dense_tokens"])

    def test_oracle_detail_requires_reference_and_marks_layout_as_oracle(self):
        video, _audio = initial_noise(19, (74, 32))
        with self.assertRaisesRegex(ValueError, "ground-truth"):
            configure_control(SimpleNamespace(), SimpleNamespace(unet_loras=[]),
                              self.adapter, "oracle_detail")
        network = SimpleNamespace(unet_loras=[])
        model = OracleHead(video.sin())
        controller = configure_control(model, network, self.adapter, "oracle_detail",
                                       warmup_steps=2, oracle_latent=video)
        self.assertTrue(torch.equal(controller.selection_latent, video))

    def test_schedule_and_control_validation(self):
        self.assertEqual(comfy_sigmas(12)[0], 1)
        self.assertEqual(comfy_sigmas(12)[-1], 0)
        self.assertAlmostEqual(audio_sigma(1), 1)
        self.assertAlmostEqual(audio_sigma(0), 0)
        with self.assertRaises(ValueError):
            comfy_sigmas(0)
        with self.assertRaises(ValueError):
            configure_control(SimpleNamespace(), SimpleNamespace(unet_loras=[]), self.adapter, "oracle")


if __name__ == "__main__":
    unittest.main()
