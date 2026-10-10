import argparse
import copy
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from budget_layout import EntropyCalibration
from train_h3 import main, sample_layout, training_objective
from training_experiment import (
    balanced_records,
    dataset_identity,
    dense_teacher_prediction,
    evaluate_confirmation_checkpoint,
    evaluation_contract,
    evaluation_partition,
    evaluation_report_paths,
    frozen_dense_teacher,
    json_safe,
    prediction_diagnostics,
    require_finite_loss,
    split_records,
    teacher_distance,
)


class MockDenseH3(torch.nn.Module):
    """A frozen base and trainable LoRA with observable forward controls."""

    def __init__(self):
        super().__init__()
        self.base = torch.nn.Parameter(torch.tensor(2.0), requires_grad=False)
        self.lora = torch.nn.Linear(1, 1, bias=False)
        self.lora.weight.data.fill_(3.0)
        self.lora.multiplier = .25
        self.mode_child = torch.nn.Dropout()
        self.mode_child.eval()
        self._lot = None
        self.calls = []
        self.fail_teacher = False

    def forward(self, x_t, timestep, text, *, audio_noise):
        out = torch.ones_like(x_t) * (self.base + self.lora.multiplier * self.lora.weight.sum())
        self.calls.append({
            "x_t": x_t, "timestep": timestep, "text": text, "audio_noise": audio_noise,
            "lot": self._lot, "training": self.training,
            "multiplier": self.lora.multiplier, "grad_enabled": torch.is_grad_enabled(),
            "out": out,
        })
        if self.fail_teacher and self.lora.multiplier == 0:
            raise RuntimeError("teacher failed")
        return out


class DensePreservationContracts(unittest.TestCase):
    def setUp(self):
        self.dit = MockDenseH3()
        self.network = SimpleNamespace(unet_loras=[self.dit.lora])
        self.x0 = torch.ones(1, 1, 1, 1, 1)
        self.noise = torch.full_like(self.x0, 5.0)
        self.sigma = torch.tensor([.5])
        self.text = torch.tensor([[[2.0]]])
        self.audio = torch.tensor([[4.0]])

    def objective(self, *, kind="dense", **weights):
        return training_objective(
            self.dit, None, self.network, self.x0, self.text, None, kind,
            self.sigma, self.noise, self.audio, **weights,
        )

    def assert_restored(self):
        self.assertIsNone(self.dit._lot)
        self.assertEqual(self.dit.lora.multiplier, .25)
        self.assertTrue(self.dit.training)
        self.assertTrue(self.dit.lora.training)
        self.assertFalse(self.dit.mode_child.training)

    def test_default_dense_preservation_keeps_original_data_loss_and_gradient(self):
        loss, _out, _state, _clean, gap, data_loss = self.objective(
            data_weight=.5, distill_weight=100.0,
        )
        self.assertEqual(len(self.dit.calls), 1)
        self.assertIsNone(gap)
        self.assertEqual(float(data_loss.detach()), 45.5625)
        self.assertEqual(float(loss.detach()), 22.78125)
        loss.backward()
        self.assertEqual(float(self.dit.lora.weight.grad), 1.6875)
        self.assert_restored()

    def test_dense_teacher_matches_exact_inputs_and_preserves_frozen_base(self):
        loss, _out, _state, _clean, gap, data_loss = self.objective(
            data_weight=.5, distill_weight=100.0, dense_distill_weight=1.5,
        )
        self.assertEqual(float(data_loss.detach()), 45.5625)
        self.assertEqual(float(gap.detach()), .5625)
        self.assertEqual(float(loss.detach()), 23.625)
        student, teacher = self.dit.calls
        self.assertIs(student["x_t"], teacher["x_t"])
        self.assertIs(student["text"], teacher["text"])
        self.assertIs(student["audio_noise"], teacher["audio_noise"])
        self.assertTrue(torch.equal(student["timestep"], teacher["timestep"]))
        self.assertIsNone(teacher["lot"])
        self.assertFalse(teacher["training"])
        self.assertEqual(teacher["multiplier"], 0.0)
        self.assertFalse(teacher["grad_enabled"])
        self.assertFalse(teacher["out"].requires_grad)
        self.assert_restored()
        loss.backward()
        # A differentiated teacher would cancel the LoRA preservation gradient.
        self.assertEqual(float(self.dit.lora.weight.grad), 2.25)
        self.assertIsNone(self.dit.base.grad)

    def test_coarse_teacher_keeps_existing_weight_and_dense_noisy_target(self):
        splice = SimpleNamespace(to_y=lambda x: x * 2, to_x=lambda y: y / 2)
        with patch("train_h3.LotSplice", return_value=splice):
            loss, _out, _state, _clean, gap, data_loss = self.objective(
                kind="uniform", data_weight=.5, distill_weight=2.0, dense_distill_weight=100.0,
            )
        self.assertEqual(float(data_loss.detach()), 33.0625)
        self.assertEqual(float(gap.detach()), 9.765625)
        self.assertEqual(float(loss.detach()), 36.0625)
        student, teacher = self.dit.calls
        self.assertIs(student["lot"], splice)
        self.assertEqual(float(student["x_t"]), 3.5)
        self.assertEqual(float(teacher["x_t"]), 3.0)
        self.assert_restored()
        loss.backward()
        self.assertEqual(float(self.dit.lora.weight.grad), -.125)

    def test_dense_weight_does_not_enable_teacher_on_coarse_steps(self):
        splice = SimpleNamespace(to_y=lambda x: x, to_x=lambda y: y)
        with patch("train_h3.LotSplice", return_value=splice):
            loss, _out, _state, _clean, gap, data_loss = self.objective(
                kind="uniform", data_weight=1.0, distill_weight=0.0, dense_distill_weight=1.0,
            )
        self.assertEqual(len(self.dit.calls), 1)
        self.assertIsNone(gap)
        self.assertTrue(torch.equal(loss, data_loss))

    def test_teacher_failure_restores_exact_router_lora_and_modes(self):
        original_router = object()
        self.dit._lot = original_router
        self.dit.fail_teacher = True
        with self.assertRaisesRegex(RuntimeError, "teacher failed"):
            dense_teacher_prediction(
                self.dit, self.network, self.x0, self.noise, self.sigma,
                self.text, self.audio, x_t=self.x0,
            )
        self.assertIs(self.dit._lot, original_router)
        self.dit._lot = None
        self.assert_restored()
        self.assertIsNone(self.dit.lora.weight.grad)
        self.assertIsNone(self.dit.base.grad)

    def test_dense_weight_cli_default_and_run_config_serialization(self):
        parsed = []
        parse_args = argparse.ArgumentParser.parse_args

        def capture(parser):
            parsed.append(parse_args(parser))
            raise RuntimeError("parsed before GPU imports")

        for flags, weight in (([], 0.0), (["--dense-distill", "1.0"], 1.0)):
            with (
                self.subTest(weight=weight),
                patch.object(argparse.ArgumentParser, "parse_args", capture),
                patch.object(sys, "argv", ["train_h3.py", "--pairs", "unused", *flags]),
                self.assertRaisesRegex(RuntimeError, "parsed before GPU imports"),
            ):
                main()
            config = json.loads(json.dumps(json_safe(vars(parsed[-1]))))
            self.assertEqual(config["dense_distill"], weight)

    def test_dense_weight_cli_rejects_negative_and_nonfinite_before_gpu_imports(self):
        for weight in ("-1", "nan", "inf"):
            with (
                self.subTest(weight=weight),
                patch.object(sys, "argv", ["train_h3.py", "--pairs", "unused", "--dense-distill", weight]),
                self.assertRaisesRegex(SystemExit, "objective weights must be finite and non-negative"),
            ):
                main()

    def test_resume_dense_weight_equality_and_legacy_zero_default(self):
        modules = {
            "gpu_guard": SimpleNamespace(refuse_if_busy=lambda *_: None),
            "fizgig.minimax.loader": SimpleNamespace(load_minimax_h3_dit=None),
            "fizgig.minimax.trainer": SimpleNamespace(sample_sigmas=None),
            "fizgig.networks.lora": SimpleNamespace(create_network=None),
            "procrustes_h3": SimpleNamespace(CHECKPOINT=None, fit_bank=None, load_pairs=None, pretrained_maps=None),
        }
        settings = {
            "rank": 16, "layout_policy": "legacy", "seed": 0,
            "budget_min": .25, "budget_max": .75, "dense_p": .2, "random_layout_p": 0.0,
            "holdout": 24, "shift": 12.0, "dense_shift": 12.0, "distill": 0.0,
            "data_weight": 1.0, "lr": 1e-4, "adapter_lr": 1e-4,
            "swap": 4, "gradient_checkpointing": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoints" / "step-00000500"
            checkpoint.mkdir(parents=True)
            for stored, requested in ((None, 0.0), (None, 1.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)):
                metadata = dict(settings)
                if stored is not None:
                    metadata["dense_distill"] = stored
                (checkpoint / "manifest.json").write_text(json.dumps({"metadata": metadata}))
                argv = ["train_h3.py", "--pairs", "unused", "--resume", str(checkpoint),
                        "--out", str(root), "--check", "--dense-distill", str(requested)]
                with (
                    self.subTest(stored=stored, requested=requested),
                    patch.dict(sys.modules, modules),
                    patch.object(sys, "argv", argv),
                    patch("train_h3.load_trainer_state", return_value=(checkpoint, {})),
                    patch("train_h3.read_manifest", side_effect=RuntimeError("resume validated")),
                ):
                    if (0.0 if stored is None else stored) == requested:
                        with self.assertRaisesRegex(RuntimeError, "resume validated"):
                            main()
                    else:
                        with self.assertRaisesRegex(SystemExit, "settings differ: dense_distill"):
                            main()


class ConfirmationContracts(unittest.TestCase):
    def test_confirmation_sources_are_disjoint_from_validation_sources(self):
        held = [{"clip": f"clip-{i}", "source_id": f"source-{i // 2}",
                 "camera_motion_group": "static" if i < 4 else "moving"} for i in range(8)]
        validation, confirmation = evaluation_partition(held, 2)
        self.assertEqual(len(validation), 2)
        validation_sources = {row["source_id"] for row in validation}
        self.assertFalse(validation_sources & {row["source_id"] for row in confirmation})
        self.assertEqual(len(confirmation), 4)

    def test_identity_hashes_both_fixed_sets_independent_of_selected_split(self):
        with tempfile.TemporaryDirectory() as directory:
            tensor = Path(directory) / "tensor"
            tensor.write_bytes(b"cached tensor")
            records = [{"clip": str(i), "source_id": str(i), "latent_path": str(tensor),
                        "text_path": str(tensor)} for i in range(3)]
            identities = []
            for split in ("validation", "confirmation"):
                run_config = {"evaluation_split": split, "evaluation_items": records[:1] if split == "validation" else records[1:]}
                contract = evaluation_contract(records[:1], records[1:], seed=0, holdout=3, eval_budget=.5)
                run_config["dataset_identity"] = dataset_identity(records, contract)
                identities.append(run_config["dataset_identity"])
            self.assertEqual(*identities)
            changed = evaluation_contract(records[:1], records[1:2], seed=0, holdout=3, eval_budget=.5)
            self.assertNotEqual(identities[0], dataset_identity(records, changed))

    def test_confirmation_reports_have_separate_step_paths(self):
        validation = evaluation_report_paths(Path("run"), 500, "validation")
        confirmation = evaluation_report_paths(Path("run"), 500, "confirmation")
        self.assertEqual(validation, (Path("run/metrics-step-00000500.csv"), Path("run/metrics-step-00000500.json")))
        self.assertEqual(confirmation, (Path("run/metrics-confirmation-step-00000500.csv"),
                                        Path("run/metrics-confirmation-step-00000500.json")))
        with self.assertRaisesRegex(ValueError, "unknown evaluation split"):
            evaluation_report_paths(Path("run"), 500, "training")

    def test_confirmation_evaluates_saved_step_without_mutating_trainer_state(self):
        state = {"step": 4000, "lora_multiplier": .75, "token_counters": {"student_tokens": 2500}}
        before = copy.deepcopy(state)
        set_multiplier = Mock()
        evaluate = Mock(return_value={"eval_split": "confirmation"})
        result = evaluate_confirmation_checkpoint(state, set_multiplier, evaluate)
        self.assertEqual(result, {"eval_split": "confirmation"})
        set_multiplier.assert_called_once_with(.75)
        evaluate.assert_called_once_with(4000)
        self.assertEqual(state, before)
        with self.assertRaisesRegex(ValueError, "requires a resumable checkpoint"):
            evaluate_confirmation_checkpoint(None, set_multiplier, evaluate)

    def test_confirmation_cli_requires_checkpoint_and_excludes_training(self):
        cases = (["--baseline-only-confirmation"], ["--evaluation-split", "confirmation"],
                 ["--baseline-only-confirmation", "--resume", "unused", "--check"])
        for flags in cases:
            with (
                self.subTest(flags=flags),
                patch.object(sys, "argv", ["train_h3.py", "--pairs", "unused", *flags]),
                self.assertRaisesRegex(SystemExit, "confirmation"),
            ):
                main()

    def test_cpu_confirmation_run_preserves_validation_and_checkpoint_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "branch-b"
            checkpoint = root / "checkpoints" / "step-00004000"
            checkpoint.mkdir(parents=True)
            cache = Path(directory) / "cache"
            cache.mkdir()
            records = []
            for i in range(3):
                clip = f"clip{i}"
                latent_path = cache / f"{clip}_0384x0640_minimaxh3.safetensors"
                text_path = cache / f"{clip}_minimaxh3_te.safetensors"
                latent_path.write_bytes(b"latent")
                text_path.write_bytes(b"text")
                records.append({"clip": clip, "source_id": clip, "camera_motion_group": "unknown", "caption": "",
                                "latent_path": str(latent_path.resolve()), "text_path": str(text_path.resolve())})
            _train, held = split_records(records, 2, 0)
            validation, confirmation = evaluation_partition(held, 1)
            identity = dataset_identity(records, evaluation_contract(validation, confirmation, seed=0, holdout=2, eval_budget=.5))
            settings = {
                "rank": 16, "layout_policy": "legacy", "seed": 0, "budget_min": .25, "budget_max": .75,
                "dense_p": .2, "random_layout_p": 0.0, "holdout": 2, "shift": 12.0, "dense_shift": 12.0,
                "distill": 1.0, "dense_distill": 1.0, "data_weight": 1.0, "lr": 1e-4,
                "adapter_lr": 1e-4, "swap": 4, "gradient_checkpointing": False,
            }
            (checkpoint / "manifest.json").write_text(json.dumps({"metadata": settings}))
            torch.save({}, checkpoint / "adapter.pt")
            protected = ("run.json", "latest.json", "baseline_metrics.json", "metrics-step-00004000.csv",
                         "metrics-step-00004000.json", "progress.json")
            for name in protected:
                (root / name).write_text("validation artifact")
            resume_state = {
                "step": 4000, "lora_multiplier": 1.0, "python_rng": __import__("random").Random(0).getstate(),
                "torch_rng": torch.get_rng_state(), "token_counters": {"student_tokens": 1234},
                "entropy_calibration": {"mean": torch.zeros(96), "std": torch.ones(96)},
                "run_config": {"dataset_identity": identity, "evaluation_contract_version": 2,
                               "validation_items": validation, "confirmation_items": confirmation},
            }
            dit = MockDenseH3()
            dit.blocks = torch.nn.ModuleList([torch.nn.Identity()])
            dit.enable_block_swap = Mock()
            dit.enable_gradient_checkpointing = Mock()
            network = Mock()
            network.unet_loras = [dit.lora, *[SimpleNamespace(multiplier=1.0) for _ in range(3)]]
            network.load_weights.return_value = SimpleNamespace(missing_keys=[], unexpected_keys=[])
            network.prepare_optimizer_params.return_value = ([{"params": [dit.lora.weight]}], None)
            adapter = Mock()
            adapter.named_parameters.return_value = []
            modules = {
                "gpu_guard": SimpleNamespace(refuse_if_busy=lambda *_: None),
                "fizgig.minimax.loader": SimpleNamespace(load_minimax_h3_dit=lambda *_args, **_kwargs: dit),
                "fizgig.minimax.trainer": SimpleNamespace(sample_sigmas=None),
                "fizgig.networks.lora": SimpleNamespace(create_network=lambda *_args, **_kwargs: network),
                "procrustes_h3": SimpleNamespace(CHECKPOINT=Path("base.pt"), fit_bank=lambda *_: None,
                                                 load_pairs=lambda *_: None, pretrained_maps=lambda *_: (None, None)),
            }
            argv = ["train_h3.py", "--pairs", "unused", "--cache", str(cache), "--holdout", "2", "--eval-items", "1",
                    "--resume", str(checkpoint), "--out", str(root), "--dense-distill", "1", "--distill", "1",
                    "--baseline-only-confirmation"]
            real_device = torch.device
            latent = torch.arange(24 * 8 * 8, dtype=torch.float32).reshape(1, 24, 1, 8, 8) / 1000
            splice = SimpleNamespace(to_y=lambda x: x, to_x=lambda x: x)
            with (
                patch.dict(sys.modules, modules), patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()),
                patch("train_h3.load_trainer_state", return_value=(checkpoint, resume_state)),
                patch("train_h3.make_h3_adapter", return_value=adapter),
                patch("train_h3.load_item", return_value=(latent, torch.zeros(1, 1, 1))),
                patch("train_h3.audio_noise_for", return_value=torch.zeros(0, 1)),
                patch("train_h3.LotSplice", return_value=splice),
                patch("train_h3.torch.device", side_effect=lambda *_: real_device("cpu")),
                patch("train_h3.torch.cuda.synchronize"), patch("train_h3.torch.cuda.memory_allocated", return_value=0),
                patch("train_h3.torch.cuda.is_available", return_value=False),
                patch("train_h3.torch.optim.AdamW", side_effect=AssertionError("optimizer allocated")),
                patch("train_h3.save_checkpoint_bundle", side_effect=AssertionError("checkpoint saved")),
                patch("train_h3.training_objective", side_effect=AssertionError("optimizer objective reached")),
            ):
                main()
                # Old branch checkpoints used the validation-only hash, but also
                # stored both exact sets. They must remain eligible for final confirmation.
                legacy_settings = {"schema_version": "pilot-v2", "seed": 0, "holdout": 2,
                                   "evaluation_items": validation, "eval_budget": .5, "eval_sigmas": [.3, .6, .9]}
                resume_state["run_config"].pop("evaluation_contract_version")
                resume_state["run_config"]["dataset_identity"] = dataset_identity(records, legacy_settings)
                (root / "metrics.jsonl").unlink()
                (root / "log.jsonl").unlink()
                main()
                # Migration must reject a source-set change even if the legacy
                # validation-only hash still matches.
                resume_state["run_config"]["confirmation_items"] = validation
                with self.assertRaisesRegex(SystemExit, "data/split/evaluation contract differs"):
                    main()
            for name in protected:
                self.assertEqual((root / name).read_text(), "validation artifact")
            network.save_weights.assert_not_called()
            csv_path, summary_path = evaluation_report_paths(root, 4000, "confirmation")
            self.assertTrue(csv_path.is_file())
            summary = json.loads(summary_path.read_text())
            self.assertEqual(summary["eval_split"], "confirmation")
            self.assertEqual(summary["step"], 4000)
            self.assertEqual(summary["clips"], 1)
            for name in ("uniform1x2", "uniform2x1"):
                self.assertEqual(summary["metrics"][name]["tokens_per_frame_mean"], 8)
            rows = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 33)
            self.assertEqual({row["eval_split"] for row in rows}, {"confirmation"})
            self.assertEqual({row["source_id"] for row in rows}, {row["source_id"] for row in confirmation})
            self.assertFalse({row["source_id"] for row in rows} & {row["source_id"] for row in validation})
            self.assertEqual(resume_state["token_counters"], {"student_tokens": 1234})
            self.assertEqual(json.loads((root / "log.jsonl").read_text())["eval_split"], "confirmation")


class ExperimentContracts(unittest.TestCase):
    def test_run_config_serializes_repeatable_path_arguments(self):
        paths = {"cache": [Path("cache/a"), Path("cache/b")],
                 "evaluation_items": [{"latent_path": Path("latents/a"), "source_id": "a"}]}
        encoded = json.dumps(json_safe(paths))
        self.assertIn('"cache": ["cache/a", "cache/b"]', encoded)
        self.assertIn('"latent_path": "latents/a"', encoded)

    def test_source_group_split_is_stable_balanced_and_disjoint(self):
        rows = [{'clip': str(i), 'source_id': str(i // 2),
                 'camera_motion_group': 'static' if i // 2 % 2 else 'moving'} for i in range(40)]
        train, held = split_records(rows, 8, 13)
        self.assertEqual((train, held), split_records(list(reversed(rows)), 8, 13))
        self.assertFalse({x['source_id'] for x in train} & {x['source_id'] for x in held})
        self.assertEqual(sum(x['camera_motion_group'] == 'static' for x in held), 8)
        chosen = balanced_records(held, 8)
        self.assertEqual(sum(x['camera_motion_group'] == 'static' for x in chosen), 4)

    def test_layout_mix_can_force_all_four_training_paths(self):
        latent = torch.randn(1, 24, 1, 16, 16)
        calibration = EntropyCalibration(torch.zeros(96), torch.ones(96))
        kinds = {}
        for requested in ("dense", "uniform", "adaptive", "random_budget"):
            layout, kind = sample_layout(latent, __import__('random').Random(11), .2,
                                         policy="entropy", calibration=calibration,
                                         budget_range=(.5, .5), random_layout_p=.1,
                                         force_kind=requested)
            kinds[requested] = kind
            self.assertEqual(layout.count, layout.dense_count if requested == "dense" else layout.count)
            self.assertEqual(kind == "random_budget", requested == "random_budget")
        self.assertEqual(kinds, {"dense": "dense", "uniform": "uniform",
                                 "adaptive": "entropy_budget", "random_budget": "random_budget"})

    def test_random_layout_fraction_matches_stated_mixture(self):
        from train_h3 import layout_category
        counts = {name: 0 for name in ("dense", "uniform", "random", "adaptive")}
        for i in range(1000):
            kind = layout_category((i + .5) / 1000, dense_p=.2, random_p=.1)
            counts["random" if kind == "random_budget" else kind] += 1
        self.assertEqual(counts, {"dense": 200, "uniform": 150, "random": 100, "adaptive": 550})

    def test_identity_changes_when_contents_or_eval_contract_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'tensor'; p.write_bytes(b'one')
            rows = [{'clip': 'a', 'latent_path': str(p), 'text_path': str(p)}]
            before = dataset_identity(rows, {'sigma': .3})
            p.write_bytes(b'two')
            self.assertNotEqual(before, dataset_identity(rows, {'sigma': .3}))
            self.assertNotEqual(dataset_identity(rows, {'sigma': .3}), dataset_identity(rows, {'sigma': .6}))

    def test_teacher_restores_exact_controls_on_exception(self):
        dit = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout())
        dit.train(); dit[1].eval(); dit._lot = object()
        original = dit._lot
        network = SimpleNamespace(unet_loras=[SimpleNamespace(multiplier=.25), SimpleNamespace(multiplier=1.0)])
        with self.assertRaises(RuntimeError), frozen_dense_teacher(dit, network):
            self.assertIsNone(dit._lot)
            self.assertEqual([m.multiplier for m in network.unet_loras], [0, 0])
            self.assertFalse(dit.training)
            raise RuntimeError('teacher failed')
        self.assertIs(dit._lot, original)
        self.assertEqual([m.multiplier for m in network.unet_loras], [.25, 1])
        self.assertTrue(dit.training); self.assertFalse(dit[1].training)

    def test_detail_and_motion_diagnostics_detect_softened_output(self):
        target = torch.arange(48.).reshape(1, 1, 3, 4, 4)
        good = prediction_diagnostics(target, target, target)
        soft = prediction_diagnostics(target * .5, target, target)
        self.assertEqual(good['spatial_detail_ratio'], 1)
        self.assertEqual(good['temporal_teacher_mse'], 0)
        self.assertEqual(soft['spatial_detail_ratio'], .5)
        self.assertGreater(soft['temporal_teacher_mse'], 0)
        self.assertGreater(soft['spatial_teacher_mse'], 0)
        self.assertIsNone(prediction_diagnostics(target[:, :, :1], target[:, :, :1], target[:, :, :1])['temporal_mse'])

    def test_finite_loss_and_teacher_scaling(self):
        require_finite_loss(torch.tensor(1.))
        for loss in (float('nan'), float('inf')):
            with self.assertRaises(FloatingPointError): require_finite_loss(torch.tensor(loss))
        self.assertEqual(float(teacher_distance(torch.ones(1), torch.zeros(1), torch.tensor([.5]))), 4)
        model = torch.nn.Linear(1, 1); before = copy.deepcopy(model.state_dict())
        model(torch.ones(1)).sum().backward()
        model.weight.grad.fill_(float('inf'))
        with self.assertRaises(RuntimeError):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        self.assertTrue(torch.equal(before['weight'], model.weight))


if __name__ == '__main__': unittest.main()
