# LoT H3 holistic review — 2026-10-10

Local research review. No GPU request, optimizer update, dependency installation, or training-code change was made for this review.

## Decision

The goal assumed here is useful H3 video with lower compute while preserving the dense model's composition, detail, and motion. Numerical preflight passes establish that the implementation can run; they do not establish this quality goal.

The earlier recommendation to simply continue to 1,000 was too confident. Longer training remains a worthwhile control because low-token validation metrics continue improving. The better next experiment is a small comparison between the existing recipe and a teacher-preserving dense objective, after removing evaluation and sampler confounds. A new experiment branch can start from the saved step-500 adapter/LoRA; a fresh base-model initialization is unnecessary.

## 1. Correct the interpretation of generation metrics

The existing selector report compares generated final latents with the original source video's clean latent. Generation is text-to-video, not reconstruction. That source video can have a different person, pose, framing, and motion from a valid generated result. Source-latent MSE cannot be the primary quality ranking.

I re-scored all saved outputs against the same-seed, same-caption frozen dense generation and the trained dense-LoRA generation. This used only existing local tensors.

| Selector | MSE to frozen dense generation | MSE to trained dense-LoRA generation |
| --- | ---: | ---: |
| uniform2 | 0.52281 | 0.42197 |
| detail inference | 0.53622 | 0.38132 |
| entropy inference | 0.52604 | 0.38956 |
| reference-guided detail | 0.51909 | 0.39363 |

Detail is about 9.6% closer to the current trained dense output than uniform2 on these four clips. However, its paired 95% clip-bootstrap interval versus uniform2 includes zero for both anchors. There is no reliable winner. Dense LoRA itself differs from frozen dense by 0.28248 MSE. Sample-faithfulness metrics are useful for distillation, but still require prompt-quality and visual checks.

Keep the source comparison as a secondary diagnostic. Report frozen-teacher and current-student faithfulness separately. Rename the current "oracle upper bound" to "reference-guided diagnostic": source-video detail is not an optimal layout for a different generated scene.

Evidence and per-clip statistics: `scripts/lot/runs/fal/h3-lot-openvid-pilot-v2/results/holistic-review-metrics.json`.

## 2. The objective has a preservation gap

`train_h3.training_objective` applies teacher distillation only when `kind != "dense"`. Dense steps use the data loss alone. They exercise the full token path, but do not explicitly preserve frozen H3 predictions. This is a design tradeoff, not a numerical bug.

The evidence warrants testing that tradeoff. At sigma 0.9, the step-500 spatial-detail amplitude relative to the frozen dense teacher averages 0.683 for dense LoRA and 0.463 for uniform2. Dense LoRA remains approximately 0.989/0.997 at sigma 0.3/0.6. Much of its data-MSE improvement arises at high noise while detail decreases. This is consistent with smoothing or target conflict, but does not prove the cause.

There were 105 dense updates, of which 59 had sigma >= 0.9. Shared LoRA weights are also used by the coarse paths. Before choosing a larger distillation weight, measure data-loss and teacher-loss gradient agreement by sigma on identical examples. Then test explicit frozen-teacher preservation on dense steps while keeping the coarse recipe unchanged.

The 250-to-500 low-token improvements are real: uniform2/detail50/entropy50 teacher-gap reductions average 0.027/0.030/0.032 over the fixed 12-clip evaluation. Their clean-MSE gains are much smaller and uncertain. Dense LoRA's clean MSE increases by 0.00226 over the same interval. Aggregate teacher-gap improvement alone cannot establish visual improvement.

## 3. Inference is confounded by noise level, budget, and warmup

The 12-step shift-12 Euler schedule freezes adaptive placement after three updates at sigma 0.97297. The layout is chosen from a predicted clean estimate while the state is still at very high noise, then never refreshed. Prediction reliability at that point has not been measured.

The last model query is at sigma 0.52391, followed by a single jump to zero. The 8-step schedule ends its model queries at sigma 0.63158. Neither queries below sigma 0.3. These are properties of this short schedule, not proof of a sampler bug. Eight steps are appropriate for screening, but cannot establish a model's final quality ceiling.

Offline detail50/entropy50 evaluations use 50% of dense tokens and reference-clean layouts. The preview compares predicted selectors at 25%. Failure to transfer between those tests changes both budget and selection input. Evaluate matched 25% and 50% budgets before blaming selector quality.

Uniform2 compresses from the first step; adaptive variants first run three dense steps. Add a fixed layout that switches after the same warmup. Otherwise its comparison with adaptive layout also includes trajectory and state-transition differences. Keep extent-matched root shuffles, which already isolate placement within the adaptive trajectory.

A compact next diagnostic should compare fixed-after-warmup, detail-after-warmup, and matched shuffle at one common budget, with the same dense controls. First establish quality at 50%, then test whether it survives 25%. Select/refresh using noise level or predicted-layout reliability, not an unexamined fixed step count. Any schedule change must apply to dense controls too.

## 4. Training is small, and the video claim is narrow

The run has 741 training sources and 36 held-out sources. The split groups by source ID, and pair/calibration fitting uses training sources; these are good protections against leakage. Twelve held-out clips are repeatedly evaluated, with 24 additional held-out clips designated for confirmation. Keep confirmation data outside tuning decisions.

Training samples with replacement. Five hundred updates do not constitute one complete pass through 741 examples. Under independent uniform draws, expected unique exposure is approximately 364 examples; actual exposure is unknown because training rows omit clip/source IDs. Log those IDs, per-extent counts, and sampled sigma. Consider shuffled coverage before drawing conclusions about the benefit of more steps.

The previews are 22 pixel frames at 24 fps, approximately 0.92 seconds. They test short video behavior, not long-range temporal stability. Framewise spatial selection has no explicit temporal layout regularizer. Track subject motion separately from camera motion; the current metadata only balances the latter. Check the still-frame-fitted extent bank's reconstruction residuals on held-out video latents before increasing decoder capacity.

## 5. This borrows part of BudgetPix

Our implementation combines LoT asymmetric-flow recovery, fitted extent bases, linear extent heads, and rank-16 LoRA with an entropy/detail selector on VAE latents. [BudgetPix](https://karaozgur.com/BudgetPix/static/paper/BudgetPix.pdf) trains dedicated multiscale encoding/decoding modules, includes local patch refinement, fine-tunes its backbone, and periodically rebuilds layouts during sampling. Its pixel-space results do not establish that adding entropy selection alone will improve our latent-video model.

If corrected objective and sampler experiments still lose detail, a small residual nonlinear extent head is a reasonable next capacity hypothesis. Compare it against the existing head with matched training exposure and compute. Increasing LoRA rank, adding a decoder, expanding data, and changing objectives simultaneously would make the outcome difficult to interpret.

The [fork's issue thread](https://github.com/johndpope/MiniMax-H3/issues/1) also reports that an additional low-noise training run lowered loss without clearly improving still-image detail, whereas introducing teacher distillation produced its first visible gain. Its newer mixed-video run has different data and cannot be numerically compared with this pilot. This supports investigating targets and representation alongside duration.

## 6. Run infrastructure needs a concrete update before continuation

Trainer resumption preserves weights, optimizer, and RNG state. The archived step-500 training bundle contains `trainer_state.pt`. Checkpoint/metrics artifacts are locally integrity-verified. These parts of the lifecycle worked.

However, both the Fal request schema and local client cap the absolute target at 500, and preview-checkpoint selection also caps at 500. The proposed target of 1,000 is currently rejected. Extend the absolute target/checkpoint bounds while retaining bounded per-request increments and checkpoint/export safeguards. Also expose the chosen experimental loss settings instead of hardcoding them in the app.

Current preflight validates the existing loss, not a new teacher-preservation term. Run the exact new forward/backward path before optimizer updates.

Source-bundle changes invalidate pair identity even if only preview/client code changes. Cache identity should depend on relevant pair-generation logic and training inputs, while preflight still records the full source. Preview export should avoid redundant historical checkpoint archives when standalone weight artifacts are already preserved. Add timestamps for loading, cache assembly, hashing, sampling, decoding, packaging, upload, and download; current timings leave much of the cold-start interval unexplained.

## Recommended next sequence

1. **Offline corrections — completed in this review.** Re-score saved samples against both dense anchors; retain source-reference scores as secondary; record per-sigma training trends and sampler/budget confounds. Correct earlier report conclusions.
2. **Focused diagnostic.** In one GPU session, test the current checkpoint's data/teacher gradient tradeoff by sigma and compare matched-budget fixed/adaptive trajectories. Use short previews for screening; include a denser schedule comparison if visual fidelity is used to reject a model. Combine work to amortize cold initialization.
3. **Two controlled short training branches.** Warm-start both from the same step-500 weights, use fresh identical optimizers and a predeclared replay of examples/noise/layouts. One branch retains the recipe; one adds frozen-teacher preservation on dense steps. Use roughly 250 updates per branch as a pilot, logging per-sigma loss, detail, temporal metrics, exposure, and compute. This tests the most concrete objective concern without simultaneously altering data, architecture, and selector.
4. **Extend the winner only with evidence.** Require improved held-out teacher faithfulness and detail with no temporal/prompt-quality regression, supported by matched previews. Then train longer and use the reserved confirmation sources before claiming a quality/compute gain. Five hundred steps may be too short, but duration alone is not the established cause of the current artifacts.

Review limits: no new GPU experiment or training has been performed, no gradient-conflict measurement exists yet, four sampled clips have wide uncertainty, visual assessment used first/middle/last contact-sheet frames rather than a full motion-quality study, and no large-sample FVD or perceptual-quality claim is supported.
