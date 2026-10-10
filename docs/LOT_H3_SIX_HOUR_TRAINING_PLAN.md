# Six-hour H3 LoT training campaign

Status: approved by independent architecture and critic reviews on 2026-10-10; implemented and executed on the Fal RTX PRO 6000 under one preserved six-hour T0. Final campaign record: `scripts/lot/runs/fal/campaigns/lot-h3-six-hour-20261010-r4/report.md` and `report.json`.

Execution summary: both branches ran 500 new updates from the hash-verified parent step-500 weights. At equal 500 updates, preservation reduced validation fixed-50 teacher gap by only 0.7% versus control; its paired 95% interval crossed zero. On 24 untouched confirmation sources, the fixed-50 endpoint was effectively tied (preserve minus control 0.000088; paired 95% CI [-0.001864, 0.002073]). Preservation substantially lowered dense-LoRA teacher gap, but its confirmation detail-deviation metric regressed 5.3% against control. No branch met the promotion threshold, and neither branch met the 3% extension gate. The run ended early after the final confirmation and 40-video preview matrix. Shutdown was verified with two empty runner snapshots at least 30s apart.

## Requirements and budget

Run an informative experiment on one Fal RTX PRO 6000, evaluate during training, deliver a final comparison and recoverable weights, and verify GPU shutdown. Six hours is a maximum **wall-clock campaign window**, including queue/provisioning, diagnostics, training, evaluation, publishing, downloading, and shutdown. Start the clock immediately before the first campaign GPU submission. Prepare code and CPU checks before that clock; deployment must retain zero minimum GPU concurrency.

The campaign may finish early if both branches regress, infrastructure fails safely, or no useful experiment remains. Do not run or keep a GPU idle merely to consume six hours. No extra GPUs, new data downloads, new dependencies, or full-backbone fine-tuning are planned.

Preserve the original `h3-lot-openvid-pilot-v2` step-500 checkpoint and its results. Create a new campaign and separate A/B run directories. Branch-local update 0 means inherited step-500 weights; report inherited steps and new updates separately.

## Principles, drivers, and options

Principles: controlled comparisons; detail/motion quality ahead of aggregate loss; recoverable artifacts before every decision; bounded GPU lifetime; honest inconclusive results.

Drivers: current high-noise smoothing/dense drift; ongoing low-token validation improvement; six-hour budget and expensive cold starts.

Options:

| Option | Benefit | Cost/limitation |
| --- | --- | --- |
| Continue only the current recipe | Least implementation and clean duration study | Cannot test the objective concern; may reinforce smoothing |
| Two matched branches, then extend a supported candidate | Tests one concrete improvement while retaining a duration control | Extra preparation/evaluation and serial GPU time |
| Several new architectures/weights/data recipes | Wider exploration | Six hours is insufficient to isolate causes; multiple weak comparisons |

Choose the two-branch option. Retain unchanged continuation as the control; defer new heads, rank, data, and selector-training changes.

## Experiment specification

Parent for both branches: the identical archived step-500 adapter and LoRA, checked against manifest hashes. Rank16, original pinned base/model/dependency revisions, cached data, fitted bank, entropy calibration, layout proportions, LR 1e-4, adapter LR 1e-4, data weight1, coarse distillation weight1, dense-p 0.2, random-p 0.1, adaptive training budgets25–75%, original noise shifts.

| Branch | Dense training loss | Coarse training loss |
| --- | --- | --- |
| A — control | Existing data loss | Existing data + frozen-dense teacher loss |
| B — preservation | Same data loss + frozen-dense teacher loss, weight 1.0 | Identical to A |

Implement explicit `dense_distill_weight`, default0 for compatibility. Teacher predicts with LoRA off from the same noisy latent, text, and audio inputs; use the current clean-estimate convention and sigma weighting. B is a hypothesis, not a proven improvement. The initial diagnostic records gradient agreement/norms by sigma; do not choose a different B weight after inspecting validation results.

Both branches warm-start weights with **fresh identical optimizer initialization**. Avoid comparing an optimizer-resumed A with a reset B. Generate a fixed replay of clip/source IDs, crop offsets, layout geometry/budget, sigma, video/audio noise seeds. Replay retains the historical sampling distribution, including replacement, for both branches; log actual coverage. Separate evaluation RNG from training RNG. Confirm replay equality and equal-update checkpoints before claiming A/B differences.

Use the existing source-group split:741 training,12 validation,24 confirmation. Reuse train-only bank/calibration. Validation informs interim decisions. Confirmation is evaluated once at the final comparison and never drives continuation or parameter choice.

## Timetable — offsets from first submission

| Window | Work and deliverable |
| --- | --- |
| 0:00–0:40 | Warmup/cache verification, original checkpoint recovery, A/B initialization equality, new-objective backward checks, per-sigma loss/gradient diagnostic, baseline evaluation and matched sampler setup |
| 0:40–2:00 | A/B pilots, serially; target250 and500 new updates each; evaluate every250, screen videos at500, publish each evaluated checkpoint |
| 2:00–4:00 | Extend promising branches at matched750/1000/1500/2000 update milestones as time permits; prefer supported candidate only after equal-update comparison; stop regressions |
| 4:00–5:15 | Stop optimizer updates, evaluate preselected best and last checkpoints; final confirmation and higher-quality video comparison; publish final bundles |
| 5:15–5:45 | Download/integrity verification, final CPU report, checkpoint comparisons, preserve all publication links; stop GPU immediately once exports/essential downloads complete |
| 5:45–6:00 | Shutdown audit/retry reserve; verify zero campaign queued/running requests and zero active campaign GPU runners |

Windows are allocations, not promises of a fixed step count. Previous training logged about836 seconds for500 updates, but new loss and infrastructure overhead can change throughput. Measure initial p95 step/evaluation/export time. Work in250-update training milestones, group up to1,000 new updates into a bounded request when safe, and size subsequent requests from measured throughput and remaining reserves. Within the45minprocessinglimit, subtract model-load time, measuredp95 evaluation/render time, up to15minexportreserve, and3mingrace before granting any optimizer time; stopping/evaluation also respect the absolute phase budget. Cap each branch at4,000 new updates in this campaign; an exploratory winning branch may receive additional updates within that cap only after the paired comparison is logged. This is a ceiling, not a target to force.

All training obeys an absolute optimizer-stop timeT+4:00, independently of step targets. Allow its checkpoint/export cleanup toT+4:15. Nominal final evaluation startsT+4:00; no GPU request of any kind may be submitted afterT+4:30 or after the stricter lifetime admission check below. Late initialization consumes experiment time; it never moves the six-hour deadline or eats export/shutdown reserves.

## Intermediate evaluation and decisions

At every250-update milestone, use the same12 validation clips, sigma 0.3/0.6/0.9, explicit shared noise/text/audio, and matched25%/50% token budgets. Report dense controls, fixed layouts, and the preselected adaptive policy. Compare frozen-base and trained-dense anchors separately. Use reference-clean layout results only as diagnostics; source-video sample MSE must not be the primary generation quality ranking.

Predeclare one primary endpoint: frozen-teacher gap on fixed 50% tilings, equally averaged over1x2 and2x1 orientations and sigma 0.3/0.6/0.9, with whole-clip paired bootstrap. This uses a common deployable geometry rather than oracle-selected positions. Fixeduniform2 at25% and adaptive detail at25%/50% are secondary diagnostics; training adaptive policy remains entropy for both arms.

Metrics: teacher-gap and spatial/temporal teacher disagreement; spatial-detail deviation; dense-path drift; per-extent error; per-sigma values; clip/source coverage; update/exposure counts; non-finite/gradient-clipping events; total GPU phase time, sampling time including selection/warmup, decoder time and export time. Training loss alone cannot select a model.

The inference diagnostic policy is **detail**, with training entropy unchanged. Freeze these settings in the manifest before submission: pure Euler, existing `comfy_sigmas` shift12 with video/audio shifts12/3;8steps for screens,30steps for final quality;CFG1; stored seeds per source. For diagnostic adaptive and fixed-after-warmup controls, run dense until the first Euler boundary with next_sigma<=0.90, then switch once and freeze (refresh0). This later switch is a declared common inference hypothesis, not optimized from A/B outcomes. Keep the historical three-step switch separately labelled if measured. Unit/no-splice parity and parent/frozen baseline must qualify the common schedule before optimizer updates; failure aborts rather than silently selecting another schedule. Direct fixed tilings remain the primary quality/compute baseline.

Interim previews use the first two existing validation prompts (onecamera-moving,onestatic), at local500 and1000updates when time permits, capped at16generated videos per milestone. Final confirmation preview source IDs are chosen deterministically before launch from the24confirmation sources: two static,two moving,without inspecting their generated results.

Mandatory final render matrix is these four prompts across seven configurations: frozen base_dense,parent500 dense,bestA dense,bestB dense,parent500fixed1x2(50%),bestAfixed1x2(50%),bestBfixed1x2(50%) =28generated videos at30steps. At most12additional diagnostic videos may test25%uniform2/detail or a longer-trainedselectedmodel; totalfinalcap40. Reuse one loadedbase in the final comparator while switching verified checkpoint states; reset routing/multipliers between configurations. Check remainingrequest/campaigntime beforeeveryevaluation/render, reserving measuredexporttime. If the mandatory28-video matrix cannot fit, use the first predeclared static/moving pair (14videos) and explicitlylabelreducedvisualscope. Never substitute flatteringprompts. Recordexactsigmas/switchcompute; morestepsdoesnotguaranteequality.

Decision rules, predeclared:

- Stop immediately for non-finite loss/gradients, artifact corruption, unexpected source/config mismatch, or deadline breach. Preserve the latest verified publication; do not auto-start another GPU recovery request outside admission limits.
- Compare A/B at equal new updates. Prefer B only if the predeclared fixed 50% teacher-gap endpoint improves by >=5% versus A at equal updates and its paired12-clip bootstrap95% upper delta is <0; mean detail deviation and temporalteachererror each regress by <=5%, dense teacher-gap is <=1.05 times both A's same-update value and B's initializedbaseline (denominatorfloor1e-6), and no per-sigma mean teacher/detail error regresses by >10%. Screened videos must show no new clear artifacts/prompt failure. Apply symmetric rules to selecting A over B. Improvement confined to dense preservation is insufficient to declare a better sparse generator.
- Keep both branches through at least500 new updates unless safety fails. If results are tied, extend matched branches while their validation trends improve and budget permits; report an unresolved tie if intervals overlap. Do not invent a winner from the four-clip preview.
- If an arm worsens two consecutive250-update evaluations by >10% on a primary error, relative to its initialized baseline, and passes neither the primary endpoint nor the declared quality guards, stop that arm. A clear severe visual regression can also stop it, with the comparison evidence saved.
- If both arms fail the quality guards, stop training early and use the remaining time for final diagnosis/export, then shut down. Do not start an unplanned third recipe.

Best checkpoint selection uses validation only; retain initialized baseline, best, and last per arm. Final objective-choice evidence must use the most recent quality-qualified equal-update A/Bpair. A longer-trained selectedcheckpoint has a separate duration comparison and cannot prove itsobjectivewon. A is warm-start continuation of the lossrecipe with reset optimization, not an exact optimizer continuation. Final confirmation reports both preselected arm best checkpoints even if they do not win. If there is no defensible candidate, step500 remains the retained usable baseline and the report says inconclusive/degraded.

## Six-hour enforcement and shutdown

Use a dedicated private campaign app alias, one GPU, `min_concurrency=0`, `max_concurrency=1`, `keep_alive=0`. Record campaign ID, frozen UTC start/deadline, app revision, request IDs, runner IDs, branch/phase, and parent hashes durably before waiting. Queue submission and reattachment must not reset the deadline; no queue-ahead batches or automatic retries afterSTOP/expiry.

Use three independent layers:

1. **Admission/queue expiry:** submit with Fal `start_timeout<=900s`. Every request must satisfy `remaining >= start_timeout + startup_timeout + request_timeout + termination_grace + 300s safety margin`. Never submit afterT+4:30. This prevents a stale queued request from starting much later after the campaign has ended.
2. **Remote bounded processing:** campaign app `startup_timeout<=900s`, `request_timeout<=2700s`. Pass the absolute UTC campaign/phase deadline into each request; reject expired work. Convert remaining time into the server's monotonic command deadline. Reserve up to900s of each request for checkpoint/publication; trainer stops at a batch boundary and saves before SIGTERM. Existing process-group watchdog escalates after <=180s grace if necessary. No child process may outlive request teardown.
3. **Ownership-aware finalizer:** on success, failure, user stop, or expiry, set durableSTOP, cancel known campaign requests, save last published indices, and terminate only positively identified campaign runner IDs if cancellation/automatic teardown does not finish. Never kill an unrelated shared-app request or clear a stop marker implicitly. Confirm two shutdown snapshots at least30s apart show no campaign queued/running requests and no active campaign GPU runners. `FINALIZED` requires shutdown evidence; otherwise report `shutdown_unverified` and retry within reserve, without starting new work.

Worst conservative lifetime for a submission:15min queue +15min startup +45min processing +3min termination =78min. The latest allowed submission atT+4:30 therefore completes byT+5:48, leaving12min before the hard deadline, even if the client disconnects. Admission also reserves an additional5min margin. Actual Fal timeout semantics/settings must be verified on the deployed revision; if these bounds cannot be established, do not launch the six-hour campaign. API outages or provider failures cannot be concealed as verified shutdown.

Fal's `start_timeout` applies before processing; a client HTTP timeout does not terminate a running job. [Official client docs](https://fal-ai.github.io/fal/client/fal_client.html), [app lifecycle](https://fal.ai/docs/documentation/development/app-lifecycle).

## Artifact retention and monitoring

Atomic resumable local checkpoints every50 updates. Publish complete milestone checkpoints and manifests at each250-update evaluation, every15min of GPU work if slower, before branch/phase changes, and on clean stop. Include LoRA, adapter, optimizer/RNG/replay cursor, run config, calibration identity, metrics, and SHA-256/size manifest. Commit/checksum the bundle before publishing its index; fetch indices and small logs while the job is active.

CDN publication links and verified local files are independent of GPU lifetime. Preserve best and last per branch plus immutable parent. Verify the essential final weights and trainer-state archives locally before normal shutdown. If transfers stall near deadline, the GPU still shuts down on schedule and CPU-only downloading retries from the already published links; report incomplete downloads explicitly. A complete campaign result requires verified downloads, but a download failure never grants GPU time beyond six hours.

Monitoring: durable heartbeat/progress at least every minute during active compute; lightweight request/runner checks every1–2min; full metric decisions every250 updates; visual review at major milestones, not every poll. User updates occur on a branch decision, meaningful metric change, failure/recovery, final report, or shutdown issue. Optional desktop heartbeat wakes the reviewing agent every10–15min, but does not enforce runtime safety. If the agent/local controller disappears, remote deadlines still end work; the next monitor resumes from durable records rather than launching duplicate requests.

Final report by the six-hour endpoint whenever infrastructure permits: original500 versus A/B initialized/best/last, matched update/exposure and GPU-time comparisons, validation and untouched confirmation, paired intervals and per-sigma/detail/temporal tables, seeds/sampler/budgets, videos/contact sheets, recipe recommendation, all failure notes, verified artifact paths, and shutdown receipts. Results may be improved, inconclusive, or degraded.

## Implementation prerequisites before first billed submission

1. Extend `train_h3.training_objective` at `scripts/lot/train_h3.py:294` with explicit dense distillation, correct dense/no-splice handling, strict resume metadata, and exact new-objective backward preflight. Add recorded clip/source IDs, replay cursors, sigma/extent/loss/gradient detail, and absolute batch-boundary deadlines.
2. Build `scripts/lot/training_campaign.py` as a deterministic manifest/state/decision controller. Initialize isolated A/B directories from the verified parent, bind data/replay/seed/config, choose bounded phases and produce final CPU reports. Reuse `run_artifacts.py` integrity/atomic primitives and existing result downloader.
3. Extend campaign-specific Fal request/config at `scripts/lot/fal_h3_app.py:54` and client at `scripts/lot/fal_h3_client.py:388`: support branch targets beyond500 with <=1,000-update request deltas; init/fork provenance; campaign/phase deadlines; configurable loss; dedicated alias/runtime bounds; start_timeout, idempotent reattach, ownership registry and safe cleanup. Preserve old single-pilot behavior. The existing500 limit makes the proposed continuation impossible without this work.
4. Correct matched-budget/noise-level evaluation and fixed-after-warmup controls in `train_h3.py`, `render_h3_pilot.py`, `h3_splice.py`, and `summarize_h3_run.py`. Keep historical sampling as an explicitly labelled comparison. Maintain separate immutable frozen-dense and current-dense generation anchors.
5. First reuse verified existing cache/pair/calibration artifacts through immutable references, without overwriting parent dataset/run. Cache-identity refactoring is optional and performed only if safe reuse fails qualification; it must not silently trust unrelated-source manifests. Keep full-source preflight provenance. Trim redundant preview bundles and add phase timestamps if needed to meet reserves. No resident training-model refactor is required; the final multi-checkpoint renderer reuses the existing sampler functions on one loadedbase.
6. Complete CPU/mock lifecycle and experiment checks; deploy the private campaign revision with zero minimum concurrency; inspect applied timeout/scaling settings. GPU preflight and short timeout qualification are the first timed diagnostic tasks, before any training updates.

## Acceptance criteria and verification

Before launch: parent hashes match; A/B initialized outputs agree under shared noise; only dense loss weight differs; replay is deterministic and confirmation sources excluded; new objective has absent teacher gradients (frozen/no_grad), finite student gradients and nonzero gradients in each exercised trainable path on exact video/audio input; expired/STOP requests cannot start; final artifact selection cannot accept partial/checksum-mismatched checkpoints.

During campaign: max1 GPU and1 request; only bounded admitted requests; original checkpoint untouched; committed milestone artifacts locally/CDN recoverable; every selection recorded with comparison and decision rationale; no inference ranking based solely on unrelated-source MSE.

At completion: report exists with actual budgets/time/updates, final confirmation completed or honestly marked missing, best/last weights and resume archives integrity-verified or pending CDN retries recorded, all campaign requests ended/cancelled, and shutdown verified in two spaced provider snapshots. Strict wall-clock GPU lifetime must stay inside six hours under the qualified provider timeout assumptions.

Tests should target changed scientific/lifecycle contracts: teacher loss on/off dense path, clean-estimate sign/space, A/B initialization/RNG/replay equality, resume weights/config/cursor, matched budgets and warmup, deadline admission including queue/setup/export/clock skew, SIGTERM checkpointing, client loss, STOP/no duplicate requests, corrupted/partial uploads, ownership-safe cancellation/kill, and shutdown confirmation. Follow existing focused suites; do not re-run unrelated environment-broken full tests as a campaign blocker. The detailed cases are in the accompanying test specification.

## Pre-mortem

1. **Local controller dies or queue restarts work late:** server start/runtime limits, frozen absolutedeadline, conservative admission and dedicated alias bound all existing work; reconnect reads records without submitting duplicates.
2. **Lower loss hides blur or branch comparisons are unfair:** same parent/fresh optimizer/replay/budget/sampler, per-sigma detail/temporal guards, frozen/current dense anchors, matched-update and untouched confirmation reports; permit inconclusive outcome.
3. **GPU dies before exporting results or transfer monopolizes time:** atomic checkpoints, repeated CDN indices, early export reserve, best/last retention, CPU-only transfer retries, independent shutdown audit. Never extend GPU lifetime to wait on local downloads.

## ADR

Decision: two serial, matched objective branches, then evidence-driven bounded extension, with final evaluation/export/shutdown inside a six-hour wall-clock envelope.

Drivers: ongoing low-token learning, high-noise dense drift, invalid earlier generation ranking, and user requirement for saved results and GPU shutdown.

Alternatives considered: duration-only continuation remains the control; multiple new architecture/data recipes deferred; a single six-hour request rejected because the current request timeout is two hours and recovery becomes fragile.

Why chosen: isolates the most concrete objective concern and tests duration simultaneously with recoverable increments. Conservative server limits protect against local disconnection and late queue starts.

Consequences: preparation requires code/test/deployment work before the six-hour timer. Startup/export overhead reduces training exposure. Final result may be inconclusive. The plan establishes a wall-clock bound, not a promised monetary total or per-step quality gain.

Follow-ups: larger/head/data experiments only after this comparison; further duration only if quality gates support it; long-video validation remains separate from this short-video pilot.

## Review changes

Architecture review required absent teacher gradients, a bounded named policy/schedule/render matrix, one primary endpoint with numerical guards, explicit optimizer-reset and unequal-extension labels, optional cache-identity work, and evaluation/export reserves inside each request. These changes are incorporated. Architecture and critic subsequently approved. The critic's editorial clarification to the regression gate was applied.

## Execution staffing and handoff

Available roles relevant here: executor, architect, critic, verifier, debugger, test-engineer, writer, explorer. Keep one root owner of campaign/GPU controls. No child agent may launch, cancel, or kill GPU requests.

Ralph path: one root completion owner; bounded executor lanes for (1) objective/replay/evaluation and (2) Fal/controller/artifact lifecycle; test-engineer for fault tests; verifier for exact preflight/artifact/shutdown evidence. Use high reasoning for design/owner and standard-to-high for implementation/verification.

Team path: the same two independent implementation scopes; root integrates, reviews, and alone operates GPU. A verifier checks contract tests and deployed limits before root begins the timed campaign. Finish source implementation first; the running experiment is a single-owner control loop, not parallel GPU workers.

Launch hints after a separate execution request: `$team implement .omx/plans/lot-h3-six-hour-campaign.md using objective/replay and lifecycle/controller lanes; root alone owns GPU operations`; `$ralph execute the verified six-hour campaign with .omx/plans/prd-lot-h3-six-hour-campaign.md and .omx/plans/test-spec-lot-h3-six-hour-campaign.md; preserve absolute deadline and export/shutdown reserves`. No launch hint is an instruction to execute in this planning turn.

Verification handoff: implementation lanes prove meaningful contract/fault tests; root inspects app revision/timeouts and initialization equality; verifier audits milestone exports; root conducts final evaluation and ownership-safe shutdown; verifier checks final artifact manifests and two shutdown receipts before marking complete.
