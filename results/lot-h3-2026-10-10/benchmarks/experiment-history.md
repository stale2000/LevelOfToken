# LoT H3 experiment history and benchmark notes

This page records what was tried and which results are comparable. Runtime microbenchmarks, held-out latent-prediction metrics, and generated-video examples measure different things. Lower MSE and teacher gap are better; reduced token count is a compute reduction, not a quality result.

## Experiment sequence

### Runtime and token-count tests

On one RTX PRO 4000 Blackwell (24 GB), a streamed 37-frame 768×1344 DiT forward took 51.9 s dense and 19.6 s with LoT (2.64×). At 768×1152 with blocks resident, dense took 0.69 s, bands 0.42 s (1.65×), and uniform 2×2 0.30 s (2.30×). VAE decode measured 31.8 s either way. These are DiT-forward speedups, not end-to-end generation speedups. See [`runtime-benchmarks.csv`](runtime-benchmarks.csv).

### Step-0 layout screen on OpenVidHD

The pilot selected 777 clips and reserved 36 source-disjoint clips: 12 validation and 24 untouched confirmation. The step-0 layout screen used the 12 validation clips with three noise levels each (36 samples). It compared dense LoRA with detail50, entropy50, uniform2 at 25% token fraction, a legacy mosaic, and random controls. Dense LoRA averaged 240 tokens/frame and 0.427 s/forward. Detail50 and entropy50 each averaged 120 tokens/frame and 0.281 s; uniform2 averaged 60 tokens/frame and 0.196 s. The step-0 teacher gaps were 0.416 (detail50), 0.429 (entropy50), and 0.565 (uniform2). This was a small prediction screen, not a final generated-video quality ranking. Raw summaries and per-evaluation rows are in [`openvid-pilot-step0-layout-screen.csv`](openvid-pilot-step0-layout-screen.csv), [`openvid-pilot-step0-baseline.json`](openvid-pilot-step0-baseline.json), and the pilot CSVs.

### Still-image training runs

Three earlier runs tested the training objective on still images. Run 1 reduced visible grid artifacts but left coarse regions soft. Run 2 changed the LoT noise shift to 3; held-out loss improved only about 4–5% and renders changed little. Run 3 added a frozen dense-teacher target. After 2,000 still-image steps, all-2×2 and mosaic outputs were 15.5% and 18.2% closer to dense H3 by the reported weighted gap; all-2×2 was more readable while the coarsest mixed areas remained soft. This is still-image evidence and does not establish transfer to video.

### OpenVidHD video pilot

The video pilot trained the LoT adapter and rank-16 DiT LoRA to step 500, with dense anchors and uniform/detail/entropy/random-budget layouts. Its held-out evaluation reported flow MSE, clean-latent MSE, temporal MSE, teacher gap, spatial detail ratio, and DiT forward time. Step-0 and step-250/500 per-source metrics are included as CSVs. Those validation results were used to choose the parent checkpoint for r4; they are not the later untouched confirmation set.

### Dense-preservation A/B campaign (r4)

Both branches started from the same hash-verified parent step-500 checkpoint and received 500 new updates with matched replay. `control` kept the prior objective; `preserve` added a frozen dense-teacher preservation term on dense steps. On 12 validation clips at step 250, fixed-50 teacher gap was 0.30015 (control) vs 0.30252 (preserve), preserve-minus-control +0.00237, paired 95% interval [+0.00037, +0.00456]. At step 500 it was 0.29698 vs 0.29488, difference −0.00210, interval [−0.00540, +0.00114]; that interval includes zero. Per-source CSVs are included.

On the 24 untouched confirmation sources, parent/control/preserve fixed-50 teacher gaps were 0.300283 / 0.292050 / 0.292138. Preserve-minus-control was +0.000088, paired 95% clip-bootstrap interval [−0.001864, +0.002073]. Control had lower detail deviation (0.253563 vs 0.267013); preserve had lower dense-LoRA teacher gap (0.046739 vs 0.061488) and slightly lower temporal teacher MSE (0.047606 vs 0.047860). No branch met the promotion threshold or the 3% extension gate. The aggregate confirmation result is recorded here. The portable bundle includes r4 validation per-source CSVs; it does not include confirmation per-source CSVs.

### Inference-time selector screens

The team tested predicted-detail and entropy placement against fixed 1×2/2×1 tilings, root-shuffled placement controls, and warm-up switch points σ=0.9, 0.7, and 0.5. Screens used two prompts for an 8-step pass and four prompts for 30-step validation previews. The first generic screen mixed a 25% selector budget with 50% fixed layouts, so it is only a rough failure screen and cannot support a selector-vs-fixed comparison. Later screens matched the 50% budget and reused prompts/noise. They did not complete an untouched confirmation set; no selector winner is claimed.

A temporal-shuffle correction was also planned, but no GPU request was submitted; three CPU layout checks passed and no video result exists.

## How to read these results

Teacher gap measures agreement with frozen dense H3, not human preference. Source-video latent MSE is not a direct generation-quality score because these are text-to-video outputs, not reconstructions. The linked example videos are matched visual comparisons, not independent confirmation of a quality win.

Training data and captions: OpenVid-1M / OpenVidHD, NJU-PCALab, CC BY 4.0.
