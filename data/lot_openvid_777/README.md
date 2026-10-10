# OpenVidHD H3 starter set

This directory contains a curated **777-clip** starter set for `scripts/lot/train_h3.py`.
The source is the OpenVidHD subset of OpenVid-1M, a text-to-video dataset released under
CC BY 4.0. The clips here were filtered from archive part 1 at dataset revision
`d8a63bd22989c80b5734ec2bb989f4e1b61a5807` to keep the download bounded.

## Curation and H3 format

- One clip per source video, selected deterministically with seed `20261009`.
- Source filters: aesthetic score ≥ 6.0; temporal-consistency score ≥ 0.99; motion score
  between 1 and 25; duration 3–15 seconds; source frame rate ≥ 23.5 fps; caption length
  200–1,200 characters.
- The selection balances 389 static-camera clips and 388 clips tagged with camera movement.
- Each source contributes one deterministic 22-frame window, converted to 24 fps, scaled and
  center-cropped to 640×384 (multiples of 32), H.264, silent. Each `.mp4` has a matching `.txt`
  caption sidecar. This matches Fizgig's 17n+5 frame rule (`22 = 17×1 + 5`) and the LoT 12×20
  token grid used by the guide.
- Selection scores, original source filename, sample time, and caption are recorded in
  `manifest.csv`.

## License and attribution

OpenVidHD is listed as CC BY 4.0. Attribute the source as: *OpenVid-1M / OpenVidHD, NJU-PCALab,
CC BY 4.0*; see the [dataset card](https://huggingface.co/datasets/nkp37/OpenVid-1M) and
[license text](https://creativecommons.org/licenses/by/4.0/). Keep this attribution with any
redistribution. The source records are from OpenVidHD archive part 1; the exact filenames are in
`manifest.csv`.

## Prepare the Fizgig cache

Copy `clips/` to the GPU host, for example `/data/lot_openvid_777/clips/`, then use
[`dataset.toml.example`](dataset.toml.example). Cache video latents and text with the commands in
[`scripts/lot/README.md`](../../scripts/lot/README.md#2-cache-latents-and-text-fizgig), then pass the
resulting cache directory to `train_h3.py --cache`.

The videos are intentionally kept out of Git; only this README, the manifest, and the example
configuration are tracked.
