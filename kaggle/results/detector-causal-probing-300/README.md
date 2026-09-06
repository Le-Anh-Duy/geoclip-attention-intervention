# Detector causal probing pilot (Img2GPS3K, n=300)

This is the result of Kaggle kernel version 1 at
`giahuytran1/geoclip-detector-causal-probing`. The private notebook ran with
Internet disabled on an NVIDIA RTX PRO 6000 Blackwell Server Edition. The 300
images were round-robin sampled from 36 coarse geographic cells.

## Main result

| Configuration | Mean error (km) | Mean delta (km) | Bootstrap 95% CI (km) | Acc@25 | Acc@200 | Acc@750 | Acc@2500 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 2729.14 | 0.00 | [0.00, 0.00] | 0.300 | 0.450 | 0.557 | 0.710 |
| DINO∩WeDetect boost, early layers 0–7 | 2676.68 | +52.46 | [-75.57, +204.29] | 0.293 | 0.443 | 0.550 | 0.710 |
| DINO boost, all layers | 3014.89 | -285.75 | [-530.46, -64.85] | 0.250 | 0.393 | 0.507 | 0.687 |
| DINO∩WeDetect boost, all layers | 3126.40 | -397.25 | [-703.29, -121.22] | 0.267 | 0.410 | 0.510 | 0.677 |
| DINO∩WeDetect boost, late layers 16–23 | 3191.70 | -462.55 | [-770.68, -198.04] | 0.260 | 0.403 | 0.500 | 0.670 |
| DINO∪WeDetect boost, all layers | 3503.92 | -774.78 | [-1178.59, -423.78] | 0.220 | 0.353 | 0.477 | 0.637 |
| WeDetect boost, all layers | 3635.73 | -906.59 | [-1345.25, -486.67] | 0.230 | 0.377 | 0.487 | 0.630 |

Positive delta means lower geolocation error than baseline. The early-layer
consensus mean is not statistically conclusive: its interval crosses zero.
Only 33/300 top-1 predictions changed under that configuration (17 improved,
16 worsened), so its positive mean is driven by a few large movements. This
pilot supports a causal-probing claim—early layers are more stable and
detector consensus is less destructive than broad union masks—not an accuracy
improvement claim.

## Detector observations

- WeDetect produced at least one retained patch mask for 197/300 images.
- Grounding DINO produced at least one retained patch mask for 193/300 images.
- Their patch-level intersection was non-empty for 98/300 images.
- Mean DINO–WeDetect patch IoU was 0.162 over all images.
- Correlation between WeDetect proposal count and the all-layer WeDetect delta was -0.120.
- Increasing WeDetect coverage from top-1 to top-3 to all proposals remained harmful, although the all-proposal mask was less harmful on mean error than top-1.

The complete per-image table and figures were downloaded locally under the
ignored `outputs/kaggle-detector-causal-probing/` directory.
