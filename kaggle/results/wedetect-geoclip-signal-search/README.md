# WeDetect-guided GeoCLIP signal search (Img2GPS3K, n=2,997)

The private Kaggle notebooks ran offline on an NVIDIA RTX PRO 6000 Blackwell:

- `giahuytran1/wedetect-img2gps3k-cache` version 2
- `giahuytran1/wedetect-geoclip-signal-search` version 1

WeDetect processed the complete test set once in 19.8 seconds. The search then
reused the cached proposals and finished in 481.2 seconds. A deterministic hash
split allocated 1,079 images to discovery and locked 1,918 images for holdout.
Configuration selection never used holdout labels.

## Locked result

Discovery selected layer 2, all heads, all proposals with score at least 0.4,
using attention bias `a=+2, b=-2`.

| Metric | Baseline | Intervention |
|---|---:|---:|
| Mean error (km) | 1,984.157 | 1,976.694 |
| Median error (km) | 390.130 | 395.441 |
| Acc@25 | 0.09854 | 0.09750 |
| Acc@200 | 0.36757 | 0.36496 |
| Acc@750 | 0.62357 | 0.62252 |
| Acc@2500 | 0.80396 | 0.80448 |

The raw mean improvement was +7.464 km; the per-image delta clipped to
[-2,500, 2,500] km was +2.602 km. A 10,000-sample image bootstrap gave a 95%
CI of [-29.86, +42.92] km for the raw mean and [-10.93, +16.75] km for the
clipped mean. This is not evidence of an accuracy improvement. Only 3.49% of
holdout images improved and 4.17% worsened; the remainder kept the same nearest
gallery item.

## Signals beyond choosing one of 24 layers

1. **Depth is the clearest signal.** Layer index correlated with locked-holdout
   clipped improvement at r=-0.818. Single early layers were relatively stable,
   while layers 17, 19, 21, and 23 produced clipped mean deltas of -44.25,
   -51.54, -54.84, and -62.12 km. Discovery and holdout layer profiles still
   correlated at r=0.714, so the depth pattern is more stable than the winning
   individual layer.
2. **Stacking intervention across layers is strongly destructive.** Early
   layers 0-7 yielded -24.28 km clipped mean delta, middle 8-15 -188.86 km,
   late 16-23 -303.25 km, and all-layer boost -252.61 km. All-layer suppress
   was also harmful (-228.38 km), suggesting accumulated perturbation is more
   important than polarity alone.
3. **Head identity did not generalize.** Discovery chose head 5, but it scored
   -1.48 km on holdout. Head 4 was best post hoc (+3.55 km clipped), and must
   not be reported as a selected result. Baseline CLS-to-proposal attention
   enrichment correlated only moderately with causal head effect (r=0.395,
   n=16); high observed attention is not sufficient evidence of causal utility.
4. **CLS-only bias is much less destructive.** Applying the three-layer window
   only to the CLS query yielded +2.22 km clipped mean delta and Acc@25 0.10167,
   versus 0.09854 baseline. This is exploratory rather than locked selection,
   but motivates separating direct CLS aggregation from patch-to-patch effects.
5. **Small feature changes can cause large geographic jumps.** The locked
   intervention retained mean feature cosine 0.99917 yet moved its predicted
   coordinate by 141.05 km on average. This indicates nearest-neighbor gallery
   boundary amplification and is a useful output-space faithfulness signal.
6. **Simple confidence surrogates were uninformative.** Proposal count versus
   best-layer delta had r=-0.012, and baseline retrieval margin versus prediction
   shift had r=-0.052.

## Proposal cache and dose

At score thresholds 0.15, 0.2, 0.3, 0.4, and 0.5, WeDetect retained respectively
20,264, 15,579, 9,570, 6,152, and 3,929 boxes. The best post-hoc holdout dose was
threshold 0.3 with all proposals (+5.226 km clipped mean), but it was not selected
on discovery and therefore is only a hypothesis for a future confirmatory run.

Complete CSV/NPZ artifacts and the overview figure are kept under the ignored
`outputs/wedetect-geoclip-signal-search/` directory.
