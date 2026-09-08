# Demonstration images

Images screened with `backend/scripts/screen_demo_images.py` and verified
against a running backend, so the numbers below are what the system actually
produces rather than what it ought to produce. Roughly a quarter of images move
the prediction by 0 km under intervention, so do not substitute an unscreened
photograph without running the screen first.

## `kauai-hanalei.png`

Baseline prediction: **Hanalei, Kauai** — `22.1980, -159.6219`.

### Manual region

Draw a rectangle over the centre of the image, about `x=0.30 y=0.30 w=0.40
h=0.40`, which lands 64 of the 256 patches inside the region. With `a=+2` and
`b=-2`:

| Active layers | Prediction | Shift |
|---|---|---:|
| All 24 | Kula, Maui — `20.7696, -156.2893` | **380 km** |
| Layer 23 only | Kailua, Oahu — `21.3879, -157.8102` | **208 km** |
| Layers 0-7 only | Hanalei — unchanged | **0 km** |
| All biases zero | Hanalei — identical to baseline | **0.0 km** |

The last row is worth showing: `a = b = 0` is an exact no-op, so the baseline
on screen is the unmodified model rather than an approximation.

### WeDetect proposals

This is a landscape photograph, so its highest objectness is only `0.302` and
the default `0.4` threshold returns nothing. Set the threshold to **0.05** and
the proposal limit to **4**, then use *Evaluate every proposal independently*:

| Objectness | Patches | Shift |
|---:|---:|---:|
| 0.084 | 84 | 397 km |
| 0.302 | 56 | 318 km |
| 0.061 | 154 | 210 km |
| 0.127 | **208** | **0.0 km** |

The largest region covers 81% of the patches and moves the prediction not at
all, while a 56-patch region moves it 318 km. That contrast is the answer to
the obvious objection that the effect merely tracks how much of the image was
perturbed.

A street-level photograph would return proposals at the paper's `0.4`
threshold; screen candidates first:

    .venv/Scripts/python.exe scripts/screen_demo_images.py <dir> --threshold 0.4
