# GeoCLIP grouped intervention random search

Companion documentation for `geoclip_group_random_search_colab.ipynb`.

## Purpose

The notebook searches for one global attention-intervention configuration over a complete geolocation dataset. Bounding-box extraction is deliberately out of scope: every dataset record must already contain its image, one or more boxes, and ground-truth GPS coordinates.

GeoCLIP is installed from the Transformers-compatible fork at commit `976a3d494f43c5f43572348571ba38c28c7877b6`:

<https://github.com/Le-Anh-Duy/geo-clip>

## Intervention

The 24 CLIP vision layers are split into three contiguous groups:

| Group | Layers | Parameter |
|---|---:|---:|
| Early | 0–7 | `delta_1` |
| Middle | 8–15 | `delta_2` |
| Late | 16–23 | `delta_3` |

For every layer in group `i`, the pre-softmax key biases are:

```text
a_i = base_a + delta_i   # patches inside any bounding box
b_i = base_b - delta_i   # patches outside all bounding boxes
```

The CLS key remains unbiased. Positive delta values favor boxed regions; negative values favor the outside region. Multiple boxes are combined into one union mask.

## Colab input

Use a GPU runtime. Upload or mount a directory containing a JSON manifest and its images, then set `DATASET_JSON` in the configuration cell.

The manifest is a non-empty JSON array:

```json
[
  {
    "id": "sample-0001",
    "image": "images/sample-0001.jpg",
    "boxes": [
      [120.5, 80.0, 460.0, 390.0],
      [500.0, 100.0, 620.0, 260.0]
    ],
    "ground_truth": {
      "lat": 10.8231,
      "lon": 106.6297
    }
  }
]
```

Rules:

- `image` may be absolute or relative to the manifest directory.
- Each box is `[x1, y1, x2, y2]` in original-image pixels.
- Coordinates must satisfy `0 <= x1 < x2 <= image_width` and `0 <= y1 < y2 <= image_height`.
- Latitude must be in `[-90, 90]`; longitude must be in `[-180, 180]`.
- Each record must contain at least one box that overlaps CLIP's center crop.
- Record IDs must be unique.

## Search configuration

The main settings are in the first Python cell:

```python
BASE_A = 0.0
BASE_B = 0.0
DELTA_LOW = -3.0
DELTA_HIGH = 3.0
NUM_RANDOM_TRIALS = 30
BATCH_SIZE = 8
TOP_K = 5
SEED = 42
OBJECTIVE = "mean_distance_km"
```

The delta range and number of trials are defaults, not experimental claims. Set them before a real benchmark. `OBJECTIVE` supports `mean_distance_km` and `median_distance_km`; both are minimized.

The notebook evaluates zero delta first, followed by uniformly sampled candidates. With `BASE_A = BASE_B = 0`, zero delta is exactly the unmodified baseline and its predictions are reused rather than recomputed.

## Execution flow

1. Install the pinned GeoCLIP fork.
2. Load GeoCLIP once.
3. Encode and normalize the fixed GPS gallery once in bounded CPU batches.
4. Decode and CLIP-preprocess every image once; build its region-to-patch mask.
5. Run the unmodified baseline once over the dataset.
6. For every delta candidate, run the vision encoder over the full dataset and compare its image embeddings against the cached location embeddings.
7. Aggregate top-1 distance and threshold accuracy over the complete dataset.
8. Checkpoint the current search table and best result after every trial.

Only the vision encoder is rerun because its attention changes between candidates. The location encoder and GPS gallery embeddings are never recomputed during random search.

## Metrics and outputs

The notebook reports:

- Mean and median top-1 geodesic distance.
- Accuracy within `1`, `25`, `200`, `750`, and `2500` km.
- Mean-distance improvement relative to the baseline.
- Best-so-far optimization trace, threshold comparison, and distance histograms.

Files are written to `OUTPUT_DIR`:

| File | Contents |
|---|---|
| `baseline_per_image.csv` | Baseline prediction, top-k results, and error for every image. |
| `random_search_results.csv` | Deltas and aggregate metrics for every completed trial. |
| `best_per_image.csv` | Per-image results for the current best candidate. |
| `best_config.json` | Best deltas, layer groups, objective, metrics, and seed. |

## Resource notes

- Preprocessed image tensors are cached in CPU memory for speed. At float32 and `224x224`, budget roughly `0.57 MiB` per image, excluding Python overhead.
- Reduce `BATCH_SIZE` if Colab runs out of GPU memory.
- Search cost is approximately `number_of_candidates × number_of_images` vision forwards. Increasing trials is the primary runtime multiplier.
- Output files are checkpointed, but `/content` is temporary. Set `OUTPUT_DIR` to mounted Google Drive if results must survive a runtime reset.
