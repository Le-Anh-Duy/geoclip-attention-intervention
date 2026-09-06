# Grounding DINO Img2GPS3K data preparation

Companion documentation for `grounding_dino_prepare_img2gps3k_colab.ipynb`.

## Purpose

The notebook converts an Img2GPS3K image directory, metadata JSON, and text vocabulary into the manifest consumed by `geoclip_group_random_search_colab.ipynb`. Grounding DINO runs independently from GeoCLIP; it never receives ground-truth coordinates as model input.

## Expected input

```text
img2gps3k/
├── images/
│   ├── <image-id>.jpg
│   └── ...
└── im2gps3k_metadata.json

geolocation_vocab.txt
```

The checked local dataset contains 2,997 metadata records and 2,997 matching JPEG files. Metadata is keyed by image stem and stores ground truth as:

```json
{
  "<image-id>": {
    "gt_coords": [32.325436, -64.764404]
  }
}
```

The metadata's stored `image_path` is not used because it contains the stale directory name `im2gps3k`. The notebook resolves each key as `images/<image-id>.jpg`. Missing metadata-linked images remain an error, while extra files without metadata—such as Google Drive copies ending in `(1)`—are reported and ignored.

## Vocabulary

Use one concept per UTF-8 line. Blank lines, duplicate phrases, and lines beginning with `#` are ignored.

The included `geolocation_vocab.txt` contains the eight concepts already associated with this dataset:

```text
street sign
utility pole
road marking
bollard
license plate
architecture
vegetation
sidewalk tiles
```

## Colab configuration

```python
DATASET_DIR = Path('/content/dataset/img2gps3k')
VOCAB_TXT = Path('/content/geolocation_vocab.txt')
MODEL_ID = 'IDEA-Research/grounding-dino-base'
BOX_THRESHOLD = 0.4
TEXT_THRESHOLD = 0.3
LIMIT = None
```

Use a GPU runtime. For a smoke test, set `MODEL_ID` to `IDEA-Research/grounding-dino-tiny` and `LIMIT` to a small number. Restore `grounding-dino-base` and `LIMIT = None` for the full run.

The notebook uses the official Transformers Grounding DINO interface and returns boxes as `[x1, y1, x2, y2]` in original-image pixels. Model boxes are clamped to image bounds; degenerate boxes are discarded.

## Resume behavior

`checkpoint.jsonl` receives one flushed JSON record after each image:

- `ok`: one or more valid detections.
- `no_boxes`: inference completed but no detection passed the thresholds.
- `error`: loading or inference failed.

Rerunning the detection cell skips `ok` and `no_boxes` IDs. Errors are retried. The checkpoint is append-only, and the most recent record for an ID wins.

A signature covers the resolved model revision, thresholds, and vocabulary. If those settings change, the notebook refuses to mix new detections into the old checkpoint; select a new `OUTPUT_DIR` for the new experiment.

## Output

Files are written under `DATASET_DIR/grounding_dino_output`:

| File | Contents |
|---|---|
| `checkpoint.jsonl` | Resumable per-image results, including errors. |
| `manifest.json` | Completed records in the GeoCLIP notebook schema. |
| `run_summary.json` | Model revision, thresholds, vocabulary, coverage, errors, and box counts. |

Each manifest record looks like:

```json
{
  "id": "1000269685_e60e9cdfb4_1125_78841376@N00",
  "image": "../images/1000269685_e60e9cdfb4_1125_78841376@N00.jpg",
  "boxes": [[120.5, 80.0, 460.0, 390.0]],
  "ground_truth": {"lat": 32.325436, "lon": -64.764404},
  "detections": [
    {
      "label": "street sign",
      "score": 0.81,
      "box": [120.5, 80.0, 460.0, 390.0]
    }
  ],
  "detection_status": "ok",
  "image_size": {"width": 1024, "height": 768}
}
```

Records with no detections remain in the manifest with `boxes: []`. Grounding DINO boxes are never removed based on CLIP's crop. The downstream GeoCLIP notebook maps boxes through CLIP's resize and center crop, reports IDs whose resulting patch mask is empty, and excludes those IDs from intervention search.

## Quality check

The final cell samples successful records and draws their boxes, labels, and scores. Inspect this grid and `run_summary.json` before starting the more expensive GeoCLIP random search. Threshold changes produce a different dataset and should therefore be recorded as part of the experiment configuration.
