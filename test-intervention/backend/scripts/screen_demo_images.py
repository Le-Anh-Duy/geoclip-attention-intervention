"""Screen candidate photographs before demonstrating them live.

    python scripts/screen_demo_images.py IMAGE_OR_DIR [...] [--threshold 0.4]

Roughly a quarter of images with proposals do not move the prediction at all
under intervention, which is fine in aggregate but ruins a live demonstration.
For each candidate this reports how many proposals WeDetect returns and how far
each one moves the top-1 prediction, so a strong example can be chosen in
advance rather than discovered on camera.

Requires the backend to be running on http://127.0.0.1:8000.
"""
import argparse
import json
import math
import sys
import urllib.request
import uuid
from pathlib import Path

BACKEND = 'http://127.0.0.1:8000'
SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
ALL_LAYERS_BOOST = {str(layer): [2.0, -2.0] for layer in range(24)}


def post(path, fields, files):
    """Minimal multipart POST; the backend's endpoints are all form-encoded."""
    boundary = uuid.uuid4().hex
    body = bytearray()
    for name, value in fields.items():
        body += (f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'
                 f'{value}\r\n').encode()
    for name, filename, payload in files:
        body += (f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; '
                 f'filename="{filename}"\r\nContent-Type: application/octet-stream\r\n\r\n').encode()
        body += payload + b'\r\n'
    body += f'--{boundary}--\r\n'.encode()

    request = urllib.request.Request(
        f'{BACKEND}{path}', data=bytes(body),
        headers={'Content-Type': f'multipart/form-data; boundary={boundary}'})
    with urllib.request.urlopen(request, timeout=900) as response:
        return json.load(response)


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    value = (math.sin((lat2 - lat1) / 2) ** 2
             + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 6371.0088 * 2 * math.asin(math.sqrt(min(1.0, value)))


def screen(path, threshold, top_k):
    payload = path.read_bytes()
    result = post('/evaluate-proposals',
                  {'layer_configs': json.dumps(ALL_LAYERS_BOOST),
                   'score_threshold': threshold,
                   'proposal_top_k': top_k},
                  [('image', path.name, payload)])

    base = result['baseline']['predictions'][0]
    rows = []
    for entry in result['evaluations']:
        pred = entry['intervention']['predictions'][0]
        rows.append({
            'objectness': entry['proposal']['objectness'],
            'patches': entry['proposal']['patch_count'],
            'shift_km': haversine_km(base['lat'], base['lon'], pred['lat'], pred['lon']),
        })
    rows.sort(key=lambda r: r['shift_km'], reverse=True)
    return {
        'name': path.name,
        'baseline': (base['lat'], base['lon']),
        'proposals': result['raw_proposal_count'],
        'rows': rows,
        'best_shift_km': rows[0]['shift_km'] if rows else 0.0,
        # A demonstrable example needs both a region that moves the prediction a
        # visible distance and one that barely moves it: the contrast is the
        # point, and it is what answers "isn't this just perturbation?" on camera.
        'has_contrast': bool(rows) and rows[0]['shift_km'] > 300 and rows[-1]['shift_km'] < 100,
    }


def collect(targets):
    for target in targets:
        path = Path(target)
        if path.is_dir():
            yield from sorted(p for p in path.iterdir() if p.suffix.lower() in SUFFIXES)
        elif path.is_file():
            yield path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('targets', nargs='+', help='image files or directories')
    parser.add_argument('--threshold', type=float, default=0.4, help='WeDetect objectness threshold')
    parser.add_argument('--top-k', type=int, default=5, help='proposals evaluated per image')
    args = parser.parse_args()

    candidates = list(collect(args.targets))
    if not candidates:
        sys.exit('no images found')
    print(f'{len(candidates)} candidates, ~{args.top_k * 4 + 8}s each\n')

    results = []
    for path in candidates:
        try:
            summary = screen(path, args.threshold, args.top_k)
        except Exception as error:  # a bad file should not abandon the batch
            print(f'{path.name}: FAILED ({error})')
            continue
        results.append(summary)
        verdict = 'GOOD' if summary['has_contrast'] else (
            'weak' if summary['best_shift_km'] > 300 else 'SKIP')
        print(f"{summary['name']:<28} proposals {summary['proposals']:>3}  "
              f"best shift {summary['best_shift_km']:>8.0f} km  {verdict}")
        for row in summary['rows']:
            print(f"    objectness {row['objectness']:.3f}  patches {row['patches']:>3}  "
                  f"shift {row['shift_km']:>8.1f} km")

    usable = [r for r in results if r['has_contrast']]
    print(f'\n{len(usable)}/{len(results)} usable for the demonstration')
    for summary in sorted(usable, key=lambda r: r['best_shift_km'], reverse=True):
        print(f"  {summary['name']}  ({summary['best_shift_km']:.0f} km)")
