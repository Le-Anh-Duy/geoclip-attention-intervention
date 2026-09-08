"""Turn test-intervention/results/*/per_image.csv into the paper's table CSV.

    python paper/scripts/collect_results.py [--all-images]

By default metrics are restricted to images for which WeDetect returned at
least one proposal; the others are a no-op for every intervened row and would
only dilute the comparison. Writes paper/tables/paper_table.csv, which
make_tables.py renders into LaTeX.
"""
import csv
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / 'test-intervention' / 'results'
OUT = ROOT / 'paper' / 'tables'
THRESHOLDS = (1, 25, 200, 750, 2500)

# config key -> (result directory, distance column, predicted lat/lon columns, deltas)
RUNS = [
    ('baseline', 'img2gps3k_wedetect_union_all_layers_a2_bneg2_threshold_0.4',
     'baseline_distance_km', ('baseline_lat', 'baseline_lon'), None),
    ('amplify_all_layers', 'img2gps3k_wedetect_union_all_layers_a2_bneg2_threshold_0.4',
     'union_distance_km', ('union_lat', 'union_lon'), (2.0, 2.0, 2.0)),
    ('amplify_last_layer', 'img2gps3k_wedetect_union_threshold_0.4',
     'union_distance_km', ('union_lat', 'union_lon'), (0.0, 0.0, 1.0)),
    ('top_proposal_last_layer', 'img2gps3k_wedetect_threshold_0.4',
     'top_objectness_distance_km', ('top_objectness_lat', 'top_objectness_lon'), (0.0, 0.0, 1.0)),
    ('oracle_proposal', 'img2gps3k_wedetect_threshold_0.4',
     'oracle_proposal_distance_km', ('oracle_proposal_lat', 'oracle_proposal_lon'), (0.0, 0.0, 1.0)),
    ('oracle_with_baseline', 'img2gps3k_wedetect_threshold_0.4',
     'oracle_with_baseline_distance_km', None, (0.0, 0.0, 1.0)),
]


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    value = (math.sin((lat2 - lat1) / 2) ** 2
             + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 6371.0088 * 2 * math.asin(math.sqrt(min(1.0, value)))


def read(directory, with_proposals_only):
    with open(RESULTS / directory / 'per_image.csv', newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    return [r for r in rows if not with_proposals_only or int(r['proposal_count']) > 0]


def metrics(distances):
    return {
        'mean_distance_km': statistics.fmean(distances),
        'median_distance_km': statistics.median(distances),
        **{f'acc_{t}km': sum(d <= t for d in distances) / len(distances) for t in THRESHOLDS},
    }


def collect(with_proposals_only):
    table = []
    baseline_mean = None
    for key, directory, column, coordinates, deltas in RUNS:
        rows = read(directory, with_proposals_only)
        distances = [float(r[column]) for r in rows]
        summary = metrics(distances)
        if key == 'baseline':
            baseline_mean = summary['mean_distance_km']

        if coordinates is None or key == 'baseline':
            displacement = [0.0] * len(rows)
        else:
            displacement = [haversine_km(float(r['baseline_lat']), float(r['baseline_lon']),
                                         float(r[coordinates[0]]), float(r[coordinates[1]]))
                            for r in rows]

        errors = [float(r[column]) - float(r['baseline_distance_km']) for r in rows]
        table.append({
            'config': key,
            'delta_1': '' if deltas is None else deltas[0],
            'delta_2': '' if deltas is None else deltas[1],
            'delta_3': '' if deltas is None else deltas[2],
            'mask': '' if deltas is None else 'real',
            'n': len(rows),
            **summary,
            'mean_error_delta_vs_baseline_km': summary['mean_distance_km'] - baseline_mean,
            'mean_displacement_km': statistics.fmean(displacement),
            'median_displacement_km': statistics.median(displacement),
            'worsened': sum(e > 1e-9 for e in errors) / len(errors),
            'improved': sum(e < -1e-9 for e in errors) / len(errors),
            'unchanged': sum(abs(e) <= 1e-9 for e in errors) / len(errors),
        })
    return table


def outcomes(with_proposals_only):
    """How many images the all-layer amplification helps, hurts and leaves alone."""
    rows = read('img2gps3k_wedetect_union_all_layers_a2_bneg2_threshold_0.4', with_proposals_only)
    deltas = [float(r['union_distance_km']) - float(r['baseline_distance_km']) for r in rows]
    return {
        'n': len(rows),
        'worsened': sum(d > 1e-9 for d in deltas),
        'improved': sum(d < -1e-9 for d in deltas),
        'unchanged': sum(abs(d) <= 1e-9 for d in deltas),
        'mean_error_increase_km': statistics.fmean(deltas),
    }


if __name__ == '__main__':
    with_proposals_only = '--all-images' not in sys.argv
    table = collect(with_proposals_only)

    OUT.mkdir(exist_ok=True)
    target = OUT / 'paper_table.csv'
    with open(target, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)

    facts = {
        'subset': 'images with at least one proposal' if with_proposals_only else 'all images',
        'n_evaluated': table[0]['n'],
        'n_excluded_empty_mask': 2997 - table[0]['n'],
        'outcomes_amplify_all_layers': outcomes(with_proposals_only),
        'source_runs': sorted({directory for _, directory, _, _, _ in RUNS}),
    }
    (OUT / 'paper_facts.json').write_text(json.dumps(facts, indent=2), encoding='utf-8')

    print(f'Wrote {target} ({facts["subset"]}, N={facts["n_evaluated"]})')
    for row in table:
        print(f"  {row['config']:<24} mean {row['mean_distance_km']:8.1f} km   "
              f"median {row['median_distance_km']:7.1f} km   "
              f"acc@25 {100 * row['acc_25km']:5.1f}%   "
              f"displacement {row['mean_displacement_km']:7.0f} km")
    print('  outcomes:', json.dumps(facts['outcomes_amplify_all_layers']))
