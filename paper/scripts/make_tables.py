"""Regenerate the paper's tables from the notebook's paper_table.csv.

    python paper/scripts/make_tables.py <paper_table.csv> [paper_facts.json]

Overwrites paper/tables/tab_main.tex and paper/tables/tab_groups.tex, which
main.tex \\inputs, and prints the loose numbers the prose still needs.
"""
import csv
import json
import sys
from pathlib import Path

TABLES = Path(__file__).resolve().parent.parent / 'tables'
THRESHOLDS = (1, 25, 200, 750, 2500)

MAIN_ROWS = [
    ('baseline', r'GeoCLIP (baseline)'),
    ('suppress', r'\quad + suppress regions'),
    ('amplify', r'\quad + amplify regions'),
    ('searched', r'\quad + searched $\delta$'),
    ('random_control', r'\quad + random regions (control)'),
]

GROUP_ROWS = [
    ('baseline', r'None (baseline)'),
    ('group_early', r'Early ($0$--$7$)'),
    ('group_middle', r'Middle ($8$--$15$)'),
    ('group_late', r'Late ($16$--$23$)'),
    ('searched', r'All three'),
]


def km(value):
    return '--' if value is None else f'{float(value):.1f}'


def pct(value):
    return '--' if value is None else f'{100 * float(value):.1f}'


def delta(row, index):
    value = row.get(f'delta_{index}', '')
    return '0' if value in ('', None) else f'{float(value):.2f}'


def main_table(rows):
    body = []
    for key, label in MAIN_ROWS:
        row = rows.get(key)
        if row is None:
            body.append(f'{label} & ' + ' & '.join(['--'] * 7) + r' \\')
            continue
        cells = [km(row['mean_distance_km']), km(row['median_distance_km'])]
        cells += [pct(row[f'acc_{t}km']) for t in THRESHOLDS]
        body.append(f'{label} & ' + ' & '.join(cells) + r' \\')
    return r"""\begin{table}[t]
\centering
\caption{Geolocation accuracy on Img2GPS3k under attention intervention. The
first row is the unmodified GeoCLIP model; the suppression and amplification
rows apply a uniform bias to every layer.}
\label{tab:main}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{L{3.7cm}|P{1.2cm}P{1.2cm}|P{0.9cm}P{0.9cm}P{0.9cm}P{0.9cm}P{1cm}}
\toprule
\multirow{2}{*}{\textbf{Configuration}} &
\multicolumn{2}{c|}{\textbf{Error (km)}} &
\multicolumn{5}{c}{\textbf{Accuracy (\%) within}} \\
 & Mean & Median & 1 km & 25 km & 200 km & 750 km & 2500 km \\
\midrule
""" + body[0] + '\n\\midrule\n' + '\n'.join(body[1:4]) + '\n\\midrule\n' + body[4] + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def group_table(rows):
    body = []
    for key, label in GROUP_ROWS:
        row = rows.get(key)
        if row is None:
            body.append(f'{label} & ' + ' & '.join(['--'] * 5) + r' \\')
            continue
        cells = [delta(row, 1), delta(row, 2), delta(row, 3),
                 km(row['mean_distance_km']),
                 km(row['mean_error_delta_vs_baseline_km']) if key != 'baseline' else '---']
        body.append(f'{label} & ' + ' & '.join(cells) + r' \\')
    return r"""\begin{table}[h]
\centering
\caption{Effect of restricting the intervention to a single layer group, at the
searched magnitude for that group with the other two parameters zeroed.}
\label{tab:groups}
\setlength{\tabcolsep}{5pt}
\begin{tabular}{L{3.4cm}|P{1.4cm}P{1.4cm}P{1.4cm}|P{1.7cm}P{1.7cm}}
\toprule
\textbf{Active group} & $\delta_1$ & $\delta_2$ & $\delta_3$ &
\textbf{Mean (km)} & \textbf{$\Delta$ base} \\
\midrule
""" + '\n'.join(body) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def report(rows, facts):
    print('\n--- numbers the prose still needs -------------------------------')
    if facts:
        print(f"  N evaluated / excluded : {facts['n_evaluated']} / {facts['n_excluded_empty_mask']}")
        print(f"  T trials, seed         : {facts['num_random_trials']}, {facts['seed']}")
        print(f"  uniform delta          : {facts['uniform_delta']}")
        best = facts.get('best_deltas')
        if best:
            print('  best deltas            : ' + ', '.join(f'{v:.2f}' for v in best))
    for key, row in rows.items():
        print(f"  {key:<16} mean displacement from baseline: "
              f"{float(row['mean_displacement_km']):.0f} km "
              f"(median {float(row['median_displacement_km']):.0f} km)")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    with open(sys.argv[1], newline='', encoding='utf-8') as handle:
        rows = {row['config']: row for row in csv.DictReader(handle)}
    facts = json.loads(Path(sys.argv[2]).read_text(encoding='utf-8')) if len(sys.argv) > 2 else None

    TABLES.mkdir(exist_ok=True)
    (TABLES / 'tab_main.tex').write_text(main_table(rows), encoding='utf-8')
    (TABLES / 'tab_groups.tex').write_text(group_table(rows), encoding='utf-8')
    print(f'Wrote {TABLES / "tab_main.tex"} and {TABLES / "tab_groups.tex"}')
    report(rows, facts)
