"""Regenerate the paper's tables from paper_table.csv.

    python paper/scripts/collect_results.py      # writes paper/tables/paper_table.csv
    python paper/scripts/make_tables.py          # renders both LaTeX tables

Overwrites paper/tables/tab_main.tex and paper/tables/tab_effect.tex, which the
sections \\input, and prints the loose numbers the prose still needs. A
configuration missing from the CSV renders as a row of dashes.
"""
import csv
import json
from pathlib import Path

TABLES = Path(__file__).resolve().parent.parent / 'tables'
THRESHOLDS = (1, 25, 200, 750, 2500)

MAIN_ROWS = [
    ('baseline', r'GeoCLIP (baseline)'),
    ('amplify_all_layers', r'\quad + amplify, all 24 layers'),
    ('amplify_last_layer', r'\quad + amplify, final layer'),
    ('top_proposal_last_layer', r'\quad + amplify top proposal'),
    ('random_control', r'\quad + random regions (control)'),
    ('oracle_proposal', r'Oracle over proposals'),
    ('oracle_with_baseline', r'Oracle incl. baseline'),
]

EFFECT_ROWS = MAIN_ROWS[1:]


def km(row, field):
    return '--' if row is None else f'{float(row[field]):.1f}'


def pct(row, field):
    return '--' if row is None else f'{100 * float(row[field]):.1f}'


def main_table(rows):
    body = []
    for key, label in MAIN_ROWS:
        row = rows.get(key)
        cells = [km(row, 'mean_distance_km'), km(row, 'median_distance_km')]
        cells += [pct(row, f'acc_{t}km') for t in THRESHOLDS]
        body.append(f'{label} & ' + ' & '.join(cells) + r' \\')
    return r"""\begin{table}[t]
\centering
\caption{Geolocation accuracy on Img2GPS3k under attention intervention,
restricted to the images for which the proposal generator returns at least one
region. The first row is the unmodified GeoCLIP model; the oracle rows select,
per image, the proposal minimising the error and are an upper bound rather than
a deployable configuration.}
\label{tab:main}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{L{3.7cm}|P{1.2cm}P{1.2cm}|P{0.9cm}P{0.9cm}P{0.9cm}P{0.9cm}P{1cm}}
\toprule
\multirow{2}{*}{\textbf{Configuration}} &
\multicolumn{2}{c|}{\textbf{Error (km)}} &
\multicolumn{5}{c}{\textbf{Accuracy (\%) within}} \\
 & Mean & Median & 1 km & 25 km & 200 km & 750 km & 2500 km \\
\midrule
""" + body[0] + '\n\\midrule\n' + '\n'.join(body[1:5]) + '\n\\midrule\n' + '\n'.join(body[5:]) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def effect_table(rows):
    body = []
    for key, label in EFFECT_ROWS:
        row = rows.get(key)
        cells = [km(row, 'mean_displacement_km'), km(row, 'median_displacement_km'),
                 pct(row, 'worsened'), pct(row, 'improved'), pct(row, 'unchanged')]
        body.append(f'{label.replace(chr(92) + "quad + ", "").capitalize()} & '
                    + ' & '.join(cells) + r' \\')
    return r"""\begin{table}[h]
\centering
\caption{Size and direction of the effect. Displacement separates the
unmodified and intervened predictions; the last three columns give the
proportion of images whose error grows, shrinks and is unchanged.}
\label{tab:effect}
\setlength{\tabcolsep}{5pt}
\begin{tabular}{L{4.0cm}|P{1.6cm}P{1.6cm}|P{1.2cm}P{1.2cm}P{1.2cm}}
\toprule
\multirow{2}{*}{\textbf{Configuration}} &
\multicolumn{2}{c|}{\textbf{Displacement (km)}} &
\multicolumn{3}{c}{\textbf{Images (\%)}} \\
 & Mean & Median & Worse & Better & Same \\
\midrule
""" + '\n'.join(body) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def report(rows, facts):
    print('\n--- numbers the prose still needs -------------------------------')
    if facts:
        print(f"  subset                 : {facts['subset']}")
        print(f"  N evaluated / excluded : {facts['n_evaluated']} / {facts['n_excluded_empty_mask']}")
        outcome = facts['outcomes_amplify_all_layers']
        print(f"  all-layer amplification: +{outcome['mean_error_increase_km']:.0f} km mean error, "
              f"{outcome['worsened']} worse / {outcome['improved']} better / {outcome['unchanged']} same")
    for key, _ in MAIN_ROWS:
        row = rows.get(key)
        if row and key != 'baseline':
            print(f"  {key:<24} error {float(row['mean_error_delta_vs_baseline_km']):+8.1f} km vs baseline")


if __name__ == '__main__':
    with open(TABLES / 'paper_table.csv', newline='', encoding='utf-8') as handle:
        rows = {row['config']: row for row in csv.DictReader(handle)}
    facts_path = TABLES / 'paper_facts.json'
    facts = json.loads(facts_path.read_text(encoding='utf-8')) if facts_path.exists() else None

    (TABLES / 'tab_main.tex').write_text(main_table(rows), encoding='utf-8')
    (TABLES / 'tab_effect.tex').write_text(effect_table(rows), encoding='utf-8')
    print(f'Wrote {TABLES / "tab_main.tex"} and {TABLES / "tab_effect.tex"}')
    report(rows, facts)
