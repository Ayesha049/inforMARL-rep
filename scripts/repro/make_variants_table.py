#!/usr/bin/env python
"""Compare evaluation-protocol variants (see eval_variants.sh) with the Table 2
protocol and with the paper's Tables 1 and 2.

usage: python scripts/repro/make_variants_table.py results_repro/eval results_repro/eval_variants
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Table 2 (train n, test m): Reward/m, T, (# col)/m, S%
T2 = {
    (3, 3): (63.21, 0.39, 0.40, 100), (7, 3): (63.25, 0.40, 0.46, 100), (10, 3): (62.87, 0.40, 0.49, 99),
    (3, 7): (61.16, 0.38, 0.74, 100), (7, 7): (62.23, 0.40, 0.66, 100), (10, 7): (61.32, 0.40, 0.70, 100),
    (3, 10): (58.59, 0.38, 0.95, 100), (7, 10): (58.23, 0.40, 0.88, 99), (10, 10): (58.67, 0.39, 0.87, 100),
}
# Table 1, InforMARL at N = n (team Reward, T, total # col, S%)
T1 = {3: (205.24, 0.17, 1.45, 100), 7: (399.01, 0.37, 3.72, 100), 10: (429.14, 0.39, 4.73, 100)}

LABEL = {"T2": "Table 2 protocol: 0-10 obstacles, deterministic",
         "A": "A: 3 obstacles, deterministic",
         "B": "B: 0-10 obstacles, stochastic",
         "C": "C: 3 obstacles, stochastic (training conditions)"}


def load(eval_dir, variants_dir):
    vals = defaultdict(list)  # (variant, n, m, key) -> per-seed values
    files = [(f, "T2") for f in Path(eval_dir).glob("informarl_n*_s*.json")]
    files += [(f, f.stem.rsplit("_", 1)[1]) for f in Path(variants_dir).glob("informarl_n*_s*_*.json")]
    for f, v in files:
        n = int(re.match(r"informarl_n(\d+)_s\d+", f.stem).group(1))
        for row in json.load(open(f))["summary"]:
            for k in ("reward_per_m", "T", "col_per_m", "S_pct", "col_aa_per_m", "col_obs_per_m"):
                if k in row:
                    vals[(v, n, row["m"], k)].append(row[k])
    return vals


def main(eval_dir, variants_dir):
    vals = load(eval_dir, variants_dir)
    cells = sorted({(n, m) for (_, n, m, _) in vals if m in (n, 7)})
    print("# Evaluation-protocol variants\n")
    print("Mean ± std over 3 seeds, 100 episodes each. `team` = Reward/m × m. "
          "Paper rows: Table 2 for every cell; Table 1 (InforMARL, N = n) where m = n.\n")
    print("| Train n | Test m | Protocol | Reward/m | team | T | (# col)/m | agent-agent/m | obstacle/m | S% |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for n, m in cells:
        for v in ("T2", "A", "B", "C"):
            r = vals.get((v, n, m, "reward_per_m"))
            if not r:
                continue
            g = lambda k: (np.mean(vals[(v, n, m, k)]), np.std(vals[(v, n, m, k)]))
            rw, t, c, s = g("reward_per_m"), g("T"), g("col_per_m"), g("S_pct")
            aa = f"{np.mean(vals[(v, n, m, 'col_aa_per_m')]):.2f}" if vals.get((v, n, m, "col_aa_per_m")) else "–"
            ob = f"{np.mean(vals[(v, n, m, 'col_obs_per_m')]):.2f}" if vals.get((v, n, m, "col_obs_per_m")) else "–"
            print(f"| {n} | {m} | {LABEL[v]} | {rw[0]:.2f} ± {rw[1]:.2f} | {rw[0]*m:.0f} | {t[0]:.2f} | "
                  f"{c[0]:.2f} | {aa} | {ob} | {s[0]:.0f} ± {s[1]:.0f} |")
        p = T2.get((n, m))
        if p:
            print(f"| {n} | {m} | **paper Table 2** | {p[0]:.2f} | {p[0]*m:.0f} | {p[1]:.2f} | {p[2]:.2f} | – | – | {p[3]} |")
        if m == n:
            q = T1[n]
            print(f"| {n} | {m} | **paper Table 1** | {q[0]/n:.2f} | {q[0]:.0f} | {q[1]:.2f} | {q[2]/n:.2f} | – | – | {q[3]} |")


if __name__ == "__main__":
    main(*(sys.argv[1:3] if len(sys.argv) > 2 else ("results_repro/eval", "results_repro/eval_variants")))
