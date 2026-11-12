#!/usr/bin/env python
"""Aggregate eval_transfer.py outputs into a Table 2 style markdown table.

Each cell shows mean ± std over training seeds (each seed's value is the mean
over 100 test episodes), next to the number reported in Nayak et al. (2023),
Table 2.

usage: python scripts/repro/make_table2.py results_repro/eval > table2.md
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Nayak et al. (ICML 2023), Table 2: PAPER[m][n] = (Reward/m, T, (#col)/m, S%)
PAPER = {
    3: {3: (63.21, 0.39, 0.40, 100), 7: (63.25, 0.40, 0.46, 100), 10: (62.87, 0.40, 0.49, 99)},
    7: {3: (61.16, 0.38, 0.74, 100), 7: (62.23, 0.40, 0.66, 100), 10: (61.32, 0.40, 0.70, 100)},
    10: {3: (58.59, 0.38, 0.95, 100), 7: (58.23, 0.40, 0.88, 99), 10: (58.67, 0.39, 0.87, 100)},
    15: {3: (53.19, 0.39, 1.28, 100), 7: (53.46, 0.40, 1.21, 99), 10: (54.21, 0.40, 1.20, 99)},
}
METRICS = [("reward_per_m", "Reward/m", "{:.2f}"), ("T", "T", "{:.2f}"),
           ("col_per_m", "(# col)/m", "{:.2f}"), ("S_pct", "S%", "{:.0f}")]


def main(eval_dir: str):
    # vals[(n, m, metric)] -> list over seeds
    vals = defaultdict(list)
    seeds = defaultdict(set)
    for f in sorted(Path(eval_dir).glob("informarl_n*_s*.json")):
        mt = re.match(r"informarl_n(\d+)_s(\d+)", f.stem)
        n, s = int(mt.group(1)), int(mt.group(2))
        seeds[n].add(s)
        for row in json.load(open(f))["summary"]:
            for key, _, _ in METRICS:
                vals[(n, row["m"], key)].append(row[key])
    ns = sorted(seeds)
    ms = sorted({k[1] for k in vals})
    print("# InforMARL Table 2 reproduction (Target task)\n")
    print("Cells: ours = mean ± std over seeds (100 test episodes per seed); "
          "paper = Nayak et al. (2023) Table 2.\n")
    print("Seeds per training size: " + ", ".join(f"n={n}: {sorted(seeds[n])}" for n in ns) + "\n")
    header = "| Test | Metric | " + " | ".join(f"n={n} ours | n={n} paper" for n in ns) + " |"
    print(header)
    print("|" + "---|" * (2 + 2 * len(ns)))
    for m in ms:
        for key, label, fmt in METRICS:
            cells = []
            for n in ns:
                v = vals.get((n, m, key), [])
                if v:
                    mu, sd = np.mean(v), np.std(v)
                    cells.append(f"{fmt.format(mu)} ± {fmt.format(sd)}")
                else:
                    cells.append("–")
                p = PAPER.get(m, {}).get(n)
                idx = [k for k, _, _ in METRICS].index(key)
                cells.append(fmt.format(p[idx]) if p else "–")
            print(f"| m={m} | {label} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results_repro/eval")
