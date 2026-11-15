# Evaluation-protocol variants

Mean ± std over 3 seeds, 100 episodes each. `team` = Reward/m × m. Paper rows: Table 2 for every cell; Table 1 (InforMARL, N = n) where m = n.

| Train n | Test m | Protocol | Reward/m | team | T | (# col)/m | agent-agent/m | obstacle/m | S% |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 3 | Table 2 protocol: 0-10 obstacles, deterministic | 66.64 ± 1.62 | 200 | 0.38 | 0.81 | 0.24 | 0.57 | 100 ± 0 |
| 3 | 3 | A: 3 obstacles, deterministic | 69.29 ± 1.28 | 208 | 0.37 | 0.58 | 0.28 | 0.30 | 100 ± 0 |
| 3 | 3 | B: 0-10 obstacles, stochastic | 62.96 ± 1.49 | 189 | 0.39 | 0.79 | 0.22 | 0.57 | 100 ± 0 |
| 3 | 3 | C: 3 obstacles, stochastic (training conditions) | 65.35 ± 1.56 | 196 | 0.38 | 0.55 | 0.24 | 0.31 | 100 ± 0 |
| 3 | 3 | **paper Table 2** | 63.21 | 190 | 0.39 | 0.40 | – | – | 100 |
| 3 | 3 | **paper Table 1** | 68.41 | 205 | 0.17 | 0.48 | – | – | 100 |
| 3 | 7 | Table 2 protocol: 0-10 obstacles, deterministic | 60.70 ± 1.13 | 425 | 0.40 | 1.28 | 0.72 | 0.56 | 99 ± 1 |
| 3 | 7 | A: 3 obstacles, deterministic | 63.56 ± 1.24 | 445 | 0.38 | 1.24 | 0.90 | 0.34 | 100 ± 0 |
| 3 | 7 | B: 0-10 obstacles, stochastic | 56.38 ± 0.80 | 395 | 0.41 | 1.35 | 0.78 | 0.57 | 98 ± 2 |
| 3 | 7 | C: 3 obstacles, stochastic (training conditions) | 60.08 ± 0.99 | 421 | 0.39 | 1.25 | 0.91 | 0.34 | 100 ± 0 |
| 3 | 7 | **paper Table 2** | 61.16 | 428 | 0.38 | 0.74 | – | – | 100 |
| 7 | 7 | Table 2 protocol: 0-10 obstacles, deterministic | 55.20 ± 2.39 | 386 | 0.40 | 1.34 | 0.78 | 0.56 | 100 ± 0 |
| 7 | 7 | A: 3 obstacles, deterministic | 58.84 ± 2.51 | 412 | 0.39 | 1.21 | 0.87 | 0.34 | 100 ± 0 |
| 7 | 7 | B: 0-10 obstacles, stochastic | 48.40 ± 3.16 | 339 | 0.42 | 1.41 | 0.81 | 0.60 | 94 ± 4 |
| 7 | 7 | C: 3 obstacles, stochastic (training conditions) | 50.80 ± 3.77 | 356 | 0.40 | 1.26 | 0.89 | 0.37 | 97 ± 3 |
| 7 | 7 | **paper Table 2** | 62.23 | 436 | 0.40 | 0.66 | – | – | 100 |
| 7 | 7 | **paper Table 1** | 57.00 | 399 | 0.37 | 0.53 | – | – | 100 |
| 10 | 7 | Table 2 protocol: 0-10 obstacles, deterministic | 53.51 ± 2.20 | 375 | 0.43 | 1.40 | 0.84 | 0.55 | 93 ± 5 |
| 10 | 7 | A: 3 obstacles, deterministic | 57.45 ± 1.73 | 402 | 0.41 | 1.16 | 0.84 | 0.32 | 97 ± 2 |
| 10 | 7 | B: 0-10 obstacles, stochastic | 43.11 ± 2.37 | 302 | 0.46 | 1.43 | 0.82 | 0.61 | 88 ± 2 |
| 10 | 7 | C: 3 obstacles, stochastic (training conditions) | 45.38 ± 2.77 | 318 | 0.44 | 1.31 | 0.95 | 0.37 | 87 ± 8 |
| 10 | 7 | **paper Table 2** | 61.32 | 429 | 0.40 | 0.70 | – | – | 100 |
| 10 | 10 | Table 2 protocol: 0-10 obstacles, deterministic | 52.83 ± 1.68 | 528 | 0.42 | 1.77 | 1.23 | 0.54 | 95 ± 0 |
| 10 | 10 | A: 3 obstacles, deterministic | 53.89 ± 2.35 | 539 | 0.42 | 1.51 | 1.18 | 0.33 | 92 ± 4 |
| 10 | 10 | B: 0-10 obstacles, stochastic | 41.16 ± 2.39 | 412 | 0.45 | 2.01 | 1.40 | 0.61 | 87 ± 7 |
| 10 | 10 | C: 3 obstacles, stochastic (training conditions) | 42.31 ± 2.69 | 423 | 0.45 | 1.65 | 1.29 | 0.36 | 81 ± 6 |
| 10 | 10 | **paper Table 2** | 58.67 | 587 | 0.39 | 0.87 | – | – | 100 |
| 10 | 10 | **paper Table 1** | 42.91 | 429 | 0.39 | 0.47 | – | – | 100 |
