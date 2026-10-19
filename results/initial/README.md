# Initial baseline — 2026-10-19

Command: `python baseline.py --output results/initial`

Three prompts, ten random controls per signal and prompt, 20% token-embedding ablation. The original prediction remains the target in every comparison. Values below are mean clean-minus-ablated target log probabilities (nats).

| Signal | Top | Bottom | Random |
| --- | ---: | ---: | ---: |
| Mean attention | 2.0302 | 2.0703 | 2.2626 |
| Gradient L2 | 4.8189 | 1.6961 | 2.2626 |
| Absolute gradient × input | 4.5876 | 1.5873 | 2.2626 |

Gradient rankings outperform attention on this tiny smoke test under zero-embedding ablation. Attention's top-ranked tokens do not outperform its bottom or random controls on average. These observations do not establish general faithfulness; prompt count is only three, variance is large, and intervention choice matters.

See `metrics.json` for individual trials, token scores, target values, package versions, pinned model revision, and timings. `summary.csv` reports the aggregate table with across-prompt standard deviations. Install `requirements-lock.txt` to reproduce the tested dependency versions on a compatible Python/platform.

Validation: four offline unit/integration tests passed. A second full run with the same settings and cached weights matched every non-timing JSON field exactly. The clean predictions were ` the`, ` cold`, and ` paper`, respectively; these are model predictions, not externally supplied correct answers.
