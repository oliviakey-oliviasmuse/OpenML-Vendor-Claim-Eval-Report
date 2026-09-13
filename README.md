# OpenML Vendor Claim Evaluation Report

**Lean Six Sigma MBB methodology applied to ML evaluation in capital-intensive manufacturing.**

[![Methodology: Six Sigma + ML](https://img.shields.io/badge/methodology-6%CF%83%20%2B%20ML-blue)](https://en.wikipedia.org/wiki/Six_Sigma)
[![Dataset: OpenML 42890](https://img.shields.io/badge/dataset-OpenML%2042890-orange)](https://www.openml.org/d/42890)
[![License: MIT](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

## What this is

An open, reproducible application of **Lean Six Sigma Master Black Belt methodology** to machine-learning model evaluation. Every "94% accurate" vendor claim is translated into the three numbers that actually matter for capital-intensive, regulated operations:

- **DPMO** (Defects Per Million Opportunities) — target ≤ 3.4
- **Sigma level** — target ≥ 6.0
- **Cpk** (process capability index) — target ≥ 2.0

A model that hits the 6σ bar is safe to deploy against safety-critical workloads. A model that doesn't — no matter how good its marketing looks — needs the **4-layer Operational Stability Infrastructure** (FMEA + Poka-Yoke + ML + capability monitoring) to close the gap.

## Dataset

This repository evaluates models against the **AI4I 2020 Predictive Maintenance Dataset** hosted on OpenML.

| Field | Value |
|---|---|
| **OpenML ID** | [42890](https://www.openml.org/d/42890) |
| **Direct download** | https://openml.org/data/v1/download/22045521/AI4I2020.arff |
| **Name on OpenML** | `AI4I2020` |
| **Description** | Synthetic dataset reflecting real predictive maintenance data in industry |
| **Format** | ARFF |
| **Instances** | 10,000 |
| **Features** | 6 (UDI, type, air temp, process temp, rotational speed, torque, tool wear) |
| **Target** | Binary + 5-mode failure type (TWF, HDF, PWF, OSF, RNF) |

### Citation

If you use this dataset, cite the original authors:

```bibtex
@misc{susto_ai4i_2020,
  title={AI4I 2020 Predictive Maintenance Dataset},
  author={Susto, Gian Antonio and Schirru, Andrea and Pampuri, Simone and McLoone, Seamas and Beghi, Alessandro},
  year={2020},
  howpublished={UCI Machine Learning Repository},
  note={DOI: 10.24432/C5BS66},
  url={https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset}
}

@misc{openml_ai4i_2020,
  title={AI4I2020 (OpenML Dataset 42890)},
  year={2020},
  url={https://www.openml.org/d/42890}
}
```

The dataset is referenced through [OpenML](https://www.openml.org), the open platform for sharing datasets, algorithms, and experiments.

## Methodology summary

The full capability math, control-chart pattern rules, leakage heuristics, and 4-layer framework specification live in the **`lean-six-sigma-expert`** consulting agent — not in this repository. This repo contains only the reproducible experiments.

**The one-line takeaway:** every claim of model accuracy in this domain must report mean ± std, sigma level, DPMO, AND Cpk together. Reporting only the point estimate (e.g., "94% accurate") is the most common vendor-evaluation failure mode.

Six Sigma capability targets (Motorola 1986 program, 1.5σ shift):

| Metric | 6σ target | Industry minimum |
|---|---|---|
| DPMO | ≤ 3.4 | ~66,800 (3.4σ) |
| Sigma level | ≥ 6.0 | ≥ 4.0 |
| Cpk | ≥ 2.0 | ≥ 1.0 |

## Files

| File | Purpose |
|---|---|
| `dpmo.py` | The core helper: computes mean, std, DPMO, sigma level, Cpk from any per-fold scores |
| `01-reproduce-credit-g.py` | Baseline: kNN on credit-g (Project 1) |
| `02-compare-3-algorithms-ai4i.py` | kNN vs RandomForest vs GradientBoosting on AI4I — **with target-leakage artefact** (Project 2) |
| `02b-compare-3-algorithms-ai4i-no-leakage.py` | Same comparison, leakage-clean — honest ceiling (Project 2b) |
| `03-kanungo-ple-aapl.py` | Reproducibility test: Pleias 6.7B on AAPL returns (sanity check on methodology) |
| `04-hdf-narrow-problem.py` | Narrowing the failure-mode target to heat dissipation failure only (Project 4) |
| `05-multi-class-ensemble.py` | Five specialist classifiers for the five failure modes (Project 7) |
| `06-poka-yoke-plus-ensemble.py` | Naive combination of Poka-Yoke rules + ML — **the killer finding** (Project 8) |
| `07-production-poka-yoke-plus-ensemble.py` | Production rule: deploy precise Poka-Yoke rules only (Project 7b) |
| `08-l4-capability-monitoring.py` | Per-chunk precision/recall with drift detection (Project 9) |
| `09-experiments-notebook.ipynb` | Interactive Jupyter walkthrough — **start here** |
| `2026-09-13 - 10-vendor-ml-claim-eval.py` | Vendor ML claim evaluation harness — applies the methodology to any HF model |
| `build_notebook.py` | Re-runnable notebook generator |
| `requirements.txt` | Pinned Python dependencies |
| `eval-results/` | Dated capability reports from vendor model evaluations |

## Empirical capstone

Project 7b (precise Poka-Yoke + ML fallback):

| Metric | Value | Verdict |
|---|---|---|
| Sigma level | **3.71σ** | Below 6σ target |
| DPMO | **~1,400** | Far below industry minimum but above 6σ target |
| **Cpk** | **2.22** | **First experiment to cross the 6σ Cpk threshold** |

Naive Poka-Yoke + ML (Project 8, **the killer finding**): sigma dropped to 2.75σ. **Deploying imprecise Poka-Yoke rules makes the system worse than ML alone.** This is the central IP of the 4-layer framework.

## Running the experiments

```bash
pip install -r requirements.txt
python dpmo.py                              # self-test: prints known Project 1+2 results
python "2026-09-13 - 10-vendor-ml-claim-eval.py"   # vendor ML claim evaluation harness
jupyter lab 09-experiments-notebook.ipynb     # interactive walkthrough
```

Requires `HF_TOKEN` environment variable for the vendor-claim harness (free HF account, fine-grained token with read access to public repos).

## License

Code in this repository: **MIT** — commercial-friendly reuse permitted.

The `AI4I 2020` dataset is provided by its authors under the terms specified at the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset) entry.

## Author

**Olivia Key** — Operational Consultant specialising in Lean Six Sigma + ML application for capital-intensive manufacturing and regulated operations.

The methodology is delivered through the **`lean-six-sigma-expert`** consulting agent (Mavis). Reproducible experiments like this one are the empirical evidence base for Six Sigma capability claims in client engagements.
