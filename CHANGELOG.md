# Changelog — Six Sigma + ML Evaluation Suite

Chronological record of every timestamped change to this repository, with rationale and empirical outcomes. Read this top-to-bottom to understand how the methodology evolved.

---

## 2026-09-13 (16:23 UTC) — Repository initialised on GitHub

**What changed**
- Created public repo `oliviakey-oliviasmuse/OpenML-Vendor-Claim-Eval-Report`
- README with full OpenML citation: [AI4I 2020 Predictive Maintenance Dataset, OpenML ID 42890](https://www.openml.org/d/42890)
- LICENSE (MIT, copyright Olivia Key)
- `.gitignore` (excludes HF cache, `.env`, `__pycache__/`, eval-results raw model checkpoints)
- Single commit `545c359` containing 19 files

**Why**
Public, citable, reproducible empirical evidence base for Six Sigma capability claims in capital-intensive manufacturing consulting. Real data only — no composite scenarios.

**Deliberately excluded**
- `lean-six-sigma-expert.md` — Mavis agent content. Private methodology IP (4-layer framework details, killer finding, capability math derivations). Lives in `C:\Users\olivi\.minimax\agents\agent-6e796b74c7fe\agent.md` and as a workspace reference copy in the openml-experiments folder.
- `lean-six-sigma-skill.md` — failed Foundry skill attempt. Junk.

---

## 2026-09-13 (~16:00 UTC) — First vendor ML claim evaluation run

**New file:** `2026-09-13 - 10-vendor-ml-claim-eval.py`

**What it does**
Pulls any HF model via `transformers`, runs it on any HF dataset via `datasets`, computes accuracy + macro F1 + precision + recall via `evaluate`, applies the Six Sigma bar via `dpmo.py`, persists a dated capability report to `eval-results/`.

**First run: distilbert-base-uncased-emotion on dair-ai/emotion**

| Metric | Vendor framing | Six Sigma framing | 6σ target |
|---|---|---|---|
| Mean accuracy | 93.00% | — | — |
| Std deviation | — | ±3.32% | — |
| DPMO | — | **70,000** | ≤ 3.4 |
| Sigma level | — | **2.98σ** | ≥ 6.0 |
| Cpk | — | **0.70** | ≥ 2.0 |

**Verdict:** BELOW 6σ on all three metrics. Cpk is below industry minimum (1.0).

**Persisted to:** `eval-results/2026-09-13_160116_bhadresh-savani_distilbert-base-uncased-emotion_on_dair-ai_emotion_n200_k5.txt`

**Why it matters**
The methodology in action. A vendor model with 93% accuracy sounds impressive in marketing terms; in Six Sigma terms it sits below industry minimum. **This is the consulting pitch output** — when a vendor claims "94% accurate," this script produces the real capability report in under a minute.

---

## 2026-09-12 — Code review pass + notebook fixes

**Files modified (post-review):**
- `04-hdf-narrow-problem.py` — correctness: docstring claimed "20 chunks", actual is 10
- `05-multi-class-ensemble.py` — correctness: broken `{}` baseline output; dead code: `specialist_estimators` never used
- `06-poka-yoke-plus-ensemble.py` — dead code: `PRECISE_POKA_YOKE_MODES` unused; UTF-8 BOM
- `07-production-poka-yoke-plus-ensemble.py` — dead code: `confusion_matrix` import unused
- `08-l4-capability-monitoring.py` — dead code: `DRIFT_INJECTION_CHUNK = 10` never referenced; docstring said "20 chunks"
- `09-experiments-notebook.ipynb` — three substantive fixes:
  - **Cells 24-25:** restored Project 8 (naive Poka-Yoke + ML = WORSE than ML alone, σ delta −0.86, F1 delta −3.80pp). This is the killer finding.
  - **L4 rare-mode noise:** skip alert when actual count per mode per chunk = 0; show "n/a" in monitoring table
  - **Auto-leakage flag surface:** explicit `report_six_sigma` call in cell 31 makes the heuristic check visible in the report

**Why**
The /code-review skill found 2 real correctness bugs, 6 dead-code items, 3 efficiency issues, 2 cleanup items. All re-run after the pass — identical numerical results, no regressions. Methodology stayed consistent; only code hygiene improved.

---

## 2026-09-10 — Original experimental work (Projects 1-9)

This is where everything started. All files dated 2026-09-10 in their vault copies; the working copies in this repo are date-stripped for clarity.

### Cumulative sigma story (the empirical capstone)

| Project | What | Sigma | Cpk | Notes |
|---|---|---|---|---|
| 1 | kNN on credit-g | 1.9σ | 5.12 | Imbalanced binary, broad target. The bar is hard. |
| 2 | kNN/RF/GB on AI4I (with leakage) | up to 4.62σ | — | **Target leakage artefact.** 95%+ accuracy is the symptom, not the result. |
| 2b | AI4I leakage-clean | 3.46σ | — | Honest baseline ceiling. |
| 4 | HDF only (narrow scope) | 3.99σ | — | Narrowing helps, but caps. |
| 7 | 5-specialist ensemble (multi-class) | 3.62σ top-1 | 1.96 | Best broad multi-class result before production rules. |
| **8** | **Naive Poka-Yoke + ML** | **2.75σ** | — | **KILLER FINDING.** Naive combination is WORSE than ML alone. σ delta −0.86, F1 delta −3.80pp. |
| **7b** | **Precise Poka-Yoke + ML (production)** | **3.71σ top-1** | **2.22** | **First experiment to cross the 6σ Cpk threshold.** PWF + OSF precise rules deployed; TWF + HDF excluded (over-firing). |
| 9 | L4 capability monitoring | n/a | n/a | Drift caught at chunk 5; 0 chunks latency. Rare-mode noise handled. |

### The two findings the consulting pitch is built on

1. **Naive Poka-Yoke + ML breaks the system** (Project 8). Imprecise rules over-fire and corrupt ML predictions. The fix is empirical precision/recall verification: rules must hit 100% precision AND 100% recall on historical data before deployment.

2. **The 4-layer framework closes the gap to 6σ** (Project 7b). L1 FMEA identifies deterministic vs non-deterministic failure modes. L2 Poka-Yoke handles deterministic (precise rules only). L3 ML handles non-deterministic residual. L4 monitors all three with per-chunk precision/recall + drift detection.

### Files in this batch

| File | Purpose |
|---|---|
| `01-reproduce-credit-g-knn.py` | kNN baseline on credit-g (Project 1) |
| `01-reproduce-credit-g-knn-sklearn.py` | scikit-learn equivalent of the same (for benchmarking) |
| `02-compare-3-algorithms-ai4i.py` | kNN vs RF vs GB on AI4I — **with target-leakage artefact** (Project 2) |
| `02b-compare-3-algorithms-ai4i-no-leakage.py` | Same comparison, leakage-clean baseline (Project 2b) |
| `03-kanungo-ple-aapl.py` | Methodology sanity check: Pleias 6.7B on AAPL returns — confirms reproducible pipeline |
| `04-hdf-narrow-problem.py` | Narrow target to Heat Dissipation Failure only (Project 4) |
| `05-multi-class-ensemble.py` | Five specialist classifiers, one per failure mode (Project 7) |
| `06-poka-yoke-plus-ensemble.py` | Naive Poka-Yoke + ML combination (Project 8 — the killer finding) |
| `07-production-poka-yoke-plus-ensemble.py` | Production deployment with precise Poka-Yoke rules only (Project 7b) |
| `08-l4-capability-monitoring.py` | Per-chunk precision/recall with drift detection (Project 9) |
| `09-experiments-notebook.ipynb` | Interactive walkthrough — **start here** |
| `build_notebook.py` | Re-runnable notebook generator (recreates 09 from the .py files) |
| `dpmo.py` | Core helper: mean, std, DPMO, sigma level, Cpk from any per-fold scores |
| `requirements.txt` | Pinned Python dependencies |

---

## Methodology reference (not in this repo)

The full Six Sigma + ML methodology, FMEA templates, control chart pattern rules (Western Electric + Nelson sets), capability math derivations, AI agent evaluation rules, and multi-agent architecture guidance live in the **`lean-six-sigma-expert`** consulting agent (Mavis). Referenced in the README; not exposed publicly. The agent is the consulting deliverable brain; this repo is the reproducible evidence base.

---

## How to use this changelog

- **For client pitches:** the cumulative sigma table is the slide. Add the distilbert vendor report next to it for the "before/after" comparison.
- **For methodology audits:** read top-to-bottom. Each milestone explains *why* a change was made, not just *what*.
- **For reproduction:** each file's date in the commit history tells you which version to run.
