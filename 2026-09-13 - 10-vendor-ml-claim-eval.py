"""
2026-09-13 - 10-vendor-ml-claim-eval.py

Vendor ML Claim Evaluation - Starter Template
=============================================

Demonstrates Olivia's Six Sigma + ML evaluation methodology for vendor ML
claim evaluation in capital-intensive manufacturing consulting. This is the
smallest piece of code that exercises the full stack:

  1. Hugging Face model pull (with HF_TOKEN auth)
  2. Hugging Face datasets library for data loading
  3. Hugging Face evaluate library for classification metrics
  4. dpmo.py for Six Sigma capability reporting (sigma / DPMO / Cpk)

Same methodology Olivia uses for AI4I 2020 (Projects 1-9), applied here to
a vendor-style "our model is X% accurate" claim. Swap MODEL_ID and DATASET_ID
below to evaluate any vendor ML claim against the 6sigma bar.

Requirements:
  pip install "transformers[torch]" datasets evaluate
  HF_TOKEN set as environment variable (User-level, persistent)

Usage:
  python "2026-09-13 - 10-vendor-ml-claim-eval.py"
"""

# ============================================================================
# METADATA HEADER
# ============================================================================
# Created:  2026-09-13
# Updated:  2026-09-13
# Version:  1.0
# Status:   starter
# Related:  09 - OpenML Experimentation Log.md (vault)
#           lean-six-sigma-expert agent (Mavis) - methodology reference
# ============================================================================

import os
import sys
from datetime import datetime
from pathlib import Path

# UTF-8 stdout so sigma/mu render cleanly on Windows cp1252 consoles.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import numpy as np
from datasets import load_dataset
from transformers import pipeline

# dpmo.py is in the same directory as this script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dpmo import report_six_sigma

# ============================================================================
# CONFIGURATION - swap these for vendor ML claim evaluation
# ============================================================================
HF_TOKEN = os.environ.get("HF_TOKEN")
assert HF_TOKEN, (
    "HF_TOKEN not set. In PowerShell:\n"
    "  [Environment]::SetEnvironmentVariable('HF_TOKEN', '<your-token>', 'User')\n"
    "Then restart this script."
)

# Vendor ML claim to evaluate - replace with any HF classification model.
MODEL_ID = "bhadresh-savani/distilbert-base-uncased-emotion"

# Held-out evaluation data - replace with any HF dataset with 'text' and 'label'.
DATASET_ID = "dair-ai/emotion"

# Chunked evaluation: N chunks of the held-out set give a distribution of
# accuracy values, which feeds Six Sigma capability (sigma + DPMO + Cpk).
SAMPLE_SIZE = 200
N_CHUNKS = 5

# Random-guess baseline = 1 / number_of_classes. The 6-class emotion dataset
# has random baseline ~16.67%. Adjust when swapping in a different dataset.
RANDOM_BASELINE = 1.0 / 6.0  # = 0.1667

# ============================================================================
# 1. LOAD MODEL
# ============================================================================
print(f"[1/4] Loading model: {MODEL_ID}")
classifier = pipeline(
    task="text-classification",
    model=MODEL_ID,
    token=HF_TOKEN,
    device=-1,  # CPU; change to 0 for GPU
    truncation=True,
    max_length=512,
)

# Build the model's string -> int label mapping from its config so we do not
# hardcode label names. This makes the script work for any HF classification
# model that exposes id2label in its config.
model_id2label = classifier.model.config.id2label  # e.g. {0: 'sadness', 1: 'joy', ...}
str_to_int = {v: int(k) for k, v in model_id2label.items()}
print(f"   Model label map: {model_id2label}")

# ============================================================================
# 2. LOAD DATASET
# ============================================================================
print(f"[2/4] Loading {SAMPLE_SIZE} samples from {DATASET_ID} (test split)")
dataset = load_dataset(
    DATASET_ID,
    split=f"test[:{SAMPLE_SIZE}]",
    token=HF_TOKEN,
)
texts = list(dataset["text"])
true_labels = list(dataset["label"])
print(f"   Loaded {len(texts)} samples. Class distribution: "
      f"{dict(zip(*np.unique(true_labels, return_counts=True)))}")

# ============================================================================
# 3. RUN PREDICTIONS - chunked for sigma distribution
# ============================================================================
# Chunking produces a distribution of accuracy values (one per chunk) instead
# of a single point estimate. This is what feeds the 6sigma capability math:
# std of the distribution drives Cpk; mean drives DPMO + sigma level.
print(f"[3/4] Running predictions in {N_CHUNKS} chunks...")
chunk_size = len(texts) // N_CHUNKS
chunk_accuracies = []

for chunk_idx in range(N_CHUNKS):
    start = chunk_idx * chunk_size
    end = start + chunk_size if chunk_idx < N_CHUNKS - 1 else len(texts)

    chunk_texts = texts[start:end]
    chunk_labels = true_labels[start:end]

    preds = classifier(chunk_texts)
    pred_labels = [str_to_int[p["label"]] for p in preds]
    accuracy = sum(p == t for p, t in zip(pred_labels, chunk_labels)) / len(chunk_labels)
    chunk_accuracies.append(accuracy)
    print(f"   Chunk {chunk_idx + 1}/{N_CHUNKS}: accuracy = {accuracy * 100:6.2f}%")

# ============================================================================
# 4. APPLY THE 6sigma BAR
# ============================================================================
print()
print(f"[4/4] Computing Six Sigma capability from {N_CHUNKS} chunk accuracies...")

metrics = report_six_sigma(
    chunk_accuracies,
    model_name=MODEL_ID,
    dataset_name=(
        f"{DATASET_ID} (test split, {SAMPLE_SIZE} samples, "
        f"{N_CHUNKS} chunks of {chunk_size})"
    ),
    lower_spec_limit=RANDOM_BASELINE,
)

# Final verdict
print()
print("=" * 70)
print("VERDICT")
print("=" * 70)
pass_sigma = metrics["sigma_level"] >= 6.0
pass_dpmo = metrics["dpmo"] <= 3.4
pass_cpk = metrics["cpk"] >= 2.0
overall = pass_sigma and pass_dpmo and pass_cpk

print(f"  Sigma level:  {metrics['sigma_level']:.2f}sigma "
      f"(target: >= 6.00)  {'PASS' if pass_sigma else 'FAIL'}")
print(f"  DPMO:         {metrics['dpmo']:>10,.0f} "
      f"(target: <= 3.4)  {'PASS' if pass_dpmo else 'FAIL'}")
print(f"  Cpk:          {metrics['cpk']:>6.2f} "
      f"(target: >= 2.00)  {'PASS' if pass_cpk else 'FAIL'}")
print()
if overall:
    print("  Result: 6sigma-capable. This ML model meets the bar.")
else:
    print("  Result: Below 6sigma. Apply the 4-layer Operational Stability")
    print("          Infrastructure (FMEA + Poka-Yoke + ML + capability monitoring)")
    print("          to close the gap. Do NOT accept below-6sigma as 'good enough'.")
print("=" * 70)

# ============================================================================
# 5. PERSIST REPORT TO FILE (consulting pitch artefact)
# ============================================================================
# Every run writes a dated results file next to this script. This is the
# artefact to attach to client proposals, cite in capability reviews, or
# archive in the vault for traceability.
results_dir = Path(__file__).resolve().parent / "eval-results"
results_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
safe_model = MODEL_ID.replace("/", "_").replace("\\", "_")
safe_dataset = DATASET_ID.replace("/", "_").replace("\\", "_")
results_file = results_dir / (
    f"{timestamp}_{safe_model}_on_{safe_dataset}_n{SAMPLE_SIZE}_k{N_CHUNKS}.txt"
)

with open(results_file, "w", encoding="utf-8") as f:
    f.write("VENDOR ML CLAIM EVALUATION REPORT\n")
    f.write("=" * 70 + "\n")
    f.write(f"Generated:      {datetime.now().isoformat(timespec='seconds')}\n")
    f.write(f"Model:          {MODEL_ID}\n")
    f.write(f"Dataset:        {DATASET_ID}\n")
    f.write(f"Sample size:    {SAMPLE_SIZE}\n")
    f.write(f"Chunks:         {N_CHUNKS}\n")
    f.write(f"Random baseline:{RANDOM_BASELINE:.4f}\n")
    f.write("-" * 70 + "\n")
    f.write("Chunk accuracies (raw distribution):\n")
    for i, acc in enumerate(chunk_accuracies, start=1):
        f.write(f"  Chunk {i}/{N_CHUNKS}: {acc * 100:6.2f}%\n")
    f.write("-" * 70 + "\n")
    f.write("Six Sigma capability report:\n")
    f.write(f"  Mean accuracy:  {metrics['mean'] * 100:>7.2f}%\n")
    f.write(f"  Std deviation:  +/-{metrics['std'] * 100:>6.2f}%\n")
    f.write(f"  DPMO:           {metrics['dpmo']:>10,.0f}\n")
    f.write(f"  Sigma level:    {metrics['sigma_level']:>6.2f}sigma\n")
    f.write(f"  Cpk:            {metrics['cpk']:>6.2f}\n")
    f.write(f"  n_folds:        {metrics['n_folds']}\n")
    f.write("-" * 70 + "\n")
    f.write("Verdict (6sigma bar):\n")
    f.write(f"  Sigma >= 6.00:  {'PASS' if pass_sigma else 'FAIL'}\n")
    f.write(f"  DPMO  <= 3.4:   {'PASS' if pass_dpmo else 'FAIL'}\n")
    f.write(f"  Cpk   >= 2.00:  {'PASS' if pass_cpk else 'FAIL'}\n")
    f.write(f"  Overall:        {'6sigma-capable' if overall else 'Below 6sigma'}\n")
    f.write("=" * 70 + "\n")
    f.write("\nMethodology reference: lean-six-sigma-expert agent (Mavis)\n")
    f.write("Six Sigma bar: 6sigma = 3.4 DPMO = Cpk >= 2.0 (Motorola 1.5sigma shift)\n")

print()
print(f"Report saved to: {results_file}")
