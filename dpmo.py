"""
dpmo.py - Six Sigma / DPMO / Cpk reporting for ML model evaluation.

Computes five quality metrics from any cross_val_score output:
  - Mean accuracy
  - Standard deviation
  - DPMO (defects per million opportunities)
  - Sigma level (with classic 1.5σ shift)
  - Cpk (process capability index)

The math is standard Six Sigma - same DPMO and Cpk formulas from
the Motorola 1986 Six Sigma program. The novelty is applying them
to ML model evaluation. Use this anywhere a "model accuracy" claim
needs to be translated into a quality-process statement.

Usage:
    from dpmo import report_six_sigma, report_comparison

    # Single model
    scores = cross_val_score(pipe, X, y, cv=10, scoring="accuracy")
    metrics = report_six_sigma(
        scores,
        model_name="Random Forest (100 trees)",
        dataset_name="AI4I 2020",
        lower_spec_limit=0.9661,    # majority-class baseline for imbalanced
        target_leakage_warning=True,
    )

    # Multi-model comparison
    results = {
        "kNN (k=5)": knn_scores,
        "Random Forest (100 trees)": rf_scores,
        "Gradient Boosting (100 trees)": gb_scores,
    }
    all_metrics = report_comparison(
        results,
        dataset_name="AI4I 2020 Predictive Maintenance",
        lower_spec_limit=0.9661,
    )
"""

import sys

# Make sure Unicode (σ, μ, σ², etc.) prints cleanly on Windows cp1252 consoles.
# No-op on Linux/macOS terminals that already default to UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass  # older Python or already reconfigured elsewhere

import numpy as np
from scipy import stats


# Six Sigma standard (the Olivia standard)
SIGMA_TARGET = 6.0      # 6σ = 3.4 DPMO = 99.9997% accuracy
CPK_TARGET = 2.0        # 6σ process capability index
SIGMA_SHIFT = 1.5       # classic Six Sigma 1.5σ drift


def compute_six_sigma(scores, lower_spec_limit=0.0):
    """
    Compute mean, std, DPMO, sigma level, and Cpk from cross-validation scores.

    Parameters
    ----------
    scores : array-like
        Accuracy scores from cross_val_score (or any per-fold metric in [0, 1]).
    lower_spec_limit : float
        Lower specification limit (default 0.0 = perfect non-prediction).
        For imbalanced data, set to the majority-class baseline so Cpk
        measures the gap from "doing nothing."

    Returns
    -------
    dict with keys: mean, std, dpmo, sigma_level, cpk, n_folds
    """
    scores = np.asarray(scores, dtype=float)
    mean_acc = float(scores.mean())
    std_acc = float(scores.std())
    n_folds = len(scores)

    # DPMO: defects per million opportunities (1 opportunity per prediction for binary)
    dpmo = max(0.0, (1.0 - mean_acc) * 1_000_000.0)

    # Sigma level with classic 1.5σ shift
    if mean_acc >= 1.0:
        sigma_level = float("inf")
    elif mean_acc <= 0.0:
        sigma_level = float("-inf")
    else:
        # Clamp the probability to avoid ±inf from ppf at the tails
        p = max(min(mean_acc, 1.0 - 1e-12), 1e-12)
        sigma_level = float(stats.norm.ppf(p) + SIGMA_SHIFT)

    # Cpk = min((USL - mean) / (3*std), (mean - LSL) / (3*std))
    usl = 1.0
    lsl = float(lower_spec_limit)
    if std_acc > 0.0:
        cpk_upper = (usl - mean_acc) / (3.0 * std_acc)
        cpk_lower = (mean_acc - lsl) / (3.0 * std_acc)
        cpk = float(min(cpk_upper, cpk_lower))
    else:
        cpk = float("inf")

    return {
        "mean": mean_acc,
        "std": std_acc,
        "dpmo": dpmo,
        "sigma_level": sigma_level,
        "cpk": cpk,
        "n_folds": n_folds,
    }


def _auto_detect_leakage(metrics):
    """
    Heuristic: suspiciously high mean + very low std is a target-leakage signature.

    Returns (is_suspicious: bool, reason: str).
    """
    if metrics["mean"] >= 0.95 and metrics["std"] <= 0.005 and metrics["n_folds"] >= 5:
        return True, (
            "auto-detected: mean >= 95% with std <= 0.5% across 5+ folds. "
            "This pattern is a target-leakage signature. Audit the feature "
            "list for columns that are direct causes of the target."
        )
    return False, ""


def report_six_sigma(
    scores,
    model_name="Model",
    dataset_name="Dataset",
    lower_spec_limit=0.0,
    target_leakage_warning=False,
    auto_detect_leakage=True,
):
    """
    Print a Six Sigma quality report for a model's cross-validation scores.

    Parameters
    ----------
    scores : array-like
        Accuracy scores from cross_val_score.
    model_name : str
        Display name for the model.
    dataset_name : str
        Display name for the dataset.
    lower_spec_limit : float
        Lower spec limit (default 0.0). Set to majority-class baseline for imbalanced.
    target_leakage_warning : bool
        If True, prints an explicit warning to scrutinize the feature list.
    auto_detect_leakage : bool
        If True, runs a heuristic check for target-leakage signature.

    Returns
    -------
    dict with the computed metrics.
    """
    metrics = compute_six_sigma(scores, lower_spec_limit=lower_spec_limit)

    print("=" * 70)
    print(f"Model:   {model_name}")
    print(f"Dataset: {dataset_name}")
    print("=" * 70)
    print(f"  Mean accuracy:        {metrics['mean'] * 100:>7.2f}%")
    print(f"  Std deviation:        +/- {metrics['std'] * 100:>6.2f}%")
    print(f"  DPMO:                 {metrics['dpmo']:>10,.0f}")
    print(f"  Sigma level:          {metrics['sigma_level']:>6.2f}σ")
    print(f"  Cpk:                  {metrics['cpk']:>6.2f}")
    print()

    # Six Sigma verdict
    gap = SIGMA_TARGET - metrics["sigma_level"]
    if metrics["sigma_level"] >= SIGMA_TARGET:
        verdict = f"PASS - {metrics['sigma_level']:.2f}σ meets the 6σ bar"
    else:
        verdict = (
            f"BELOW 6σ - gap of {gap:.2f}σ from bar. "
            f"Communicate honestly with residual DPMO and FMEA."
        )
    print(f"  Six Sigma verdict:    {verdict}")
    print()

    # Flags
    flags = []
    if target_leakage_warning:
        flags.append(
            "Manual target-leakage flag set. Audit the feature list for "
            "columns that are direct causes of the target."
        )

    if auto_detect_leakage:
        is_suspicious, reason = _auto_detect_leakage(metrics)
        if is_suspicious:
            flags.append(reason)

    if metrics["mean"] < 0.5:
        flags.append(
            f"Mean accuracy ({metrics['mean'] * 100:.2f}%) is WORSE THAN RANDOM "
            f"on a balanced binary task. Model is broken."
        )

    if metrics["std"] > 0.10:
        flags.append(
            f"Std ({metrics['std'] * 100:.2f}%) is high - model is unreliable "
            f"across data partitions. Production-unworkable."
        )

    if metrics["cpk"] < 1.0 and metrics["std"] > 0:
        flags.append(
            f"Cpk ({metrics['cpk']:.2f}) is below 1.0 - process capability is "
            f"below industry minimum."
        )

    if flags:
        print("  Flags:")
        for flag in flags:
            print(f"    - {flag}")
        print()

    print("=" * 70)
    return metrics


def report_comparison(results, dataset_name="Dataset", lower_spec_limit=0.0):
    """
    Print a Six Sigma comparison table for multiple models.

    Parameters
    ----------
    results : dict[str, array-like]
        Mapping of model_name -> cross_val_score output.
    dataset_name : str
        Display name for the dataset.
    lower_spec_limit : float
        Lower spec limit passed through to each model's Cpk calculation.

    Returns
    -------
    dict[str, dict] mapping model_name -> metrics dict.
    """
    all_metrics = {}

    print("=" * 70)
    print(f"Six Sigma comparison: {dataset_name}")
    print("=" * 70)
    print(
        f"  {'Model':<32} {'Mean':>9} {'Std':>9} "
        f"{'DPMO':>12} {'Sigma':>8} {'Cpk':>7}  Verdict"
    )
    print(
        f"  {'-' * 32} {'-' * 9} {'-' * 9} "
        f"{'-' * 12} {'-' * 8} {'-' * 7}  {'-' * 12}"
    )

    for name, scores in results.items():
        m = compute_six_sigma(scores, lower_spec_limit=lower_spec_limit)
        all_metrics[name] = m

        if m["sigma_level"] >= SIGMA_TARGET:
            verdict = "6σ OK"
        else:
            gap = SIGMA_TARGET - m["sigma_level"]
            verdict = f"-{gap:.1f}σ from bar"

        print(
            f"  {name:<32} "
            f"{m['mean'] * 100:>8.2f}% "
            f"{m['std'] * 100:>+8.2f}% "
            f"{m['dpmo']:>12,.0f} "
            f"{m['sigma_level']:>7.2f}σ "
            f"{m['cpk']:>7.2f}  "
            f"{verdict}"
        )

    print("=" * 70)
    return all_metrics


# ---------------------------------------------------------------------------
# Self-test: known values from Projects 1 and 2
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print()
    print("Self-test: known values from Project 1 (credit-g) and Project 2 (AI4I)")
    print()

    # Project 1: kNN credit-g, 10 folds
    project1_knn = [
        0.6900, 0.6500, 0.6500, 0.7000, 0.6400,
        0.6200, 0.6400, 0.6600, 0.6500, 0.6500,
    ]

    # Project 2: kNN AI4I, 10 folds
    project2_knn = [
        0.9670, 0.6150, 0.4760, 0.3960, 0.3720,
        0.4090, 0.3860, 0.3940, 0.4100, 0.9660,
    ]

    # Project 2: Random Forest with target leakage
    project2_rf_leakage = [
        1.0000, 0.9990, 0.9990, 0.9990, 1.0000,
        1.0000, 0.9990, 0.9970, 1.0000, 0.9980,
    ]

    # Project 2: Gradient Boosting with leakage
    project2_gb_leakage = [
        0.9880, 0.9970, 0.9990, 0.9990, 0.9990,
        0.9990, 0.9980, 0.9970, 0.9820, 0.9980,
    ]

    # Single-model report: kNN credit-g
    report_six_sigma(
        project1_knn,
        model_name="kNN (k=5) on credit-g",
        dataset_name="credit-g (OpenML ID 31)",
    )

    # Single-model report: kNN AI4I
    report_six_sigma(
        project2_knn,
        model_name="kNN (k=5) on AI4I 2020",
        dataset_name="AI4I 2020 Predictive Maintenance",
        lower_spec_limit=0.9661,
    )

    # Multi-model comparison: AI4I 2020 (with leakage)
    print()
    results = {
        "kNN (k=5)": project2_knn,
        "Random Forest (100 trees)": project2_rf_leakage,
        "Gradient Boosting (100 trees)": project2_gb_leakage,
    }
    report_comparison(
        results,
        dataset_name="AI4I 2020 Predictive Maintenance (with target leakage)",
        lower_spec_limit=0.9661,
    )

    # Single-model with explicit leakage flag
    report_six_sigma(
        project2_rf_leakage,
        model_name="Random Forest (100 trees)",
        dataset_name="AI4I 2020 Predictive Maintenance",
        lower_spec_limit=0.9661,
        target_leakage_warning=True,
    )

    print()
    print("Self-test complete.")
    print()
    print("Next: import this module in your next experiment script:")
    print("    from dpmo import report_six_sigma, report_comparison")