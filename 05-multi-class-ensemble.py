"""
Project 7 - Multi-class across 5 failure modes (TWF, HDF, PWF, OSF, RNF)
            vs ensemble of 5 narrow binary specialists.

Background (from the experimentation log):
    Project 4 (narrow problem definition, HDF only) showed that narrowing
    the problem lifts sigma from 3.4sigma (Project 2b broad) to 3.99sigma
    for Gradient Boosting. But that's still 2sigma short of 6sigma.

    This project tests whether an ENSEMBLE of narrow specialists closes
    the remaining gap. The hypothesis:
        - A single multi-class model has to discriminate all 5 failure
          modes AND the "no failure" class simultaneously. That's 6
          classes with extreme imbalance (96.61% no failure, 0.23-1.15%
          for each specific mode).
        - 5 binary specialists, each tuned to one failure mode's signature,
          can be aggregated via argmax to give better mode-specific
          precision. Each specialist only has to beat "everyone is no
          failure," which is its own baseline.

Hypothesis:
    Ensemble of 5 binary specialists > single multi-class classifier,
    measured by macro F1 (better for imbalanced multi-class) and
    translated to Six Sigma via top-1 accuracy.

The 5 failure modes in AI4I 2020:
    TWF = Tool Wear Failure
    HDF = Heat Dissipation Failure
    PWF = Power Failure
    OSF = Overstrain Failure
    RNF = Random Failure

    Plus a 6th effective class: "no failure" (all 5 modes = 0).

Setup:
    - Dataset: AI4I 2020 (OpenML id 42890), 10,000 rows
    - Honest feature set (6): Type, Air temperature [K], Process temperature [K],
      Rotational speed [rpm], Torque [Nm], Tool wear [min]
    - Dropped columns: UDI, Product ID, Machine failure, TWF, HDF, PWF, OSF, RNF
      (all 5 failure mode columns are direct causes of `Machine failure`,
       so keeping them in the feature list = target leakage for any mode
       we'd try to predict)
    - 10-fold stratified CV

Strategies compared:
    A. Multi-class Gradient Boosting (single model, 6 classes)
    B. Multi-class Random Forest (single model, 6 classes)
    C. Ensemble of 5 binary Gradient Boosting specialists + argmax aggregation
    D. Majority-class baseline (predict "no failure" always)

Run:
    python 05-multi-class-ensemble.py
"""

import openml
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from dpmo import SIGMA_SHIFT, report_comparison


# The 5 specific failure modes. All 5 must be dropped from features for
# every specialist to prevent cross-mode target leakage.
FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]


def banner(text):
    print()
    print("=" * 70)
    print(text)
    print("=" * 70)


# ---------------------------------------------------------------------------
# Step 1 - Locate AI4I 2020 on OpenML
# ---------------------------------------------------------------------------
banner("STEP 1: Locate AI4I 2020 Predictive Maintenance on OpenML")

candidate_names = [
    "AI4I_2020_Predictive_Maintenance",
    "ai4i_2020",
    "AI4I 2020 Predictive Maintenance",
    "Predictive_Maintenance",
    "predictive_maintenance",
]

dataset = None
for name in candidate_names:
    try:
        ds = openml.datasets.get_dataset(
            name,
            download_data=False,
            download_qualities=False,
            download_features_meta_data=False,
        )
        if ds is not None:
            dataset = ds
            print(f"Found by name '{name}': {dataset.name} (id={dataset.dataset_id})")
            break
    except Exception:
        continue

if dataset is None:
    print("Direct name lookup failed; searching OpenML index...")
    all_datasets = openml.datasets.list_datasets(output_format="dataframe")
    ai4i = all_datasets[
        all_datasets["name"].str.contains(
            "ai4i|predictive.*maintenance",
            case=False,
            regex=True,
            na=False,
        )
    ]
    if not ai4i.empty:
        first_id = int(ai4i.iloc[0]["did"])
        dataset = openml.datasets.get_dataset(first_id)
        print(f"Found via search: {dataset.name} (id={dataset.dataset_id})")
    else:
        raise ValueError(
            "Could not locate AI4I 2020 on OpenML. "
            "Run: openml.datasets.list_datasets(output_format='dataframe') "
            "and search manually."
        )


# ---------------------------------------------------------------------------
# Step 2 - Fetch data
# ---------------------------------------------------------------------------
banner("STEP 2: Fetch raw data, build multi-class target + binary specialists")

# Fetch with Machine failure as a placeholder target so we get all columns
# We need access to the 5 mode columns to build our own multi-class label.
X_raw, _, _, _ = dataset.get_data(target="Machine failure")
print(f"Raw shape: {X_raw.shape}")

# Drop identifier columns: both string columns (high cardinality) and
# integer ID columns (UDI). The 100+ unique string check catches Product ID.
# UDI is an integer with 10,000 unique values - drop it explicitly.
identifier_cols = [
    col for col in X_raw.columns
    if pd.api.types.is_string_dtype(X_raw[col]) and X_raw[col].nunique() > 100
]
if "UDI" in X_raw.columns:
    identifier_cols.append("UDI")
if identifier_cols:
    print(f"\nDropping identifier columns: {identifier_cols}")
    X_raw = X_raw.drop(columns=identifier_cols)

# Verify all 5 failure mode columns exist
missing_modes = [m for m in FAILURE_MODES if m not in X_raw.columns]
if missing_modes:
    raise ValueError(
        f"Missing expected failure mode columns: {missing_modes}. "
        f"Available columns: {list(X_raw.columns)}"
    )


# ---------------------------------------------------------------------------
# Step 2.5 - Build the multi-class target + 5 binary specialist targets
# ---------------------------------------------------------------------------
banner("STEP 2.5: Build targets (multi-class + 5 binary)")

# Multi-class label: which single mode is firing (or "none")
# Rows with at least one mode = 1: assign to the FIRST mode in FAILURE_MODES order.
# Rows with no modes = "none".
def build_multiclass_target(row):
    for mode in FAILURE_MODES:
        if row[mode] == 1:
            return mode
    return "none"


y_multiclass = X_raw.apply(build_multiclass_target, axis=1)
print(f"Multi-class target distribution:")
print(y_multiclass.value_counts().sort_index().to_string())
print(f"\nClass imbalance ratio (no failure vs rarest mode): "
      f"{(y_multiclass == 'none').sum() / max(1, min((y_multiclass != 'none').sum(), 1)):.1f}x")


# ---------------------------------------------------------------------------
# Step 2.6 - Build the honest feature set (drop ALL leakage columns)
# ---------------------------------------------------------------------------
banner("STEP 2.6: Build honest feature set (drop ALL leakage columns)")

# All 5 failure modes + Machine failure must be dropped from features to
# prevent leakage. Otherwise the multi-class model would just read the
# answer key (the mode columns ARE the targets).
leakage_cols = FAILURE_MODES + ["Machine failure"]
present = [c for c in leakage_cols if c in X_raw.columns]
print(f"Dropping leakage columns: {present}")
print("  Reason: TWF/HDF/PWF/OSF/RNF are direct causes of Machine failure.")
print("          Keeping them in features = reading the answer key.")

X = X_raw.drop(columns=present)
print(f"\nFinal feature set: {list(X.columns)}")
print(f"Final shape: X={X.shape}")


# ---------------------------------------------------------------------------
# Step 2.7 - Detect categorical vs numeric
# ---------------------------------------------------------------------------
banner("STEP 2.7: Re-detect categorical indicator")

categorical_indicator = [pd.api.types.is_string_dtype(X[col]) for col in X.columns]
n_cat = sum(categorical_indicator)
n_num = len(categorical_indicator) - n_cat

for i, (col, is_cat) in enumerate(zip(X.columns, categorical_indicator)):
    marker = "CAT" if is_cat else "NUM"
    print(f"  [{i:2d}] {marker}  {col}")
print(f"\nFinal: {n_cat} categorical, {n_num} numerical.")


# ---------------------------------------------------------------------------
# Step 3 - Build shared preprocessor
# ---------------------------------------------------------------------------
banner("STEP 3: Build shared preprocessor")

cat_indices = [i for i, is_cat in enumerate(categorical_indicator) if is_cat]
num_indices = [i for i, is_cat in enumerate(categorical_indicator) if not is_cat]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_indices),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_indices),
    ]
)
print(f"Passthrough {n_num} numerical columns; one-hot encode {n_cat} categorical columns.")


# ---------------------------------------------------------------------------
# Step 4 - Build pipelines for both strategies
# ---------------------------------------------------------------------------
banner("STEP 4: Build pipelines for multi-class and binary-specialist strategies")

# --- Strategy A + B: native multi-class ---
multiclass_pipelines = {
    "Multi-class Gradient Boosting": Pipeline(
        [
            ("preprocess", preprocessor),
            ("clf", GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ]
    ),
    "Multi-class Random Forest": Pipeline(
        [
            ("preprocess", preprocessor),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ]
    ),
}

print("\nMulti-class strategies (single model, 6 classes):")
for name in multiclass_pipelines:
    print(f"  - {name}")


# --- Strategy C: 5 binary specialists + argmax aggregation ---
# Each specialist is a binary classifier for one mode vs everything else.
# We'll use the predict_proba to get a probability per mode, then argmax
# across the 5 + "no failure" (where no failure = 1 - max(prob_failure)).
#
# Implementation note: we use sklearn's MultiOutputClassifier wrapping
# 5 binary GradientBoostingClassifier instances. The predict_proba gives
# us 5 sets of probabilities. Argmax across [P(TWF), P(HDF), P(PWF), P(OSF), P(RNF)]
# chooses the most-likely failure mode; if all 5 are below 0.5, predict "none".
print("\nBinary-specialist strategy (5 specialists + argmax):")
print("  - 5 GradientBoostingClassifier instances, one per failure mode")
print("  - Aggregation: argmax over 5 specialist probabilities")
print("    if max < 0.5 -> predict 'none' (no failure)")

# The ensemble is constructed fresh per fold inside the cross-validation loop
# below (sklearn's MultiOutputClassifier doesn't share cross_val_score cleanly
# with custom aggregation).


# ---------------------------------------------------------------------------
# Step 5 - Build the 5 binary specialist target matrix
# ---------------------------------------------------------------------------
banner("STEP 5: Build 5 binary specialist target matrix")

Y_specialists = pd.DataFrame(
    {mode: X_raw[mode].astype(int).values for mode in FAILURE_MODES},
    index=X.index,
)
print(f"Specialist target matrix shape: {Y_specialists.shape}")
print("Per-mode positive rate:")
for mode in FAILURE_MODES:
    rate = Y_specialists[mode].mean() * 100
    print(f"  {mode}: {rate:.2f}%  ({int(Y_specialists[mode].sum())} rows)")


# ---------------------------------------------------------------------------
# Step 6 - 10-fold stratified cross-validation
# ---------------------------------------------------------------------------
banner("STEP 6: 10-fold stratified cross-validation")

# --- Strategy A + B (multi-class) ---
print("\n  Running multi-class models...")
multi_results = {}
for name, pipe in multiclass_pipelines.items():
    print(f"\n  Running {name}...")
    scores = cross_val_score(
        pipe, X, y_multiclass, cv=10, scoring="accuracy", n_jobs=-1
    )
    multi_results[name] = scores
    print(f"    per-fold: {[f'{s:.4f}' for s in scores]}")


# --- Strategy C (binary specialists) ---
# sklearn's MultiOutputClassifier doesn't share cross_val_score cleanly with
# custom argmax aggregation, so we do the k-fold loop by hand. The splits
# are computed once and reused across folds (and again in Step 8).
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
splits = list(skf.split(X, y_multiclass))

print(f"\n  Running 5 binary specialists + argmax aggregation...")
ensemble_scores = []
for fold_idx, (train_idx, test_idx) in enumerate(splits):
    # Fit one MultiOutputClassifier on the training fold
    ensemble = MultiOutputClassifier(
        estimator=GradientBoostingClassifier(n_estimators=100, random_state=42),
        n_jobs=-1,
    )
    full_pipeline = Pipeline([("preprocess", preprocessor), ("clf", ensemble)])

    full_pipeline.fit(X.iloc[train_idx], Y_specialists.iloc[train_idx])

    # Get per-mode probabilities on the test fold
    # MultiOutputClassifier.predict_proba returns a list of 5 (n, 2) arrays
    proba_list = full_pipeline.predict_proba(X.iloc[test_idx])
    # proba_list[i][:, 1] = P(mode i is firing)
    proba_matrix = np.column_stack([p[:, 1] for p in proba_list])  # shape (n_test, 5)

    # Argmax aggregation: pick mode with highest probability;
    # if max < 0.5, predict "none"
    predicted_mode_idx = proba_matrix.argmax(axis=1)
    predicted_max_proba = proba_matrix.max(axis=1)

    pred_multiclass = np.where(
        predicted_max_proba >= 0.5,
        np.array(FAILURE_MODES)[predicted_mode_idx],
        "none",
    )

    # Score: top-1 accuracy vs the multi-class target
    fold_accuracy = float((pred_multiclass == y_multiclass.iloc[test_idx].values).mean())
    ensemble_scores.append(fold_accuracy)

ensemble_scores = np.array(ensemble_scores)
print(f"    per-fold: {[f'{s:.4f}' for s in ensemble_scores]}")


# ---------------------------------------------------------------------------
# Step 7 - Six Sigma comparison
# ---------------------------------------------------------------------------
banner("STEP 7: Six Sigma comparison")

# Lower spec limit = majority-class baseline (no failure)
majority_baseline = float((y_multiclass == "none").mean())
print(f"\nMajority-class baseline (lower spec limit): {majority_baseline * 100:.2f}%")
print()

# Add ensemble to results
all_results = dict(multi_results)
all_results["5-specialist ensemble (GB + argmax)"] = ensemble_scores

all_metrics = report_comparison(
    all_results,
    dataset_name="AI4I 2020 multi-class (5 failure modes + no failure)",
    lower_spec_limit=majority_baseline,
)

# Add the baseline as a comparison row (computed inline to avoid printing
# a full report banner just to extract one value).
if 0.0 < majority_baseline < 1.0:
    baseline_sigma = float(stats.norm.ppf(majority_baseline) + SIGMA_SHIFT)
else:
    baseline_sigma = float("-inf")
print()
print("=" * 70)
print("BASELINE: predict 'no failure' for every row")
print("=" * 70)
print(f"  Mean accuracy:        {majority_baseline * 100:>6.2f}%")
print(f"  DPMO:                 {(1.0 - majority_baseline) * 1_000_000:>10,.0f}")
print(f"  Sigma level:          {baseline_sigma:>6.2f}sigma")
print("=" * 70)


# ---------------------------------------------------------------------------
# Step 8 - Macro F1 comparison (better for imbalanced multi-class)
# ---------------------------------------------------------------------------
banner("STEP 8: Macro F1 comparison (handles class imbalance properly)")

print("\nMacro F1 = F1 averaged across all 6 classes, weighted equally.")
print("Unlike accuracy, macro F1 doesn't reward 'predict none every time.'")
print()

from sklearn.model_selection import cross_val_score
macro_f1_results = {}
for name, pipe in multiclass_pipelines.items():
    print(f"  Computing macro F1 for {name}...")
    scores = cross_val_score(
        pipe, X, y_multiclass, cv=10, scoring="f1_macro", n_jobs=-1
    )
    macro_f1_results[name] = scores
    print(f"    per-fold: {[f'{s:.4f}' for s in scores]}")

# Macro F1 for the ensemble. Reuse the splits from Step 5 (computed once).
print(f"\n  Computing macro F1 for 5-specialist ensemble...")
ensemble_macro_f1 = []

for fold_idx, (train_idx, test_idx) in enumerate(splits):
    ensemble = MultiOutputClassifier(
        estimator=GradientBoostingClassifier(n_estimators=100, random_state=42),
        n_jobs=-1,
    )
    full_pipeline = Pipeline([("preprocess", preprocessor), ("clf", ensemble)])
    full_pipeline.fit(X.iloc[train_idx], Y_specialists.iloc[train_idx])

    proba_list = full_pipeline.predict_proba(X.iloc[test_idx])
    proba_matrix = np.column_stack([p[:, 1] for p in proba_list])
    predicted_mode_idx = proba_matrix.argmax(axis=1)
    predicted_max_proba = proba_matrix.max(axis=1)
    pred_multiclass = np.where(
        predicted_max_proba >= 0.5,
        np.array(FAILURE_MODES)[predicted_mode_idx],
        "none",
    )

    fold_f1 = f1_score(
        y_multiclass.iloc[test_idx].values,
        pred_multiclass,
        average="macro",
        zero_division=0,
    )
    ensemble_macro_f1.append(fold_f1)

ensemble_macro_f1 = np.array(ensemble_macro_f1)
macro_f1_results["5-specialist ensemble (GB + argmax)"] = ensemble_macro_f1
print(f"    per-fold: {[f'{s:.4f}' for s in ensemble_macro_f1]}")

print()
print("Macro F1 comparison (higher is better):")
print(f"  {'Model':<40} {'Mean':>10} {'Std':>10}")
print(f"  {'-' * 40} {'-' * 10} {'-' * 10}")
for name, scores in macro_f1_results.items():
    print(f"  {name:<40} {scores.mean() * 100:>9.2f}% +/- {scores.std() * 100:>+6.2f}%")


# ---------------------------------------------------------------------------
# Step 9 - Verdict on the consulting pitch premise (Project 7)
# ---------------------------------------------------------------------------
banner("STEP 9: Verdict on the consulting pitch premise")

print()
best_top1_sigma = max(m["sigma_level"] for m in all_metrics.values())
best_top1_model = max(all_metrics.keys(), key=lambda k: all_metrics[k]["sigma_level"])
best_macro_f1 = max(s.mean() for s in macro_f1_results.values())
best_macro_f1_model = max(macro_f1_results.keys(), key=lambda k: macro_f1_results[k].mean())

print(f"Best top-1 accuracy result: {best_top1_model}")
print(f"  Sigma level (top-1): {best_top1_sigma:.2f}sigma")
print(f"Best macro F1 result: {best_macro_f1_model}")
print(f"  Macro F1: {best_macro_f1 * 100:.2f}%")
print()

if best_top1_sigma >= 5.0:
    print("VERDICT: Multi-class + ensemble closes part of the gap to 6sigma.")
    print("         Validates the 'ensemble of narrow specialists' strategy.")
elif best_top1_sigma >= 4.5:
    print("VERDICT: Multi-class + ensemble gives a meaningful gain over Project 4.")
    print("         Still 1.5sigma short of 6sigma. The 4-layer framework is still")
    print("         needed (FMEA + Poka-Yoke to close the last 1.5sigma gap).")
else:
    print("VERDICT: Multi-class + ensemble does NOT reach 6sigma.")
    print("         Even with 5 specialists + argmax, the sigma ceiling sits")
    print("         near 4-5sigma because the underlying signal in the honest")
    print("         6 features is too weak for the rare failure modes.")
    print("         The path to 6sigma requires:")
    print("         (a) better features (time series, sensor data beyond 6 columns),")
    print("         (b) the L2 Poka-Yoke layer (range checks, sensor sanity),")
    print("         (c) the L1 FMEA layer (knowing which mode is worth chasing).")

print()
print("Macro F1 is the more honest metric for this experiment because:")
print("  - 'Predict none every time' gives 96.61% accuracy but 1/6 macro F1")
print("    (~16.67%, the random baseline for 6 balanced classes)")
print("  - A model that catches SOME rare modes scores higher on macro F1")
print("    than on accuracy, which is what capital-intensive operations care about")
print("    (catching the rare event is the whole point of predictive maintenance)")

print()
print("Done. Multi-class + ensemble comparison complete.")
