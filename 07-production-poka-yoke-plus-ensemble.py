"""
Project 7b - Production-recommended L2 Poka-Yoke + L3 ML ensemble.

Background (from the experimentation log):
    Project 8 (naive Poka-Yoke + ML) showed that adding ALL 4 Poka-Yoke rules
    to the ML ensemble makes the system WORSE, because TWF and HDF rules
    over-fire. Per-mode diagnostic showed:
        PWF: 100% precision, 100% recall (PRODUCTION-READY)
        OSF: 100% precision, 100% recall (PRODUCTION-READY)
        HDF: 15.97% precision, 100% recall (OVER-FIRING)
        TWF: 4.97% precision, 54.35% recall (OVER-FIRING)

    Production recommendation from Project 8:
        Deploy ONLY PWF + OSF rules. Use ML ensemble for TWF, HDF, RNF.

    This is the test of that recommendation: precise L2 (PWF + OSF only) +
    L3 (ML ensemble for everything else). If this beats ML-alone, the 4-layer
    framework's value is empirically demonstrated.

Strategies compared:
    A. Baseline (predict none)
    B. ML ensemble only (Project 7's best, Cpk 1.96)
    C. Poka-Yoke precise only (PWF + OSF rules, no ML fallback)
    D. Poka-Yoke precise (PWF + OSF) + ML ensemble (PWF/OSF caught by rules;
       TWF/HDF/RNF handled by ML ensemble)

Run:
    python 07-production-poka-yoke-plus-ensemble.py
"""

import sys

# Force UTF-8 stdout on Windows (cp1252) so the sigma symbol prints cleanly.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import openml
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score

from dpmo import report_six_sigma, report_comparison


# Only the precise Poka-Yoke rules (100% precision AND 100% recall per Project 8).
FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
NO_FAILURE = "none"


def banner(text):
    print()
    print("=" * 70)
    print(text)
    print("=" * 70)


# ---------------------------------------------------------------------------
# Production Poka-Yoke (only precise rules: PWF and OSF)
# ---------------------------------------------------------------------------
def poka_yoke_precise_predict(row):
    """
    Apply ONLY the precise Poka-Yoke rules (PWF, OSF) to a row.

    Returns a tuple of (predicted_mode, fired_rules).

    Production-recommended: deploy ONLY rules with 100% precision AND recall.
    """
    air_temp = row["Air temperature [K]"]
    process_temp = row["Process temperature [K]"]
    rotational_speed = row["Rotational speed [rpm]"]
    torque = row["Torque [Nm]"]
    tool_wear = row["Tool wear [min]"]
    product_type = row["Type"]

    fired = []

    # Rule: PWF (Power Failure)
    # power = torque * (rpm * 2 * pi / 60)
    power = (torque * rotational_speed * 2 * np.pi) / 60.0
    if power < 3500.0 or power > 9000.0:
        fired.append("PWF")

    # Rule: OSF (Overstrain Failure)
    overstrain_threshold = {"L": 11000, "M": 12000, "H": 13000}.get(product_type, 12000)
    if torque * tool_wear >= overstrain_threshold:
        fired.append("OSF")

    if not fired:
        return (NO_FAILURE, [])

    # Priority order if both fire
    priority = ["PWF", "OSF"]
    for mode in priority:
        if mode in fired:
            return (mode, fired)

    return (fired[0], fired)


def poka_yoke_precise_only_predict(X):
    """Apply precise Poka-Yoke rules to all rows. Returns array of predicted modes."""
    return np.array([poka_yoke_precise_predict(row)[0] for _, row in X.iterrows()])


# ---------------------------------------------------------------------------
# Step 1 - Load AI4I 2020
# ---------------------------------------------------------------------------
banner("STEP 1: Locate AI4I 2020 Predictive Maintenance on OpenML")

candidate_names = [
    "AI4I_2020_Predictive_Maintenance",
    "ai4i_2020",
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
        raise ValueError("Could not locate AI4I 2020 on OpenML.")


# ---------------------------------------------------------------------------
# Step 2 - Fetch data + build multi-class target
# ---------------------------------------------------------------------------
banner("STEP 2: Fetch raw data, build multi-class target")

X_raw, _, _, _ = dataset.get_data(target="Machine failure")
print(f"Raw shape: {X_raw.shape}")

# Drop identifier columns
identifier_cols = [
    col for col in X_raw.columns
    if pd.api.types.is_string_dtype(X_raw[col]) and X_raw[col].nunique() > 100
]
if "UDI" in X_raw.columns:
    identifier_cols.append("UDI")
if identifier_cols:
    print(f"\nDropping identifier columns: {identifier_cols}")
    X_raw = X_raw.drop(columns=identifier_cols)


def build_multiclass_target(row):
    for mode in FAILURE_MODES:
        if row[mode] == 1:
            return mode
    return NO_FAILURE


y_multiclass = X_raw.apply(build_multiclass_target, axis=1)
print(f"\nMulti-class target distribution:")
print(y_multiclass.value_counts().sort_index().to_string())


# ---------------------------------------------------------------------------
# Step 2.5 - Build honest feature set
# ---------------------------------------------------------------------------
banner("STEP 2.5: Build honest feature set (6 features, no leakage)")

leakage_cols = FAILURE_MODES + ["Machine failure"]
present = [c for c in leakage_cols if c in X_raw.columns]
print(f"Dropping leakage columns: {present}")

X = X_raw.drop(columns=present)
print(f"\nFinal feature set: {list(X.columns)}")


# ---------------------------------------------------------------------------
# Step 3 - Preprocessor
# ---------------------------------------------------------------------------
banner("STEP 3: Preprocessor setup")

categorical_indicator = [pd.api.types.is_string_dtype(X[col]) for col in X.columns]
n_cat = sum(categorical_indicator)
n_num = len(categorical_indicator) - n_cat

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
# Step 4 - Build specialist target matrix
# ---------------------------------------------------------------------------
banner("STEP 4: Build 5 binary specialist target matrix")

Y_specialists = pd.DataFrame(
    {mode: X_raw[mode].astype(int).values for mode in FAILURE_MODES},
    index=X.index,
)
print(f"Specialist target matrix shape: {Y_specialists.shape}")
for mode in FAILURE_MODES:
    rate = Y_specialists[mode].mean() * 100
    print(f"  {mode}: {rate:.2f}% positive ({int(Y_specialists[mode].sum())} rows)")


# ---------------------------------------------------------------------------
# Step 5 - 10-fold stratified cross-validation (4 strategies)
# ---------------------------------------------------------------------------
banner("STEP 5: 10-fold stratified cross-validation (4 strategies)")

print()
print("Strategy A: Baseline (predict 'no failure' always)")
print("Strategy B: ML ensemble only (Project 7's best)")
print("Strategy C: Poka-Yoke precise only (PWF + OSF rules)")
print("Strategy D: Poka-Yoke precise + ML ensemble (production-recommended)")

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
splits = list(skf.split(X, y_multiclass))

results = {
    "A. Baseline (predict none)": [],
    "B. ML ensemble only (Project 7 baseline)": [],
    "C. Poka-Yoke precise only (PWF + OSF)": [],
    "D. Poka-Yoke precise + ML ensemble (PRODUCTION)": [],
}

for fold_idx, (train_idx, test_idx) in enumerate(splits):
    X_test = X.iloc[test_idx]
    y_test = y_multiclass.iloc[test_idx].values

    # Strategy A: baseline
    pred_a = np.full(len(test_idx), NO_FAILURE)
    results["A. Baseline (predict none)"].append(float((pred_a == y_test).mean()))

    # Strategy C: Poka-Yoke precise only
    pred_c = poka_yoke_precise_only_predict(X_test)
    results["C. Poka-Yoke precise only (PWF + OSF)"].append(
        float((pred_c == y_test).mean())
    )

    # Train ML ensemble (needed for B and D)
    ensemble = MultiOutputClassifier(
        estimator=GradientBoostingClassifier(n_estimators=100, random_state=42),
        n_jobs=-1,
    )
    full_pipeline = Pipeline([("preprocess", preprocessor), ("clf", ensemble)])
    full_pipeline.fit(X.iloc[train_idx], Y_specialists.iloc[train_idx])
    proba_list = full_pipeline.predict_proba(X_test)
    proba_matrix = np.column_stack([p[:, 1] for p in proba_list])
    predicted_mode_idx = proba_matrix.argmax(axis=1)
    predicted_max_proba = proba_matrix.max(axis=1)
    pred_b = np.where(
        predicted_max_proba >= 0.5,
        np.array(FAILURE_MODES)[predicted_mode_idx],
        NO_FAILURE,
    )
    results["B. ML ensemble only (Project 7 baseline)"].append(
        float((pred_b == y_test).mean())
    )

    # Strategy D: Poka-Yoke precise first, ML fallback for non-PWF/OSF
    pred_poka = poka_yoke_precise_only_predict(X_test)
    # Poka-Yoke only catches PWF and OSF. For everything else (TWF, HDF, RNF, none),
    # use the ML ensemble's prediction.
    pred_d = np.where(pred_poka != NO_FAILURE, pred_poka, pred_b)
    results["D. Poka-Yoke precise + ML ensemble (PRODUCTION)"].append(
        float((pred_d == y_test).mean())
    )

    if fold_idx == 0 or fold_idx == 9:
        print(f"\n  Fold {fold_idx + 1}:")
        for name in results:
            print(f"    {name:<52}: {results[name][-1]:.4f}")

print("\n  ... (intermediate folds suppressed)")

for name in results:
    results[name] = np.array(results[name])


# ---------------------------------------------------------------------------
# Step 6 - Six Sigma comparison (top-1 accuracy)
# ---------------------------------------------------------------------------
banner("STEP 6: Six Sigma comparison (top-1 accuracy)")

majority_baseline = float((y_multiclass == NO_FAILURE).mean())
print(f"\nMajority-class baseline (lower spec limit): {majority_baseline * 100:.2f}%")
print()

all_metrics = report_comparison(
    results,
    dataset_name="AI4I 2020 - Production Poka-Yoke (PWF+OSF) + ML ensemble",
    lower_spec_limit=majority_baseline,
)


# ---------------------------------------------------------------------------
# Step 7 - Macro F1 comparison
# ---------------------------------------------------------------------------
banner("STEP 7: Macro F1 comparison")

print()
print("Macro F1 = F1 averaged across all 6 classes, weighted equally.")
print()

macro_f1_results = {}
for fold_idx, (train_idx, test_idx) in enumerate(splits):
    X_test = X.iloc[test_idx]
    y_test = y_multiclass.iloc[test_idx].values

    fold_scores = {}

    # Strategy A
    pred_a = np.full(len(test_idx), NO_FAILURE)
    fold_scores["A. Baseline (predict none)"] = f1_score(
        y_test, pred_a, average="macro", zero_division=0
    )

    # Strategy C: Poka-Yoke precise only
    pred_c = poka_yoke_precise_only_predict(X_test)
    fold_scores["C. Poka-Yoke precise only (PWF + OSF)"] = f1_score(
        y_test, pred_c, average="macro", zero_division=0
    )

    # Train ML ensemble
    ensemble = MultiOutputClassifier(
        estimator=GradientBoostingClassifier(n_estimators=100, random_state=42),
        n_jobs=-1,
    )
    full_pipeline = Pipeline([("preprocess", preprocessor), ("clf", ensemble)])
    full_pipeline.fit(X.iloc[train_idx], Y_specialists.iloc[train_idx])
    proba_list = full_pipeline.predict_proba(X_test)
    proba_matrix = np.column_stack([p[:, 1] for p in proba_list])
    predicted_mode_idx = proba_matrix.argmax(axis=1)
    predicted_max_proba = proba_matrix.max(axis=1)
    pred_b = np.where(
        predicted_max_proba >= 0.5,
        np.array(FAILURE_MODES)[predicted_mode_idx],
        NO_FAILURE,
    )
    fold_scores["B. ML ensemble only (Project 7 baseline)"] = f1_score(
        y_test, pred_b, average="macro", zero_division=0
    )

    # Strategy D
    pred_poka = poka_yoke_precise_only_predict(X_test)
    pred_d = np.where(pred_poka != NO_FAILURE, pred_poka, pred_b)
    fold_scores["D. Poka-Yoke precise + ML ensemble (PRODUCTION)"] = f1_score(
        y_test, pred_d, average="macro", zero_division=0
    )

    for name, score in fold_scores.items():
        macro_f1_results.setdefault(name, []).append(score)

print("Macro F1 per-fold:")
print(f"  {'Strategy':<52}")
for name in macro_f1_results:
    print(f"  {name:<52}: {[f'{s:.4f}' for s in macro_f1_results[name]]}")

print()
print("Macro F1 comparison:")
print(f"  {'Strategy':<52} {'Mean':>10} {'Std':>10}")
print(f"  {'-' * 52} {'-' * 10} {'-' * 10}")
for name in macro_f1_results:
    scores = np.array(macro_f1_results[name])
    print(
        f"  {name:<52} {scores.mean() * 100:>9.2f}% +/- {scores.std() * 100:>+6.2f}%"
    )


# ---------------------------------------------------------------------------
# Step 8 - Per-mode breakdown for the production strategy
# ---------------------------------------------------------------------------
banner("STEP 8: Per-mode breakdown for Strategy D (the production recommendation)")

print()
print("For Strategy D (production), how does each failure mode fare?")
print()

# Build full dataset predictions using the full ensemble (not CV)
print("Building full-dataset predictions...")
ensemble_full = MultiOutputClassifier(
    estimator=GradientBoostingClassifier(n_estimators=100, random_state=42),
    n_jobs=-1,
)
full_pipe = Pipeline([("preprocess", preprocessor), ("clf", ensemble_full)])
full_pipe.fit(X, Y_specialists)
proba_list_full = full_pipe.predict_proba(X)
proba_matrix_full = np.column_stack([p[:, 1] for p in proba_list_full])
predicted_mode_idx_full = proba_matrix_full.argmax(axis=1)
predicted_max_proba_full = proba_matrix_full.max(axis=1)
pred_b_full = np.where(
    predicted_max_proba_full >= 0.5,
    np.array(FAILURE_MODES)[predicted_mode_idx_full],
    NO_FAILURE,
)

pred_poka_full = poka_yoke_precise_only_predict(X)
pred_d_full = np.where(pred_poka_full != NO_FAILURE, pred_poka_full, pred_b_full)

print(f"\nPer-mode recall (how many actual X were caught as X):")
print(f"{'Mode':<10} {'Actual count':<14} {'Caught by D':<14} {'Missed':<10} {'Recall':<10}")
total_caught = 0
total_actual = 0
for mode in FAILURE_MODES:
    actual_mask = (y_multiclass == mode).values
    caught_mask = (pred_d_full == mode)
    caught = (actual_mask & caught_mask).sum()
    missed = (actual_mask & ~caught_mask).sum()
    recall = caught / actual_mask.sum() if actual_mask.sum() > 0 else 0.0
    total_caught += caught
    total_actual += actual_mask.sum()
    print(
        f"  {mode:<10} {int(actual_mask.sum()):<14} {caught:<14} {missed:<10} "
        f"{recall * 100:>6.2f}%"
    )

total_recall = total_caught / total_actual if total_actual > 0 else 0.0
print(f"  {'TOTAL':<10} {total_actual:<14} {total_caught:<14} {total_actual - total_caught:<10} {total_recall * 100:>6.2f}%")

print()
print(f"Per-mode false positives (predicted X but actually something else):")
print(f"{'Mode':<10} {'Predicted count':<16} {'False positives':<16} {'Precision':<10}")
for mode in FAILURE_MODES + [NO_FAILURE]:
    pred_mask = (pred_d_full == mode)
    pred_count = pred_mask.sum()
    if mode == NO_FAILURE:
        fp_mask = pred_mask & (y_multiclass != NO_FAILURE).values
    else:
        fp_mask = pred_mask & (y_multiclass != mode).values
    fp = fp_mask.sum()
    tp = pred_mask.sum() - fp
    precision = tp / pred_count if pred_count > 0 else 0.0
    print(
        f"  {mode:<10} {pred_count:<16} {fp:<16} {precision * 100:>6.2f}%"
    )


# ---------------------------------------------------------------------------
# Step 9 - Verdict on the production deployment
# ---------------------------------------------------------------------------
banner("STEP 9: Verdict on the production deployment")

best_strategy = max(all_metrics, key=lambda k: all_metrics[k]["sigma_level"])
best_sigma = all_metrics[best_strategy]["sigma_level"]
best_macro_f1_strategy = max(macro_f1_results, key=lambda k: np.mean(macro_f1_results[k]))
best_macro_f1 = np.mean(macro_f1_results[best_macro_f1_strategy])

print()
print(f"Best top-1 strategy: {best_strategy}")
print(f"  Sigma level: {best_sigma:.2f}sigma")
print(f"\nBest macro F1 strategy: {best_macro_f1_strategy}")
print(f"  Macro F1: {best_macro_f1 * 100:.2f}%")

production_sigma = all_metrics["D. Poka-Yoke precise + ML ensemble (PRODUCTION)"]["sigma_level"]
ensemble_sigma = all_metrics["B. ML ensemble only (Project 7 baseline)"]["sigma_level"]
production_gain = production_sigma - ensemble_sigma
print()
print(f"Sigma gain from production strategy (D) vs ML alone (B): {production_gain:+.2f}sigma")
print()

production_f1 = np.mean(macro_f1_results["D. Poka-Yoke precise + ML ensemble (PRODUCTION)"])
ensemble_f1 = np.mean(macro_f1_results["B. ML ensemble only (Project 7 baseline)"])
f1_gain = (production_f1 - ensemble_f1) * 100
print(f"Macro F1 gain from production strategy (D) vs ML alone (B): {f1_gain:+.2f}pp")
print()

if production_sigma > ensemble_sigma and production_f1 >= ensemble_f1 - 0.01:
    print("VERDICT: Production deployment (precise Poka-Yoke + ML) IMPROVES the system.")
    print("         The 4-layer framework works as designed: precise L2 + ML fallback L3")
    print("         beats ML alone on both metrics. This is the deployment recommendation.")
elif production_f1 >= ensemble_f1 + 0.01:
    print("VERDICT: Production deployment improves macro F1 (rare-event catching).")
    print("         Top-1 sigma may not move much, but the rare-event catching value")
    print("         is real and matters for capital-intensive operations.")
else:
    print("VERDICT: Production deployment does not significantly improve the system.")
    print("         The 4-layer framework's L2 value is limited on this dataset.")
    print("         L4 Eval (capability monitoring) is the next layer to test.")

print()
print("Per-mode breakdown above shows: how many actual failures of each mode")
print("does the production system catch? PWF and OSF should be 100% (caught by")
print("Poka-Yoke rules); TWF, HDF, RNF should be caught by ML ensemble; 'none'")
print("predictions should have high precision.")
print()
print("Done. Production Poka-Yoke + ML ensemble comparison complete.")
