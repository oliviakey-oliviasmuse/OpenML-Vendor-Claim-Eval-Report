"""
Project 2b - The honest comparison: 3 algorithms on AI4I 2020, NO leakage.

Project 2 ran with all 13 columns and produced:
    Random Forest     99.91% +/- 0.09%   (4.62sigma, Cpk 0.32)
    Gradient Boosting 99.56% +/- 0.55%   (4.12sigma, Cpk 0.27)
    kNN               53.91% +/- 22.40%  (1.60sigma, Cpk -0.64)

The RF and GB numbers were inflated by target leakage. Five columns in
AI4I 2020 - TWF, HDF, PWF, OSF, RNF - are the direct CAUSES of the
`Machine failure` target. The target is literally computed from them.

This script drops those five columns explicitly and re-runs the same
3-algorithm comparison. The honest numbers should land:
    RF/GB: ~96% (the actual capability on the real predictive features)
    kNN:   still poor (~50% or worse)
    Majority-class baseline: 96.61%

If the honest RF/GB land at ~96%, they are barely better than predicting
"no failure" every time. That is the L4 Eval truth on AI4I without
the answer key in the feature list.

Run:
    python 02b-compare-3-algorithms-ai4i-no-leakage.py
"""

import openml
import pandas as pd
from sklearn import neighbors
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from dpmo import report_six_sigma, report_comparison


# Columns to drop: they are direct causes of the `Machine failure` target.
# Keeping them in the feature list = target leakage = inflated accuracy.
LEAKAGE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]


def banner(text):
    print()
    print("=" * 70)
    print(text)
    print("=" * 70)


# ---------------------------------------------------------------------------
# Step 1 - Locate AI4I 2020 Predictive Maintenance on OpenML
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
banner("STEP 2: Fetch data")

target_candidates = [
    "Machine failure",
    "Machine_failure",
    "machine_failure",
    "machine failure",
    "Class",
    "class",
    "target",
    "Target",
]

X = y = categorical_indicator = attribute_names = None
for target_name in target_candidates:
    try:
        X, y, categorical_indicator, attribute_names = dataset.get_data(target=target_name)
        print(f"Found target column: '{target_name}'")
        break
    except Exception:
        continue

if y is None:
    print("Could not auto-detect target. Available columns:")
    print(attribute_names)
    raise ValueError("Specify target column explicitly in the script.")

print(f"Dataset:           {dataset.name} (id={dataset.dataset_id})")
print(f"Shape (raw):       X={X.shape}, y={y.shape}")


# ---------------------------------------------------------------------------
# Step 2.5 - Inspect dtypes, drop identifier columns
# ---------------------------------------------------------------------------
banner("STEP 2.5: Inspect dtypes, drop identifier columns")

# Drop high-cardinality string columns (identifiers like Product ID)
identifier_cols = [
    col for col in X.columns
    if pd.api.types.is_string_dtype(X[col]) and X[col].nunique() > 100
]
if identifier_cols:
    print(f"\nDropping identifier columns: {identifier_cols}")
    X = X.drop(columns=identifier_cols)


# ---------------------------------------------------------------------------
# Step 2.6 - DROP TARGET LEAKAGE COLUMNS
#             These are the direct causes of `Machine failure`.
#             Keeping them in the feature list = cheating.
# ---------------------------------------------------------------------------
banner("STEP 2.6: Drop target leakage columns")

leakage_present = [col for col in LEAKAGE_COLUMNS if col in X.columns]
if leakage_present:
    print(f"\nDropping target leakage columns: {leakage_present}")
    print("  Reason: each is a direct cause of the `Machine failure` target.")
    print("  Keeping them in the feature list = reading the answer key.")
    X = X.drop(columns=leakage_present)
else:
    print("  No leakage columns found (already removed?).")

print(f"\nFinal feature set: {list(X.columns)}")
print(f"Final shape: X={X.shape}")


# ---------------------------------------------------------------------------
# Step 2.7 - Re-detect categorical indicator from final dtypes
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
# Step 4 - Build 3 pipelines
# ---------------------------------------------------------------------------
banner("STEP 4: Build 3 pipelines")

pipelines = {
    "kNN (k=5)": Pipeline(
        [
            ("preprocess", preprocessor),
            ("clf", neighbors.KNeighborsClassifier(n_neighbors=5)),
        ]
    ),
    "Random Forest (100 trees)": Pipeline(
        [
            ("preprocess", preprocessor),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ]
    ),
    "Gradient Boosting (100 trees)": Pipeline(
        [
            ("preprocess", preprocessor),
            ("clf", GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ]
    ),
}

for name in pipelines:
    print(f"  - {name}")


# ---------------------------------------------------------------------------
# Step 5 - 10-fold stratified cross-validation
# ---------------------------------------------------------------------------
banner("STEP 5: 10-fold stratified cross-validation")

results = {}
for name, pipe in pipelines.items():
    print(f"\n  Running {name}...")
    scores = cross_val_score(pipe, X, y, cv=10, scoring="accuracy", n_jobs=-1)
    results[name] = scores
    print(f"    per-fold: {[f'{s:.4f}' for s in scores]}")


# ---------------------------------------------------------------------------
# Step 6 - Comparison table (Six Sigma format)
# ---------------------------------------------------------------------------
banner("STEP 6: Comparison table (Six Sigma format, no leakage)")

# Lower spec limit = majority-class baseline
if y.dtype.kind in ("i", "u", "f"):
    majority_baseline = float((y == 0).mean())
else:
    from collections import Counter
    counts = Counter(y)
    majority_baseline = float(counts.most_common(1)[0][1] / sum(counts.values()))

print(f"\nMajority-class baseline (lower spec limit): {majority_baseline * 100:.2f}%")
print()

all_metrics = report_comparison(
    results,
    dataset_name="AI4I 2020 Predictive Maintenance (no leakage)",
    lower_spec_limit=majority_baseline,
)


# ---------------------------------------------------------------------------
# Step 7 - What this proves vs Project 2
# ---------------------------------------------------------------------------
banner("STEP 7: Honest comparison vs Project 2 (with leakage)")

print()
print("Project 2 (with leakage) → Project 2b (without leakage)")
print("-" * 70)
print(f"{'Model':<32} {'Project 2':>15} {'Project 2b':>15} {'Delta':>10}")
print("-" * 70)

project2_numbers = {
    "kNN (k=5)": 0.5391,
    "Random Forest (100 trees)": 0.9991,
    "Gradient Boosting (100 trees)": 0.9956,
}

for name, scores in results.items():
    new_mean = scores.mean()
    old_mean = project2_numbers[name]
    delta = (new_mean - old_mean) * 100
    print(
        f"  {name:<32} {old_mean * 100:>13.2f}%  {new_mean * 100:>13.2f}%  "
        f"{delta:>+8.2f}pp"
    )
print()

print("Expected deltas:")
print("  - kNN: minimal change (kNN doesn't benefit from leakage features much)")
print("  - RF/GB: should drop to ~96% (the honest capability on the real features)")
print("  - The '46 percentage point gap' from kNN to RF/GB should collapse to <2pp")
print()
print("This is the L4 Eval truth on AI4I: the real predictive signal in the")
print("remaining features (Air temp, Process temp, Rotational speed, Torque,")
print("Tool wear, Type) is modest. The '99% accurate' claim came from the answer")
print("key, not from the data.")


# ---------------------------------------------------------------------------
# Step 8 - Six Sigma verdict (the honest version)
# ---------------------------------------------------------------------------
banner("Step 8: Six Sigma verdict (the honest version)")

print()
print("Applying the Olivia standard: 6sigma is the only acceptable target.")
print()
for name, metrics in all_metrics.items():
    gap = 6.0 - metrics["sigma_level"]
    print(
        f"  {name:<32} "
        f"{metrics['sigma_level']:>5.2f}sigma "
        f"(Cpk {metrics['cpk']:>5.2f}, DPMO {metrics['dpmo']:>9,.0f})"
    )
    if metrics["sigma_level"] >= 6.0:
        print(f"    -> PASS (>= 6sigma)")
    else:
        print(f"    -> BELOW 6sigma, gap of {gap:.2f}sigma. Honest communication required.")

print()
print("If even the honest RF/GB only reach ~3.4sigma on AI4I, then '94%")
print("accurate predictive maintenance' is a 3.4sigma process at best. The")
print("real engineering target is 6sigma, and the gap is closed through the")
print("4-layer framework (FMEA, Poka-Yoke, Model, Eval) - not through a")
print("single ML model.")

print()
print("Done. Honest comparison complete.")