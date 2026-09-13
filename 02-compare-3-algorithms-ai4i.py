"""
Project 2 - Compare 3 algorithms on AI4I 2020 Predictive Maintenance.

The AI4I 2020 dataset is a synthetic predictive-maintenance dataset
from UCI that mimics realistic industrial failure modes (heat
dissipation, power failure, overstrain, tool wear). Around 10,000 rows,
6 predictive features, and a binary target (Machine failure: 0/1).

This script compares three model classes on the same dataset, with
the same preprocessing pipeline and the same evaluation procedure:

    1. kNN (k=5)                   - the project 1 baseline
    2. Random Forest (100 trees)   - typical industry default
    3. Gradient Boosting (100)     - typical "best in class" for tabular

Same data. Same evaluation. Different model class. The accuracy number
should move by 5-15 percentage points - that's the gap vendor claims
ride on.

Run:
    python 02-compare-3-algorithms-ai4i.py
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


def banner(text):
    print()
    print("=" * 70)
    print(text)
    print("=" * 70)


# ---------------------------------------------------------------------------
# Step 1 - Locate AI4I 2020 Predictive Maintenance on OpenML
#           (Dataset IDs vary across versions; dynamic lookup is safer.)
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
# Step 2 - Fetch data with robust target-column lookup
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

n_cat = sum(categorical_indicator)
n_num = len(categorical_indicator) - n_cat

print(f"Dataset:           {dataset.name} (id={dataset.dataset_id})")
print(f"Shape:             X={X.shape}, y={y.shape}")
print(f"Target classes:    {sorted(y.unique())}")
print(f"Class balance:     {y.value_counts().to_dict()}")
print(f"Categorical cols:  {n_cat} of {len(categorical_indicator)}")
print(f"Numerical cols:    {n_num} of {len(categorical_indicator)}")


# ---------------------------------------------------------------------------
# Step 2.5 - Inspect actual dtypes and drop identifier columns
#             OpenML's categorical_indicator can be wrong for some datasets
#             (e.g. AI4I 2020 reports 0 categoricals when 2 are clearly strings).
#             Always cross-check against the actual data.
# ---------------------------------------------------------------------------
banner("STEP 2.5: Inspect dtypes, drop identifier columns")

print(f"\nActual column dtypes:")
for i, col in enumerate(X.columns):
    n_unique = X[col].nunique()
    sample = X[col].iloc[0]
    print(f"  [{i:2d}] {col}: dtype={X[col].dtype}, n_unique={n_unique}, sample={sample!r}")

# Drop high-cardinality string columns (likely identifiers like Product IDs)
identifier_cols = [
    col for col in X.columns
    if pd.api.types.is_string_dtype(X[col]) and X[col].nunique() > 100
]

if identifier_cols:
    print(f"\nDropping identifier columns (high-cardinality strings): {identifier_cols}")
    X = X.drop(columns=identifier_cols)

# Recompute categorical indicator from actual dtypes
categorical_indicator = [pd.api.types.is_string_dtype(X[col]) for col in X.columns]
n_cat = sum(categorical_indicator)
n_num = len(categorical_indicator) - n_cat

print(f"\nCorrected indicator (based on actual dtypes):")
for i, (col, is_cat) in enumerate(zip(X.columns, categorical_indicator)):
    marker = "CAT" if is_cat else "NUM"
    print(f"  [{i:2d}] {marker}  {col}")

print(f"\nAfter cleanup: {n_cat} categorical, {n_num} numerical.")


# ---------------------------------------------------------------------------
# Step 3 - Shared preprocessor (same for all 3 algorithms)
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
# Step 4 - Build 3 pipelines (all share the same preprocessor)
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
# Step 5 - 10-fold stratified cross-validation for each
# ---------------------------------------------------------------------------
banner("STEP 5: 10-fold stratified cross-validation")

results = {}
for name, pipe in pipelines.items():
    print(f"\n  Running {name}...")
    scores = cross_val_score(pipe, X, y, cv=10, scoring="accuracy", n_jobs=-1)
    results[name] = scores
    print(f"    per-fold: {[f'{s:.4f}' for s in scores]}")


# ---------------------------------------------------------------------------
# Step 6 - Comparison table (Six Sigma format via dpmo)
# ---------------------------------------------------------------------------
banner("STEP 6: Comparison table (Six Sigma format)")

# Lower spec limit = majority-class baseline (for imbalanced Cpk)
if y.dtype.kind in ("i", "u", "f"):
    majority_baseline = float((y == 0).mean())
else:
    from collections import Counter
    counts = Counter(y)
    majority_baseline = float(counts.most_common(1)[0][1] / sum(counts.values()))

all_metrics = report_comparison(
    results,
    dataset_name="AI4I 2020 Predictive Maintenance",
    lower_spec_limit=majority_baseline,
)


# ---------------------------------------------------------------------------
# Step 7 - What this proves
# ---------------------------------------------------------------------------
banner("STEP 7: What this proves")

best_name = max(results, key=lambda k: results[k].mean())
worst_name = min(results, key=lambda k: results[k].mean())
gap_points = (results[best_name].mean() - results[worst_name].mean()) * 100

print()
print(f"  Worst:  {worst_name:<32} {results[worst_name].mean() * 100:.2f}%")
print(f"  Best:   {best_name:<32} {results[best_name].mean() * 100:.2f}%")
print(f"  Gap:    {gap_points:.2f} percentage points")
print()
print("Same dataset. Same evaluation procedure. Same preprocessing pipeline.")
print("The only thing that changed was the model class. The accuracy number")
print(f"moved by {gap_points:.1f} points.")
print()
print("This is the L4 Eval lesson: vendor claims ride on model-class selection,")
print("not magic. A '94% accurate' predictive maintenance model is almost")
print("certainly gradient boosting (or worse, a neural net). A kNN baseline")
print("tells you what the data actually supports without model-driven amplification.")


# ---------------------------------------------------------------------------
# Step 8 - The class-imbalance trap (the deeper L4 lesson)
# ---------------------------------------------------------------------------
banner("STEP 8: The class-imbalance trap")

# Show what "predict majority class always" achieves on this data
if y.dtype.kind in ("i", "u", "f"):
    majority_class = y.mode()[0]
    majority_acc = (y == majority_class).mean()
else:
    # string labels
    counts = y.value_counts()
    majority_class = counts.index[0]
    majority_acc = counts.iloc[0] / counts.sum()

print()
print(f"  Majority class:     '{majority_class}'")
print(f"  Predict-always accuracy: {majority_acc * 100:.2f}%")
print()
print("Notice: just predicting the majority class every time gets you")
print("close to (or above) any non-trivial classifier. That's the class-")
print("imbalance trap. On an imbalanced dataset, accuracy alone is not")
print("a useful metric.")
print()
print("Better metrics for imbalanced data: F1 score, ROC-AUC, precision-recall.")
print("Add them to this script if you want to see the gap widen between a")
print("trivial baseline and a real classifier.")


# ---------------------------------------------------------------------------
# Step 9 - Vendor claim sanity check
# ---------------------------------------------------------------------------
banner("STEP 9: Vendor claim sanity check")

print()
print("If a vendor tells you their predictive-maintenance model is '94%")
print("accurate' on your line, the L4 audit questions are:")
print()
print("  1. What model class? (Probably gradient boosting or similar.)")
print("  2. What public benchmark? (If none, ask why not.)")
print("  3. What's the baseline? (What does kNN get on the same data?)")
print("  4. What's the class imbalance? ('94% accurate' on 96% 'no failure'")
print("     data is not impressive - predict 'no failure' every time and")
print("     you get 96% accuracy.")
print("  5. What's the evaluation metric? (Accuracy hides imbalance;")
print("     ask for F1 or ROC-AUC.)")
print()
print("If the vendor can't answer 1-5, your plant is paying for vibes.")
print()
print("Done. Three algorithms, one dataset, one audit trail.")