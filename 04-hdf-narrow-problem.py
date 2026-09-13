"""
Project 4 - Narrow problem definition: predict Heat Dissipation Failure (HDF) only.

Background (from the experimentation log):
    Project 2b showed that predicting "any Machine failure" on AI4I 2020 with
    honest features (no target leakage) lands the best models (RF/GB) at
    ~96-97% accuracy, which is statistically equivalent to the 96.61%
    majority-class baseline. Real predictive signal on AI4I is ~3.4sigma,
    not 6sigma.

    One of the five interlocking root causes identified was:
    (3) "any failure" is too broad a problem definition.

    This script tests that hypothesis empirically by narrowing the problem:
    instead of "any Machine failure," predict HDF (Heat Dissipation Failure)
    only. HDF is one of the four specific failure modes in AI4I 2020 and
    has a different class balance and a different (theoretically stronger)
    relationship to thermal features.

Hypothesis:
    Narrowing from "any failure" to "HDF only" should:
    - Increase the class imbalance (HDF is rarer than "any failure")
    - Raise the majority-class baseline (~98.85% vs 96.61%)
    - Force the model to find a real thermal signal (Air temp, Process temp,
      Temp differential) instead of averaging across 5 unrelated failure modes

If sigma level rises meaningfully above the 3.4sigma ceiling, the consulting
pitch's premise - that narrow problem definition is part of the path to 6sigma
- is empirically validated.

If sigma level stays at 3.4sigma or below, then even narrow problem
definitions can't rescue the data; the path to 6sigma must come from the
4-layer framework (FMEA + Poka-Yoke + Model + Eval), not from ML alone.

Run:
    python 04-hdf-narrow-problem.py
"""

import openml
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from dpmo import report_six_sigma, report_comparison


# Force UTF-8 stdout on Windows (cp1252) so the σ symbol prints cleanly.
# dpmo.py does this too, but doing it here keeps the script self-contained.
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass


# Columns to drop: they are direct causes of EITHER the broad `Machine failure`
# target (target leakage) OR are the OTHER specific failure modes that would
# leak the HDF target if kept alongside it.
#
# AI4I 2020 columns:
#   UDI, Product ID           -- identifiers (always drop)
#   Type                      -- categorical (keep: L/M/H product type)
#   Air temperature [K]       -- numeric (keep)
#   Process temperature [K]   -- numeric (keep)
#   Rotational speed [rpm]    -- numeric (keep)
#   Torque [Nm]               -- numeric (keep)
#   Tool wear [min]           -- numeric (keep)
#   Machine failure           -- BROAD target (drop: would leak HDF)
#   TWF, HDF, PWF, OSF, RNF   -- the four other failure modes
#
# New target for this experiment: HDF (binary)
# Honest feature set: Type + 5 numeric columns (6 features total)
DROP_FOR_HDF_TARGET = [
    "UDI",                # identifier
    "Product ID",         # identifier
    "Machine failure",    # broad target - HDF is a subset, would leak
    "TWF",                # other failure mode - might leak HDF via overlap
    "PWF",                # other failure mode - might leak HDF via overlap
    "OSF",                # other failure mode - might leak HDF via overlap
    "RNF",                # other failure mode - might leak HDF via overlap
]


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
# Step 2 - Fetch data, target = HDF
# ---------------------------------------------------------------------------
banner("STEP 2: Fetch data, narrow target = HDF (Heat Dissipation Failure)")

target_candidates = ["HDF", "hdf"]

X = y = categorical_indicator = attribute_names = None
for target_name in target_candidates:
    try:
        X, y, categorical_indicator, attribute_names = dataset.get_data(target=target_name)
        print(f"Found target column: '{target_name}'")
        break
    except Exception:
        continue

if y is None:
    print("Could not auto-detect HDF target. Available columns:")
    print(attribute_names)
    raise ValueError("Specify HDF target column explicitly in the script.")

print(f"Dataset:           {dataset.name} (id={dataset.dataset_id})")
print(f"Target:            HDF (Heat Dissipation Failure)")
print(f"Shape (raw):       X={X.shape}, y={y.shape}")
print(f"HDF positive rate: {float((y == 1).mean()) * 100:.2f}%")
print(f"HDF negative rate: {float((y == 0).mean()) * 100:.2f}%")


# ---------------------------------------------------------------------------
# Step 2.5 - Drop identifiers and leakage columns
# ---------------------------------------------------------------------------
banner("STEP 2.5: Drop identifiers + cross-mode leakage columns")

# Drop high-cardinality string columns (identifiers like Product ID)
identifier_cols = [
    col for col in X.columns
    if pd.api.types.is_string_dtype(X[col]) and X[col].nunique() > 100
]
if identifier_cols:
    print(f"\nDropping identifier columns: {identifier_cols}")
    X = X.drop(columns=identifier_cols)

# Drop the four other failure modes + Machine failure (would leak HDF)
leakage_cols = [col for col in DROP_FOR_HDF_TARGET
                if col in X.columns and col not in identifier_cols]
print(f"\nDropping cross-mode leakage columns: {leakage_cols}")
print("  Reason: TWF/PWF/OSF/RNF are other failure modes;")
print("          Machine failure is the broad target that contains HDF as a subset.")
print("          Keeping them in the feature list = reading the answer key for HDF.")
X = X.drop(columns=[c for c in leakage_cols if c in X.columns])

print(f"\nFinal feature set: {list(X.columns)}")
print(f"Final shape: X={X.shape}")


# ---------------------------------------------------------------------------
# Step 2.6 - Re-detect categorical indicator
# ---------------------------------------------------------------------------
banner("STEP 2.6: Re-detect categorical indicator")

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
# Step 4 - Build 3 pipelines (same models as Projects 2 and 2b)
# ---------------------------------------------------------------------------
banner("STEP 4: Build 3 pipelines")

pipelines = {
    "kNN (k=5)": Pipeline(
        [
            ("preprocess", preprocessor),
            ("clf", KNeighborsClassifier(n_neighbors=5)),
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
# Step 6 - Six Sigma comparison
# ---------------------------------------------------------------------------
banner("STEP 6: Comparison table (Six Sigma format, HDF-only target)")

# Lower spec limit = majority-class baseline (HDF is rare so this is high)
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
    dataset_name="AI4I 2020 HDF-only (narrow problem definition)",
    lower_spec_limit=majority_baseline,
)


# ---------------------------------------------------------------------------
# Step 7 - Direct comparison: broad vs narrow problem
# ---------------------------------------------------------------------------
banner("STEP 7: Direct comparison - broad problem (Project 2b) vs narrow (Project 4)")

project2b_numbers = {
    "kNN (k=5)": {"mean": 0.5391, "sigma": 1.60, "cpk": -0.64},
    "Random Forest (100 trees)": {"mean": 0.9565, "sigma": 3.21, "cpk": -0.10},
    "Gradient Boosting (100 trees)": {"mean": 0.9680, "sigma": 3.35, "cpk": 0.03},
}

print()
print(f"{'Model':<32} {'Broad (Proj 2b)':>18} {'Narrow (Proj 4)':>18} {'Sigma delta':>14}")
print("-" * 86)

for name, metrics in all_metrics.items():
    old = project2b_numbers[name]
    new_sigma = metrics["sigma_level"]
    delta = new_sigma - old["sigma"]
    print(
        f"  {name:<32} "
        f"{old['mean'] * 100:>7.2f}% ({old['sigma']:.2f}sigma) "
        f"{metrics['mean'] * 100:>7.2f}% ({new_sigma:.2f}sigma) "
        f"{delta:>+10.2f}sigma"
    )

print()
print("Key question for the consulting pitch:")
print("  Does narrowing 'any failure' -> 'HDF only' push sigma closer to 6?")
print()


# ---------------------------------------------------------------------------
# Step 8 - Verdict on the consulting pitch premise
# ---------------------------------------------------------------------------
banner("STEP 8: Verdict on the consulting pitch premise")

best_narrow_sigma = max(m["sigma_level"] for m in all_metrics.values())
best_narrow_model = max(all_metrics.keys(), key=lambda k: all_metrics[k]["sigma_level"])

print()
print(f"Best narrow-problem result: {best_narrow_model}")
print(f"  Sigma level: {best_narrow_sigma:.2f}sigma")
print(f"  Best broad-problem result: 3.35sigma (Gradient Boosting in Project 2b)")
print(f"  Sigma gain from narrowing: {best_narrow_sigma - 3.35:+.2f}sigma")
print()

if best_narrow_sigma >= 5.0:
    print("VERDICT: Narrow problem definition moves the needle significantly.")
    print("         Validates the consulting pitch premise: scope the problem narrowly")
    print("         as part of the path from 3.4sigma to 6sigma.")
elif best_narrow_sigma >= 4.0:
    print("VERDICT: Narrow problem definition gives a meaningful but not decisive gain.")
    print("         Useful as part of the toolkit but not sufficient alone.")
    print("         The 4-layer framework (FMEA + Poka-Yoke + Model + Eval) is still required.")
else:
    print("VERDICT: Narrow problem definition alone does NOT reach 6sigma.")
    print("         Even with a focused failure mode, the honest signal tops out around")
    print("         the same 3.4sigma ceiling observed in Project 2b.")
    print("         The path to 6sigma requires layered defenses (the 4-layer framework),")
    print("         not just better-scoped ML.")

print()
print("If the verdict is 'no improvement,' the honest message to clients is:")
print("  'Even when we focus the model on a single specific failure mode, honest")
print("   predictive ML tops out around 3-4sigma. Reaching 6sigma requires the")
print("   full 4-layer Operational Stability Infrastructure, not just a better model.'")
print("  That's the consulting pitch's actual selling point: the layered defenses,")
print("  not the ML model alone.")

print()
print("Done. Narrow-problem comparison complete.")
