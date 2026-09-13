"""
Project 1 (v3) — Reproduce the OpenML intro's kNN claim on credit-g.

v1 used OpenML's run-publishing API (requires API key).
v2 used sklearn directly but failed: the credit-g dataset has 13 categorical
   columns, and kNN uses Euclidean distance which needs numeric input.
v3 (this file) wraps kNN in a sklearn Pipeline that one-hot encodes the
   categorical columns first. Same accuracy methodology, working pipeline.

The point of the exercise isn't the number. It's the audit trail.
Every step (dataset version, model class, hyperparameters, evaluation
procedure, preprocessing) is pinned down and reproducible.

Run:
    python 01-reproduce-credit-g-knn-sklearn.py
"""

import openml
from sklearn import neighbors
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from dpmo import report_six_sigma


def banner(text):
    print()
    print("=" * 70)
    print(text)
    print("=" * 70)


# ---------------------------------------------------------------------------
# Step 1 — Fetch the credit-g dataset from OpenML (no API key needed)
# ---------------------------------------------------------------------------
banner("STEP 1: Fetch credit-g dataset from OpenML")

dataset = openml.datasets.get_dataset("credit-g", download_data=True)
X, y, categorical_indicator, attribute_names = dataset.get_data(target="class")

n_cat = sum(categorical_indicator)
n_num = len(categorical_indicator) - n_cat

print(f"Dataset:           {dataset.name} (id={dataset.dataset_id})")
print(f"Shape:             X={X.shape}, y={y.shape}")
print(f"Target classes:    {sorted(y.unique())}")
print(f"Categorical cols:  {n_cat} of {len(categorical_indicator)}")
print(f"Numerical cols:    {n_num} of {len(categorical_indicator)}")


# ---------------------------------------------------------------------------
# Step 2 — Build the pipeline: one-hot encode categoricals, then kNN
# ---------------------------------------------------------------------------
banner("STEP 2: Build pipeline (one-hot encode categoricals, then kNN)")

cat_indices = [i for i, is_cat in enumerate(categorical_indicator) if is_cat]
num_indices = [i for i, is_cat in enumerate(categorical_indicator) if not is_cat]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_indices),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_indices),
    ]
)

pipe = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("clf", neighbors.KNeighborsClassifier(n_neighbors=5)),
    ]
)

print(f"Pipeline steps:")
print(f"  1. preprocess:  passthrough {n_num} numerical columns")
print(f"                   one-hot encode {n_cat} categorical columns")
print(f"  2. clf:          KNeighborsClassifier(n_neighbors=5)")


# ---------------------------------------------------------------------------
# Step 3 — 10-fold stratified cross-validation
# ---------------------------------------------------------------------------
banner("STEP 3: 10-fold stratified cross-validation")

scores = cross_val_score(pipe, X, y, cv=10, scoring="accuracy")
print("Per-fold scores:")
for i, s in enumerate(scores):
    print(f"  fold {i}: {s:.4f}  ({s * 100:.2f}%)")


# ---------------------------------------------------------------------------
# Step 4 — Report the result (Six Sigma format via dpmo)
# ---------------------------------------------------------------------------
banner("STEP 4: The result (Six Sigma format)")

metrics = report_six_sigma(
    scores,
    model_name="kNN (k=5)",
    dataset_name="credit-g (OpenML ID 31)",
)

mean_acc = metrics["mean"]
std_acc = metrics["std"]


# ---------------------------------------------------------------------------
# Step 5 — Compare to published claims
# ---------------------------------------------------------------------------
banner("STEP 5: How does this compare to published claims?")

print()
print(f"  Your kNN baseline (k=5, one-hot encoded):  {mean_acc * 100:5.2f}%")
print(f"  Vendor-style claim of '94% accurate':      94.00%")
print(f"  Realistic published range for kNN:         68-74%")
print(f"  Realistic published range for tuned")
print(f"    gradient boosting on credit-g:           77-82%")
print()
print("Lesson: a '94% accurate' claim without a pinned-down model class,")
print("dataset version, train/test split, evaluation procedure, AND")
print("preprocessing pipeline is not a number - it's marketing.")
print()
print("The 4-layer framework calls this L4 Eval. The five things that must")
print("be pinned down before any accuracy claim is citable:")
print("  1. Dataset version (OpenML ID handles this)")
print("  2. Model class + hyperparameters")
print("  3. Train/test split (10-fold stratified here)")
print("  4. Evaluation metric (accuracy)")
print("  5. Preprocessing pipeline (one-hot encoding here)")
print()
print("Six Sigma framing (per dpmo.py):")
print(f"  Sigma level: {metrics['sigma_level']:.2f}σ, DPMO: {metrics['dpmo']:,.0f}, "
      f"Cpk: {metrics['cpk']:.2f}")
print(f"  {metrics['sigma_level']:.2f}σ is below the 6σ bar. A '94% accurate'")
print(f"  vendor claim at ~3.4σ is also below the bar. Neither passes.")


# ---------------------------------------------------------------------------
# Optional — set an API key later to publish this run
# ---------------------------------------------------------------------------
banner("OPTIONAL: Publish the run later (requires an OpenML API key)")

print()
print("When you're ready to publish:")
print("  1. Sign up free at https://www.openml.org")
print("  2. Get your API key from your profile page")
print("  3. Either:")
print("     - Set environment variable:  $env:OPENML_API_KEY = 'YOUR_KEY'")
print("     - Or in Python:              openml.config.apikey = 'YOUR_KEY'")
print("  4. Then we can switch this script back to the run-publishing flow.")
print()
print("For Project #1, the unadjusted accuracy is the lesson. Publishing")
print("comes in Project #3, where the audit trail matters more.")
print()
print("Done. You have a citable, reproducible baseline.")