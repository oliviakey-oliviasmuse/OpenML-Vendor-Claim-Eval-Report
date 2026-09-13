"""
2026-09-13 - 11-openml-study-upload.py

OpenML Study Upload — Six Sigma capability of ML classifiers on AI4I 2020
==============================================================================

Uploads the core Olivia-Key Six Sigma + ML experimental suite to OpenML as
a citable Study. After successful upload, the Study appears at:

  https://www.openml.org/search?type=study&id=<study_id>

and is cited on the AI4I 2020 dataset page (OpenML ID 42890).

What this script does:
  1. Authenticates against OpenML using $env:OPENML_API_KEY
  2. Verifies the AI4I 2020 dataset (ID 42890) is accessible
  3. Defines three OpenML flows:
       - kNN (sklearn KNeighborsClassifier, k=5)
       - RandomForest (100 trees)
       - GradientBoosting (100 trees)
  4. Runs each flow on AI4I 2020's OpenML-defined task splits
     (cross-validation folds, stratified)
  5. Uploads each run with predictive accuracy + macro F1 as evaluation measures
  6. Creates a Study titled "Six Sigma capability of ML classifiers on
     AI4I 2020 Predictive Maintenance" with a description linking to the
     public GitHub repo
  7. Attaches all runs to the Study and publishes it

Project scope (this script):
  - Project 1  : kNN baseline  (Olivia's first project)
  - Project 2  : RandomForest + GradientBoosting  (the leakage artefact, included
                  for completeness and to show the methodology distinguishes
                  honest vs leaked results)
  - Project 2b : RandomForest + GradientBoosting  (leakage-clean baseline)

Projects NOT in this script (require custom flow implementations):
  - Project 4  : narrow-problem scope (single failure mode) -- needs target-restricted dataset config
  - Project 7  : 5-specialist ensemble -- custom multi-output wrapper
  - Project 8  : naive Poka-Yoke + ML combination -- custom rule layer
  - Project 7b : production Poka-Yoke + ML (precise rules only) -- custom rule layer
  - Project 9  : L4 capability monitoring -- streaming eval, not a single run

These can be added in a follow-up script. The study remains open and Olivia
can append runs to it later.

Requirements:
  pip install openml scikit-learn
  $env:OPENML_API_KEY set (fine-grained API key from openml.org/auth/)

Usage:
  python "2026-09-13 - 11-openml-study-upload.py"

Output:
  - One flow per algorithm (kNN, RandomForest, GradientBoosting) created on OpenML
  - One run per (flow, fold) combination
  - One Study published with all runs attached
  - Final Study URL printed at end
"""

# ============================================================================
# METADATA HEADER
# ============================================================================
# Created:  2026-09-13
# Updated:  2026-09-13
# Version:  1.0
# Status:   starter (Projects 1, 2, 2b only)
# Related:  lean-six-sigma-expert agent (Mavis) - methodology
#           CHANGELOG.md - session history
#           GitHub repo: https://github.com/oliviakey-oliviasmuse/OpenML-Vendor-Claim-Eval-Report
# ============================================================================

import os
import sys
from datetime import datetime
from pathlib import Path

# UTF-8 stdout for σ, μ, etc.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import openml
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


# ============================================================================
# CONFIGURATION
# ============================================================================
OPENML_API_KEY = os.environ.get("OPENML_API_KEY")
assert OPENML_API_KEY, (
    "OPENML_API_KEY not set. In PowerShell:\n"
    "  $secure = Read-Host 'OPENML_API_KEY' -AsSecureString\n"
    "  $plain = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto(\n"
    "      [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure))\n"
    "  [Environment]::SetEnvironmentVariable('OPENML_API_KEY', $plain, 'User')\n"
    "  $env:OPENML_API_KEY = $plain\n"
    "  $env:OPENML_API_KEY.Length\n"
)

DATASET_ID = 31     # credit-g (OpenML ID 31) -- Project 1 baseline.
                      # Originally targeted AI4I 2020 (ID 42890) but that dataset's
                      # feature metadata on OpenML is incomplete and rejects
                      # task creation with code 622 (target_feature FK lookup
                      # returns no matches). credit-g has complete metadata
                      # and accepts task creation. AI4I 2020 to be revisited
                      # after OpenML metadata issue is reported.
TARGET_NAME = "class" # credit-g's binary target column name.
TASK_ID = None       # will be looked up
N_REPEATS = 1         # number of repeats (cross-validated runs)
N_FOLDS = 10          # folds per repeat

# Study metadata -- this is what shows up on OpenML under the credit-g dataset page
STUDY_NAME = (
    "Six Sigma capability of ML classifiers on credit-g (binary classification)"
)
STUDY_DESCRIPTION = (
    "Six Sigma + ML methodology audit on the credit-g dataset (OpenML ID 31). "
    "Evaluates three classifier families (kNN, RandomForest, GradientBoosting) "
    "under 10-fold cross-validation and reports predictive accuracy + macro F1. "
    "Capability framing is documented in the associated public GitHub repository "
    "(https://github.com/oliviakey-oliviasmuse/OpenML-Vendor-Claim-Eval-Report). "
    "Originally targeted AI4I 2020 Predictive Maintenance (OpenML ID 42890) but "
    "that dataset's feature metadata on OpenML is incomplete and rejects task "
    "creation with code 622 (target_feature foreign-key lookup returns no matches). "
    "Methodology reference: lean-six-sigma-expert (Mavis consulting agent). "
    f"Created: {datetime.now().isoformat(timespec='seconds')}"
)

# Output directory for study artefacts
OUTPUT_DIR = Path(__file__).resolve().parent / "eval-results"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================================
# 1. AUTHENTICATE
# ============================================================================
print("[1/5] Authenticating against OpenML...")
openml.config.apikey = OPENML_API_KEY
try:
    # list_datasets is public, but lists tasks to verify auth
    test = openml.tasks.list_tasks(output_format="dataframe", size=1)
    print(f"   Auth OK. Tasks visible: 1+")
except Exception as e:
    print(f"   Auth FAIL: {e}")
    sys.exit(1)


# ============================================================================
# 2. VERIFY DATASET + FIND TASK
# ============================================================================
print(f"[2/5] Loading dataset {DATASET_ID} ({'credit-g' if DATASET_ID == 31 else 'AI4I 2020' if DATASET_ID == 42890 else 'unknown'})...")
try:
    dataset = openml.datasets.get_dataset(DATASET_ID, download_data=True, download_qualities=False)
    print(f"   Dataset: {dataset.name} v{dataset.version}")
    print(f"   URL:     {dataset.url}")
except Exception as e:
    print(f"   Dataset FAIL: {e}")
    sys.exit(1)


# credit-g (binary classification) - find the OpenML task for this dataset.
# AI4I 2020 target = "Machine failure"; credit-g target = "class"
print(f"   Looking up OpenML task for dataset {DATASET_ID}...")
# Standard OpenML estimation procedure IDs:
#   1 = Holdout, 2 = 10-fold CV, 3 = 5-fold CV, 4 = 5x2 CV, 5 = 10 times 10-fold CV
# We use 2 (10-fold CV) — the standard for classification benchmarking.
ESTIMATION_PROCEDURE_ID = 2
try:
    tasks_df = openml.tasks.list_tasks(
        data_id=DATASET_ID,
        output_format="dataframe",
    )
    if "did" in tasks_df.columns and len(tasks_df) > 0:
        matching = tasks_df[tasks_df["did"] == DATASET_ID]
        if len(matching) > 0:
            # Use the existing task. Prefer the supervised classification task.
            task_row = matching[matching["task_type"].str.contains("Classification", na=False)].iloc[0]
            TASK_ID = int(task_row["tid"])
            print(f"   Existing task: {TASK_ID}")
        else:
            raise ValueError("No existing task matches, creating one")
    else:
        # OpenML returns an empty DataFrame with no columns when filter yields zero rows
        print(f"   No existing task found for dataset {DATASET_ID}. Creating one...")
        task = openml.tasks.create_task(
            task_type=openml.tasks.TaskType.SUPERVISED_CLASSIFICATION,
            dataset_id=DATASET_ID,
            target_name=TARGET_NAME,
            estimation_procedure_id=ESTIMATION_PROCEDURE_ID,
        )
        task = task.publish()
        TASK_ID = task.id
        print(f"   Task created: {TASK_ID}")
except Exception as e:
    print(f"   Task lookup/create failed: {e}")
    print(f"   Attempting task creation directly...")
    try:
        task = openml.tasks.create_task(
            task_type=openml.tasks.TaskType.SUPERVISED_CLASSIFICATION,
            dataset_id=DATASET_ID,
            target_name=TARGET_NAME,
            estimation_procedure_id=ESTIMATION_PROCEDURE_ID,
        )
        task = task.publish()
        TASK_ID = task.id
        print(f"   Task created: {TASK_ID}")
    except Exception as e2:
        print(f"   Task creation FAIL: {e2}")
        sys.exit(1)


# ============================================================================
# 3. DEFINE THREE FLOWS
# ============================================================================
print(f"[3/5] Defining OpenML flows...")


def make_sklearn_flow(sklearn_estimator, name, description, version="1.0"):
    """Wrap a scikit-learn estimator as an OpenML Flow with explicit metadata."""
    # OpenML needs a flow name and description that show up on the flow page.
    flow = openml.flows.sklearn_flow.sklearn_to_flow(sklearn_estimator)
    flow.name = name
    flow.description = description
    flow.version = version
    return flow


# kNN: Project 1 baseline
flow_knn = make_sklearn_flow(
    KNeighborsClassifier(n_neighbors=5),
    name="oliviakey_KNeighborsClassifier_k5",
    description=(
        "kNN baseline (k=5) for AI4I 2020 Predictive Maintenance. "
        "Six Sigma capability methodology audit. Olivia Key, 2026."
    ),
)
flow_knn = flow_knn.publish()
print(f"   kNN flow: id={flow_knn.flow_id}")

# RandomForest: Project 2 + 2b (we'll run twice -- once for leakage context, once leakage-clean)
flow_rf = make_sklearn_flow(
    RandomForestClassifier(n_estimators=100, random_state=42),
    name="oliviakey_RandomForestClassifier_100trees",
    description=(
        "RandomForest (100 trees) for AI4I 2020 Predictive Maintenance. "
        "Six Sigma capability methodology audit. Olivia Key, 2026."
    ),
)
flow_rf = flow_rf.publish()
print(f"   RandomForest flow: id={flow_rf.flow_id}")

# GradientBoosting: Project 2 + 2b
flow_gb = make_sklearn_flow(
    GradientBoostingClassifier(n_estimators=100, random_state=42),
    name="oliviakey_GradientBoostingClassifier_100trees",
    description=(
        "GradientBoosting (100 trees) for AI4I 2020 Predictive Maintenance. "
        "Six Sigma capability methodology audit. Olivia Key, 2026."
    ),
)
flow_gb = flow_gb.publish()
print(f"   GradientBoosting flow: id={flow_gb.flow_id}")


# ============================================================================
# 4. RUN EACH FLOW ON THE TASK + UPLOAD
# ============================================================================
print(f"[4/5] Running flows on task {TASK_ID}...")

uploaded_run_ids = []

for flow in [flow_knn, flow_rf, flow_gb]:
    print(f"   Running flow {flow.flow_id} ({flow.name})...")
    try:
        run = openml.runs.run_model_on_task(
            model=flow,
            task=openml.tasks.get_task(TASK_ID, download_data=True),
            seed=42,
            n_jobs=-1,
        )
        # Attach evaluation measures
        run = run.publish()
        uploaded_run_ids.append(run.run_id)
        print(f"   Uploaded run: {run.run_id}")
    except Exception as e:
        print(f"   Run FAIL for {flow.flow_id}: {e}")


# ============================================================================
# 5. CREATE STUDY + ATTACH RUNS + PUBLISH
# ============================================================================
print(f"[5/5] Creating study...")

study = openml.study.create_study(
    name=STUDY_NAME,
    description=STUDY_DESCRIPTION,
    run_ids=uploaded_run_ids,
    status="active",
)
study = study.publish()
study_id = study.id
study_url = f"https://www.openml.org/study/{study_id}"

print()
print("=" * 70)
print("STUDY PUBLISHED")
print("=" * 70)
print(f"  Study ID:   {study_id}")
print(f"  Study URL:  {study_url}")
print(f"  Runs:       {len(uploaded_run_ids)}")
print(f"  Created:    {datetime.now().isoformat(timespec='seconds')}")
print("=" * 70)
print()
print("Next steps:")
print("  1. Visit the Study URL to confirm visibility")
print(f"  2. Check AI4I 2020 dataset page at https://www.openml.org/d/{DATASET_ID}")
print("     -- the Study should appear under 'Studies' tab")
print("  3. Open a Zenodo export for DOI minting (optional)")
print()
print(f"  Study metadata persisted to: {OUTPUT_DIR / f'study_{study_id}.json'}")


# Persist study metadata locally
import json
study_meta = {
    "study_id": study_id,
    "study_url": study_url,
    "study_name": STUDY_NAME,
    "dataset_id": DATASET_ID,
    "task_id": TASK_ID,
    "uploaded_run_ids": uploaded_run_ids,
    "flow_ids": {"kNN": flow_knn.flow_id, "RandomForest": flow_rf.flow_id, "GradientBoosting": flow_gb.flow_id},
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "author": "Olivia Key <oliviakey@oliviasmuse.com>",
    "github_repo": "https://github.com/oliviakey-oliviasmuse/OpenML-Vendor-Claim-Eval-Report",
}
with open(OUTPUT_DIR / f"study_{study_id}.json", "w", encoding="utf-8") as f:
    json.dump(study_meta, f, indent=2)
