"""
Project 1 — Reproduce the OpenML intro's kNN claim on credit-g.

The OpenML intro page (https://docs.openml.org/intro/) uses
KNeighborsClassifier(n_neighbors=5) as its canonical example on task 31
(supervised classification on credit-g, the German Credit dataset).

This script runs that exact flow on that exact task and reports the
accuracy. Published results for kNN on credit-g typically land in the
68-74% range — far from the "94% accurate" claims often seen in vendor
marketing material.

The point of the exercise isn't the number. It's the audit trail.
Every step (dataset version, task splits, model class, hyperparameters,
evaluation procedure) is pinned down and reproducible.

Run:
    python 01-reproduce-credit-g-knn.py
"""

import openml


def banner(text):
    print()
    print("=" * 70)
    print(text)
    print("=" * 70)


# ---------------------------------------------------------------------------
# Step 1 — Fetch task 31 (supervised classification on credit-g)
# ---------------------------------------------------------------------------
banner("STEP 1: Fetch task 31 (supervised classification on credit-g)")

task = openml.tasks.get_task(31)
dataset = task.get_dataset()

print(f"Task ID:           {task.task_id}")
print(f"Task type:         {task.task_type}")
print(f"Dataset:           {dataset.name} (id={dataset.dataset_id})")
print(f"Estimation proc:   {task.estimation_procedure}")


# ---------------------------------------------------------------------------
# Step 2 — Define the flow (kNN, n_neighbors=5)
#           This matches the OpenML intro documentation exactly.
# ---------------------------------------------------------------------------
banner("STEP 2: Define the flow (kNN, n_neighbors=5) — matches OpenML intro")

from sklearn import neighbors

clf = neighbors.KNeighborsClassifier(n_neighbors=5)
print(f"Classifier: {clf}")


# ---------------------------------------------------------------------------
# Step 3 — Run the model on the task
# ---------------------------------------------------------------------------
banner("STEP 3: Run model on task (10-fold cross-validation by default)")

run = openml.runs.run_model_on_task(clf, task)
print(f"Run ID: {run.run_id}")


# ---------------------------------------------------------------------------
# Step 4 — Report the result
# ---------------------------------------------------------------------------
banner("STEP 4: The result")

score = run.score  # default metric is predictive accuracy
print(f"Mean accuracy (10-fold CV): {score:.4f}  =  {score * 100:.2f}%")


# ---------------------------------------------------------------------------
# Step 5 — Compare to published claims
# ---------------------------------------------------------------------------
banner("STEP 5: How does this compare to published claims?")

print()
print(f"  Your result (kNN, k=5):                    {score * 100:5.2f}%")
print(f"  Vendor-style claim of '94% accurate':      94.00%")
print(f"  Realistic published range for kNN:          68-74%")
print(f"  Realistic published range for tuned")
print(f"    gradient boosting on credit-g:            77-82%")
print()
print("Lesson: a '94% accurate' claim without a pinned-down model class,")
print("dataset version, train/test split, and evaluation procedure is not")
print("a number — it's marketing. The 4-layer framework calls this L4 Eval.")
print("OpenML gives you the audit trail for free if you use it.")


# ---------------------------------------------------------------------------
# Optional — Publish the run so it's citable
# ---------------------------------------------------------------------------
banner("OPTIONAL: Publish the run (requires an OpenML API key)")

# To publish, first set your API key in one of two ways:
#   1. Environment variable:  $env:OPENML_API_KEY = "YOUR_KEY"
#   2. In this script:        openml.config.apikey = "YOUR_KEY"
#
# Then uncomment the lines below:
#
# openml.config.apikey = "YOUR_KEY"
# published_run = run.publish()
# print(f"Published. Run URL: {published_run.openml_url}")
#
# Once published, you can cite the run in a blog post, a client deck, or
# your own consulting notes. It's a permanent, citable artefact.

print()
print("Done. You have a citable, reproducible baseline.")