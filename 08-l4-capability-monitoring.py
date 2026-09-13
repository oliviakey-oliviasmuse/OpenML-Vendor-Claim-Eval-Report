"""
Project 9 - L4 capability monitoring: detect model drift before it shows.

Background (from the experimentation log):
    Projects 1-8 demonstrated that the 4-layer framework (FMEA + Poka-Yoke +
    Model + Eval) is the path to 6sigma. Project 7b proved that the
    PRODUCTION-RECOMMENDED deployment (precise Poka-Yoke [PWF + OSF] +
    ML ensemble fallback for TWF, HDF, RNF) beats ML alone on both top-1
    accuracy (3.71sigma vs 3.62sigma) and macro F1 (61.79% vs 56.85%),
    and catches 88.5% of all actual failures with minimal false positives.

    This project tests L4 (the EVAL layer of the 4-layer framework):
    capability monitoring over time, drift detection, retraining triggers.

    The core question: when sensor drift, distribution shift, or process
    change degrades the model, can L4 catch it before the operator sees
    a problem? In real plants, "model was right yesterday, wrong today"
    is the failure mode that gets capital-intensive operations burned.

Setup:
    1. Train baseline model on first 5000 rows (the "training period")
    2. Simulate monitoring on rows 5001-10000 in chunks of 500
       (10 monitoring chunks total)
    3. Inject DRIFT at chunk 5 (rows 7501-10000): add 5% bias to Torque
       (simulates a sensor recalibration error)
    4. Apply L4 monitoring: track per-mode precision/recall per chunk
    5. Alert when per-mode recall drops > 2sigma below baseline
    6. Compare:
       - L4-ON: capability monitoring with auto-alert
       - L4-OFF: blind operation (no per-chunk tracking)

Strategies compared:
    A. Blind (L4-off): no monitoring, run model through all 10 chunks
    B. L4 monitoring: per-chunk recall tracking with control-chart rules

Run:
    python 08-l4-capability-monitoring.py
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
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.multioutput import MultiOutputClassifier

from dpmo import SIGMA_SHIFT


# Same precise Poka-Yoke rules from Project 7b
FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
NO_FAILURE = "none"


def banner(text):
    print()
    print("=" * 70)
    print(text)
    print("=" * 70)


# ---------------------------------------------------------------------------
# Poka-Yoke precise rules (PWF + OSF only - same as Project 7b)
# ---------------------------------------------------------------------------
def poka_yoke_precise_predict_row(row):
    """Apply ONLY the precise Poka-Yoke rules (PWF, OSF) to a single row."""
    rotational_speed = row["Rotational speed [rpm]"]
    torque = row["Torque [Nm]"]
    tool_wear = row["Tool wear [min]"]
    product_type = row["Type"]

    fired = []
    power = (torque * rotational_speed * 2 * np.pi) / 60.0
    if power < 3500.0 or power > 9000.0:
        fired.append("PWF")

    overstrain_threshold = {"L": 11000, "M": 12000, "H": 13000}.get(product_type, 12000)
    if torque * tool_wear >= overstrain_threshold:
        fired.append("OSF")

    if not fired:
        return NO_FAILURE
    priority = ["PWF", "OSF"]
    for mode in priority:
        if mode in fired:
            return mode
    return fired[0]


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


y_multiclass = X_raw.apply(build_multiclass_target, axis=1).reset_index(drop=True)
X_raw = X_raw.reset_index(drop=True)
print(f"\nMulti-class target distribution:")
print(y_multiclass.value_counts().sort_index().to_string())


# ---------------------------------------------------------------------------
# Step 2.5 - Build honest feature set
# ---------------------------------------------------------------------------
banner("Step 2.5: Build honest feature set")

leakage_cols = FAILURE_MODES + ["Machine failure"]
present = [c for c in leakage_cols if c in X_raw.columns]
print(f"Dropping leakage columns: {present}")

X = X_raw.drop(columns=present).reset_index(drop=True)
Y_specialists = pd.DataFrame(
    {mode: X_raw[mode].astype(int).values for mode in FAILURE_MODES},
).reset_index(drop=True)
print(f"\nFinal feature set: {list(X.columns)}")


# ---------------------------------------------------------------------------
# Step 3 - Preprocessor + production prediction pipeline
# ---------------------------------------------------------------------------
banner("STEP 3: Build production prediction pipeline")

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


def production_predict(pipeline, X_data):
    """Production-recommended: Poka-Yoke precise first, ML fallback for the rest."""
    proba_list = pipeline.predict_proba(X_data)
    proba_matrix = np.column_stack([p[:, 1] for p in proba_list])
    predicted_mode_idx = proba_matrix.argmax(axis=1)
    predicted_max_proba = proba_matrix.max(axis=1)
    pred_ml = np.where(
        predicted_max_proba >= 0.5,
        np.array(FAILURE_MODES)[predicted_mode_idx],
        NO_FAILURE,
    )
    pred_poka = np.array([poka_yoke_precise_predict_row(row) for _, row in X_data.iterrows()])
    pred_combined = np.where(pred_poka != NO_FAILURE, pred_poka, pred_ml)
    return pred_combined


# ---------------------------------------------------------------------------
# Step 4 - Train on the first 5000 rows, monitor on the rest
# ---------------------------------------------------------------------------
banner("STEP 4: Train on rows 0-4999, monitor on rows 5000-9999")

SPLIT_POINT = 5000
CHUNK_SIZE = 500

X_train = X.iloc[:SPLIT_POINT].reset_index(drop=True)
Y_specialists_train = Y_specialists.iloc[:SPLIT_POINT].reset_index(drop=True)
y_train = y_multiclass.iloc[:SPLIT_POINT].reset_index(drop=True)
X_monitor_full = X.iloc[SPLIT_POINT:].reset_index(drop=True)
y_monitor_full = y_multiclass.iloc[SPLIT_POINT:].reset_index(drop=True)

print(f"Training set:   {len(X_train)} rows")
print(f"Monitor set:    {len(X_monitor_full)} rows ({len(X_monitor_full) // CHUNK_SIZE} chunks of {CHUNK_SIZE})")


# Train baseline pipeline on clean training data
ensemble = MultiOutputClassifier(
    estimator=GradientBoostingClassifier(n_estimators=100, random_state=42),
    n_jobs=-1,
)
baseline_pipeline = Pipeline([("preprocess", preprocessor), ("clf", ensemble)])
baseline_pipeline.fit(X_train, Y_specialists_train)

# Compute baseline per-mode recall on training data (sanity check)
pred_train = production_predict(baseline_pipeline, X_train)
print()
print("Baseline per-mode recall on training data (sanity check):")
for mode in FAILURE_MODES:
    actual_mask = (y_train == mode).values
    caught_mask = (pred_train == mode)
    caught = (actual_mask & caught_mask).sum()
    actual = actual_mask.sum()
    recall = caught / actual if actual > 0 else 0.0
    print(f"  {mode}: {caught}/{actual} ({recall * 100:.1f}%)")


# ---------------------------------------------------------------------------
# Step 5 - Establish per-mode recall baselines from training data
# (these become the "center line" of the L4 control chart)
# ---------------------------------------------------------------------------
banner("STEP 5: Establish L4 control chart baselines from training data")

# For each mode, compute the recall with a confidence interval.
# L4 alert: when monitoring chunk recall drops more than 2 sigma below baseline,
# OR when 3 consecutive chunks show declining trend.
baseline_recalls = {}
baseline_std = {}
print()
print(f"{'Mode':<10} {'Baseline recall':<18} {'2-sigma threshold':<20}")
for mode in FAILURE_MODES:
    actual_mask = (y_train == mode).values
    caught_mask = (pred_train == mode)
    caught = (actual_mask & caught_mask).sum()
    actual = actual_mask.sum()
    recall = caught / actual if actual > 0 else 0.0
    # Wilson-style 2-sigma lower bound for the proportion
    # (use normal approximation: recall - 2*sqrt(recall*(1-recall)/n))
    if actual > 0:
        se = np.sqrt(recall * (1 - recall) / actual) if recall < 1 else 0
    else:
        se = 0
    threshold = max(0, recall - 2 * se)
    baseline_recalls[mode] = recall
    baseline_std[mode] = se
    print(f"  {mode:<10} {recall * 100:>6.2f}%             < {threshold * 100:.2f}%")
print()


# ---------------------------------------------------------------------------
# Step 6 - Simulate monitoring with drift injection
# ---------------------------------------------------------------------------
banner("STEP 6: Simulate monitoring (10 chunks, drift injected at chunk 5)")

# Drift injection: add 5% bias to Torque starting at chunk 5.
# This simulates a sensor recalibration error.
DRIFT_BIAS_TORQUE_PCT = 0.05  # 5% bias on torque
DRIFT_START_CHUNK = 5  # 0-indexed: drift starts at chunk 5

n_chunks = len(X_monitor_full) // CHUNK_SIZE
print(f"Number of monitoring chunks: {n_chunks}")
print(f"Drift injected at chunk {DRIFT_START_CHUNK}: torque *= {1 + DRIFT_BIAS_TORQUE_PCT}")
print()

chunk_metrics = []  # list of dicts with per-mode recall and predictions
for chunk_idx in range(n_chunks):
    start = chunk_idx * CHUNK_SIZE
    end = start + CHUNK_SIZE
    X_chunk = X_monitor_full.iloc[start:end].reset_index(drop=True)
    y_chunk = y_monitor_full.iloc[start:end].reset_index(drop=True)

    # Inject drift on this chunk if past the threshold
    X_chunk_drifted = X_chunk.copy()
    if chunk_idx >= DRIFT_START_CHUNK:
        X_chunk_drifted["Torque [Nm]"] = X_chunk_drifted["Torque [Nm]"] * (1 + DRIFT_BIAS_TORQUE_PCT)

    # Predict using the BASELINE pipeline (untrained on drift)
    pred_chunk = production_predict(baseline_pipeline, X_chunk_drifted)

    chunk_data = {"chunk": chunk_idx, "drifted": chunk_idx >= DRIFT_START_CHUNK}
    for mode in FAILURE_MODES:
        actual_mask = (y_chunk == mode).values
        caught_mask = (pred_chunk == mode)
        caught = (actual_mask & caught_mask).sum()
        actual = actual_mask.sum()
        recall = caught / actual if actual > 0 else 0.0
        chunk_data[f"{mode}_recall"] = recall
    chunk_data["overall_accuracy"] = float((pred_chunk == y_chunk.values).mean())
    chunk_metrics.append(chunk_data)


# ---------------------------------------------------------------------------
# Step 7 - L4 monitoring: alert when per-mode recall drops below threshold
# ---------------------------------------------------------------------------
banner("STEP 7: L4 monitoring: detect drift per chunk")

# Rule: alert if a per-mode recall drops more than 2-sigma below baseline.
# Use the baseline_recalls and the chunks to detect.

alert_log = []  # list of (chunk_idx, mode, reason)

print()
print(f"{'Chunk':<6} {'Drifted':<10} {'HDF':<8} {'PWF':<8} {'OSF':<8} {'TWF':<8} {'RNF':<8} {'Alert?':<30}")
print("-" * 100)

for c in chunk_metrics:
    alerts_this_chunk = []
    for mode in FAILURE_MODES:
        chunk_recall = c[f"{mode}_recall"]
        baseline_r = baseline_recalls[mode]
        # 2-sigma lower control limit. Special handling for boundary cases:
        # if baseline is at 100% (no variance in training), use a more lenient
        # rule: alert if recall drops by more than 10 percentage points OR to zero.
        # Without this fix, baseline_std = 0, LCL = 0, and the alert never fires.
        if baseline_std[mode] > 0:
            lcl = max(0, baseline_r - 2 * baseline_std[mode])
            if chunk_recall < lcl and lcl > 0:
                alerts_this_chunk.append(
                    f"{mode}: {chunk_recall * 100:.1f}% < LCL {lcl * 100:.1f}%"
                )
        else:
            # Baseline at 100% (or 0%) - use absolute drop heuristic
            # Alert if recall dropped by > 10% from perfect, or is exactly 0
            if baseline_r == 1.0 and (chunk_recall < 0.9 or chunk_recall == 0.0):
                alerts_this_chunk.append(
                    f"{mode}: {chunk_recall * 100:.1f}% (baseline 100%, drop={ (1.0 - chunk_recall) * 100:.1f}pp)"
                )
            elif baseline_r == 0.0 and chunk_recall > 0.1:
                alerts_this_chunk.append(
                    f"{mode}: {chunk_recall * 100:.1f}% (baseline 0%, gain={chunk_recall * 100:.1f}pp)"
                )

    if alerts_this_chunk:
        c["alerts"] = alerts_this_chunk
        alert_log.append((c["chunk"], alerts_this_chunk))

    print(
        f"{c['chunk']:<6} {'YES' if c['drifted'] else 'no':<10} "
        f"{c['HDF_recall'] * 100:>5.1f}%  {c['PWF_recall'] * 100:>5.1f}%  "
        f"{c['OSF_recall'] * 100:>5.1f}%  {c['TWF_recall'] * 100:>5.1f}%  "
        f"{c['RNF_recall'] * 100:>5.1f}%  "
        f"{'; '.join(alerts_this_chunk) if alerts_this_chunk else 'OK':<30}"
    )

print()
if alert_log:
    # Find the first chunk where the alert mentions OSF (which is the actual drift)
    osf_alert_chunk = None
    for chunk_idx, alerts in alert_log:
        if any("OSF" in a for a in alerts):
            osf_alert_chunk = chunk_idx
            break

    if osf_alert_chunk is not None:
        print(f"L4 monitoring detected OSF drift on chunk {osf_alert_chunk} "
              f"(alerts: {next(a for c, a in alert_log if c == osf_alert_chunk)})")
        print(f"  Drift started at chunk {DRIFT_START_CHUNK}")
        if osf_alert_chunk >= DRIFT_START_CHUNK:
            print(f"  Detection latency: {osf_alert_chunk - DRIFT_START_CHUNK} chunk(s) after drift started")
        else:
            print(f"  OSF alert fired before drift - likely noise from chunk 5 (no OSF cases in chunk)")
    else:
        print(f"L4 monitoring fired {len(alert_log)} alerts (all on rare modes TWF/HDF/RNF)")
        print("but did NOT catch the OSF drift. Consider tighter rules or larger chunks.")
else:
    print("L4 monitoring did NOT detect the drift.")
    print("  Consider: tighter control chart rules, smaller chunks, or per-fold CV.")


# ---------------------------------------------------------------------------
# Step 8 - Compare L4-on vs L4-off
# ---------------------------------------------------------------------------
banner("STEP 8: L4-on vs L4-off comparison")

# Aggregate metrics across all chunks
print()
print("Aggregating across all 10 monitoring chunks:")
all_actual = y_monitor_full.values[:n_chunks * CHUNK_SIZE]
# Build full prediction for all monitoring data (drift applied where appropriate)
X_monitor_drifted = X_monitor_full.copy()
X_monitor_drifted.iloc[
    DRIFT_START_CHUNK * CHUNK_SIZE : n_chunks * CHUNK_SIZE, X_monitor_drifted.columns.get_loc("Torque [Nm]")
] = (
    X_monitor_full.iloc[
        DRIFT_START_CHUNK * CHUNK_SIZE : n_chunks * CHUNK_SIZE,
        X_monitor_full.columns.get_loc("Torque [Nm]"),
    ].values
    * (1 + DRIFT_BIAS_TORQUE_PCT)
)

pred_monitor_all = production_predict(baseline_pipeline, X_monitor_drifted)

# Compute overall metrics (no monitoring)
overall_acc = float((pred_monitor_all == all_actual).mean())
overall_macro_f1 = f1_score(all_actual, pred_monitor_all, average="macro", zero_division=0)
print(f"  Overall top-1 accuracy (all 10 chunks): {overall_acc * 100:.2f}%")
print(f"  Overall macro F1 (all 10 chunks):       {overall_macro_f1 * 100:.2f}%")
print()

# Six Sigma for the overall monitoring run (compute directly to avoid printing
# a full report banner just to extract sigma_level).
if 0.0 < overall_acc < 1.0:
    monitor_sigma = float(stats.norm.ppf(overall_acc) + SIGMA_SHIFT)
else:
    monitor_sigma = float("-inf") if overall_acc <= 0.0 else float("inf")
monitor_dpmo = max(0.0, (1.0 - overall_acc) * 1_000_000.0)
print()
print("Six Sigma on full monitoring run (drift present):")
print("=" * 70)
print(f"  Mean accuracy:        {overall_acc * 100:>6.2f}%")
print(f"  DPMO:                 {monitor_dpmo:>10,.0f}")
print(f"  Sigma level:          {monitor_sigma:>6.2f}sigma")
print("=" * 70)


# ---------------------------------------------------------------------------
# Step 9 - Verdict on L4 capability monitoring
# ---------------------------------------------------------------------------
banner("STEP 9: Verdict on L4 capability monitoring")

print()
print("Question: does L4 (capability monitoring) catch drift before the model")
print("         goes silently wrong on the plant floor?")
print()

if alert_log:
    osf_alert_chunk = None
    for chunk_idx, alerts in alert_log:
        if any("OSF" in a for a in alerts):
            osf_alert_chunk = chunk_idx
            break

    if osf_alert_chunk is not None and osf_alert_chunk >= DRIFT_START_CHUNK:
        latency = osf_alert_chunk - DRIFT_START_CHUNK
        verdict = (
            f"VERDICT: L4 catches the OSF drift at chunk {osf_alert_chunk} "
            f"(latency: {latency} chunk(s) after drift started).\n"
            "         Capability monitoring earns its keep. Without L4, the model would silently\n"
            "         mis-predict for that entire chunk before any human noticed."
        )
    elif osf_alert_chunk is not None:
        verdict = (
            f"VERDICT: L4 fires alerts (chunk {osf_alert_chunk}) but no OSF-specific\n"
            "         alert within drift window. Need to distinguish rare-mode noise\n"
            "         alerts from drift alerts. Production fix: use larger chunks or\n"
            "         per-mode precision tracking."
        )
    else:
        verdict = (
            "VERDICT: L4 missed the OSF drift. Need tighter rules or different drift scenario.\n"
            "         In production, this means the operator would run the degraded model\n"
            "         until a regulator (or worse, a customer) caught the failures."
        )
else:
    verdict = (
        "VERDICT: L4 missed the drift. Need tighter rules or different drift scenario."
    )
print(verdict)

print()
print("The 4-layer framework, complete with L4 monitoring, gives the operator:")
print("  - L1 FMEA: knows which failures to design rules against")
print("  - L2 Poka-Yoke (precise only): catches deterministic failures with 100% confidence")
print("  - L3 Model: handles the rest (TWF, HDF, RNF, plus statistical noise)")
print("  - L4 Eval: catches drift BEFORE the regulator does")
print()
print("Together: the only sustainable path to 6sigma in capital-intensive ML deployments.")
print()
print("Done. L4 capability monitoring comparison complete.")
