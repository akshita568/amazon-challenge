"""Threshold Tuning & Competition Metric Evaluation Module.

Evaluates predicted probabilities from the LightGBM classifier on the validation fold
against the exact competition metric:
- Per-Source-1-entity F_0.5 score:
    F_0.5 = (1.25 * precision * recall) / (0.25 * precision + recall)
    - If true matches == 0 (singleton): scores 1.0 if predicted empty, 0.0 if any match predicted.
    - If true matches > 0 and predicted empty: scores 0.0.
- Macro-average across all Source 1 entities in the validation set.
- Scans threshold range from 0.30 to 0.90 (in steps of 0.02).
- Compares:
    1. Standard thresholding: Keep all candidate pairs with prob >= threshold.
    2. Mutual top candidate tie-breaking: Only retain matches where candidate considers S1 its top match.
- Reports winning threshold and strategy recommendation.
"""

import gc
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

# Ensure standard output can print Unicode characters safely on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import (
    OUTPUT_DIR,
    FEATURES_DIR,
    VAL_FEATURES,
    VAL_GROUND_TRUTH,
    RANDOM_SEED,
)
from src.train import FEATURE_COLUMNS, MODEL_SAVE_PATH, MODEL_PKL_PATH


def compute_entity_f05(pred_set: Set[str], true_set: Set[str]) -> float:
    """Compute per-Source-1-entity F_0.5 score adhering to exact competition rules.

    - If true_set is empty (singleton):
        returns 1.0 if pred_set is empty, else 0.0.
    - If true_set is non-empty:
        returns 0.0 if pred_set is empty or hits == 0.
        else (1.25 * precision * recall) / (0.25 * precision + recall).
    """
    if len(true_set) == 0:
        return 1.0 if len(pred_set) == 0 else 0.0

    if len(pred_set) == 0:
        return 0.0

    hits = len(pred_set.intersection(true_set))
    if hits == 0:
        return 0.0

    prec = hits / len(pred_set)
    rec = hits / len(true_set)
    denom = (0.25 * prec) + rec
    if denom == 0:
        return 0.0
    return (1.25 * prec * rec) / denom


def evaluate_macro_f05(
    pred_dict: Dict[str, Set[str]],
    true_dict: Dict[str, Set[str]],
    all_s1_ids: List[str],
) -> float:
    """Compute macro-average F_0.5 across all validation Source 1 entities."""
    scores = []
    for s1_id in all_s1_ids:
        preds = pred_dict.get(s1_id, set())
        trues = true_dict.get(s1_id, set())
        scores.append(compute_entity_f05(preds, trues))
    return float(np.mean(scores))


def run_threshold_tuning(
    df_val_preds: pd.DataFrame,
    df_val_gt: pd.DataFrame,
    thresholds: np.ndarray = np.arange(0.30, 0.92, 0.02),
) -> pd.DataFrame:
    """Evaluate Macro F_0.5 across threshold grid for standard and mutual-tiebreak strategies."""
    print("=" * 80)
    print("         STAGE 5: THRESHOLD TUNING & COMPETITION METRIC EVALUATION")
    print("=" * 80)
    print(f"Candidate Pairs to Score: {len(df_val_preds):,}")
    print(f"Validation Reference Entities: {len(df_val_gt):,}")
    print(f"Threshold Grid: {thresholds[0]:.2f} to {thresholds[-1]:.2f} (step: 0.02)\n")

    # Build ground truth dictionary: s1_id -> set of true matches
    true_dict: Dict[str, Set[str]] = {}
    all_s1_ids = df_val_gt["source1_entity_id"].tolist()
    for _, row in df_val_gt.iterrows():
        s1 = row["source1_entity_id"]
        matches_str = str(row["matched_entity_ids"]).strip()
        if not matches_str or matches_str == "nan":
            true_dict[s1] = set()
        else:
            true_dict[s1] = {m.strip() for m in matches_str.split(",") if m.strip()}

    # Compute mutual top candidate map:
    # For each candidate, identify the Source 1 entity with the highest probability
    print("Computing mutual top candidate lookup...", flush=True)
    best_s1_for_cand: Dict[str, Tuple[str, float]] = {}
    for s1, cand, prob in zip(df_val_preds["source1_entity_id"],
                              df_val_preds["candidate_entity_id"],
                              df_val_preds["prob"]):
        if cand not in best_s1_for_cand or prob > best_s1_for_cand[cand][1]:
            best_s1_for_cand[cand] = (s1, prob)

    # Grid search across thresholds
    print("Evaluating Macro F_0.5 across threshold grid...", flush=True)
    t0 = time.time()
    results = []

    s1_series = df_val_preds["source1_entity_id"].values
    cand_series = df_val_preds["candidate_entity_id"].values
    prob_series = df_val_preds["prob"].values

    for thresh in thresholds:
        thresh = round(float(thresh), 3)

        # Strategy A: All candidates >= threshold
        mask_a = prob_series >= thresh
        pred_dict_a: Dict[str, Set[str]] = defaultdict(set)
        for s1, cand in zip(s1_series[mask_a], cand_series[mask_a]):
            pred_dict_a[s1].add(cand)

        f05_standard = evaluate_macro_f05(pred_dict_a, true_dict, all_s1_ids)

        # Strategy B: Mutual top candidate tie-breaking
        # Keep only if prob >= thresh AND cand's highest scoring S1 is this S1
        pred_dict_b: Dict[str, Set[str]] = defaultdict(set)
        for s1, cand, prob in zip(s1_series[mask_a], cand_series[mask_a], prob_series[mask_a]):
            if best_s1_for_cand[cand][0] == s1:
                pred_dict_b[s1].add(cand)

        f05_mutual = evaluate_macro_f05(pred_dict_b, true_dict, all_s1_ids)

        # Count total predictions
        n_preds_a = sum(len(v) for v in pred_dict_a.values())
        n_preds_b = sum(len(v) for v in pred_dict_b.values())

        results.append({
            "Threshold": thresh,
            "Macro F_0.5 (Standard)": f05_standard,
            "Predictions (Standard)": n_preds_a,
            "Macro F_0.5 (Mutual Top)": f05_mutual,
            "Predictions (Mutual Top)": n_preds_b,
        })

    eval_duration = time.time() - t0
    res_df = pd.DataFrame(results)

    # Display results table
    print(f"\nCompleted evaluation in {eval_duration:.2f}s.\n")
    print(res_df.to_string(index=False, float_format="{:.4f}".format))

    # Find best thresholds
    best_row_std = res_df.loc[res_df["Macro F_0.5 (Standard)"].idxmax()]
    best_row_mut = res_df.loc[res_df["Macro F_0.5 (Mutual Top)"].idxmax()]

    print("\n" + "=" * 80)
    print("                      OPTIMIZATION SUMMARY")
    print("=" * 80)
    print(f"Strategy A (Standard Thresholding):")
    print(f"  - Best Threshold:    {best_row_std['Threshold']:.2f}")
    print(f"  - Best Macro F_0.5:  {best_row_std['Macro F_0.5 (Standard)']:.4f}")
    print(f"  - Total Predictions: {int(best_row_std['Predictions (Standard)']):,}")
    print()
    print(f"Strategy B (Mutual Top Tie-Breaking):")
    print(f"  - Best Threshold:    {best_row_mut['Threshold']:.2f}")
    print(f"  - Best Macro F_0.5:  {best_row_mut['Macro F_0.5 (Mutual Top)']:.4f}")
    print(f"  - Total Predictions: {int(best_row_mut['Predictions (Mutual Top)']):,}")
    print("=" * 80)

    # Recommendation
    if best_row_mut["Macro F_0.5 (Mutual Top)"] >= best_row_std["Macro F_0.5 (Standard)"]:
        rec_strat = "Mutual Top Candidate Tie-Breaking"
        rec_thresh = best_row_mut["Threshold"]
        rec_score = best_row_mut["Macro F_0.5 (Mutual Top)"]
    else:
        rec_strat = "Standard Thresholding"
        rec_thresh = best_row_std["Threshold"]
        rec_score = best_row_std["Macro F_0.5 (Standard)"]

    print(f"\n[RECOMMENDATION]: Use '{rec_strat}' with threshold = {rec_thresh:.2f} (Macro F_0.5 = {rec_score:.4f}).")
    if rec_strat == "Mutual Top Candidate Tie-Breaking":
        print("Reason: F_0.5 is precision-heavy (beta=0.5). Mutual tie-breaking eliminates conflicting duplicate")
        print("candidate matches, preventing severe precision penalties without sacrificing true positives.")

    # Save tuning summary
    save_path = OUTPUT_DIR / "threshold_tuning.csv"
    res_df.to_csv(save_path, index=False)
    print(f"\nSaved tuning table to {save_path}.")

    return res_df


def main():
    print("Loading validation candidates and features...", flush=True)
    if not VAL_FEATURES.exists():
        print(f"Features file not found at {VAL_FEATURES}. Run src/features.py first.")
        return

    val_df = pd.read_parquet(VAL_FEATURES)
    print(f"Loaded {len(val_df):,} validation feature pairs.")

    # Load trained model
    if not MODEL_PKL_PATH.exists() and not MODEL_SAVE_PATH.exists():
        print("Trained model not found. Run src/train.py first.")
        return

    if MODEL_PKL_PATH.exists():
        clf = joblib.load(MODEL_PKL_PATH)
    else:
        clf = lgb.Booster(model_file=str(MODEL_SAVE_PATH))

    # Predict probabilities
    print("Scoring validation pairs with trained LightGBM model...", flush=True)
    X = val_df[FEATURE_COLUMNS]
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X)[:, 1]
    else:
        probs = clf.predict(X)

    val_df["prob"] = probs

    # Load validation ground truth
    val_gt = pd.read_csv(VAL_GROUND_TRUTH, sep="\t")

    # Run threshold tuning
    run_threshold_tuning(val_df, val_gt)


if __name__ == "__main__":
    main()
