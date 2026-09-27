"""LightGBM Pairwise Match Classifier Training Module for Business Entity Resolution.

Trains a classical LightGBM gradient boosted decision tree classifier (MIT licensed,
<< 8B parameters) on pairwise similarity features:
- Uses training-fit fold for training and validation fold for evaluation.
- Evaluates standard metrics (Precision, Recall, F1) at default 0.5 probability threshold.
- Extracts and visualizes feature importances (split gain & weight) to analyze predictive signals.
- Saves the trained model to disk for downstream inference and submission generation.
"""

import gc
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

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
    TRAIN_FIT_FEATURES,
    RANDOM_SEED,
)

# Model artifact paths
MODELS_DIR = OUTPUT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = MODELS_DIR / "lgbm_classifier.txt"
MODEL_PKL_PATH = MODELS_DIR / "lgbm_classifier.pkl"

# Feature column definitions
FEATURE_COLUMNS = [
    # Name features
    "name_exact_match",
    "name_clean_exact_match",
    "name_levenshtein_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_jaccard_similarity",
    "name_tfidf_cosine",
    "name_char_len_diff",
    # Address features
    "address_exact_match",
    "address_levenshtein_ratio",
    "address_token_sort_ratio",
    "address_token_set_ratio",
    "address_jaccard_similarity",
    "address_char_len_diff",
    "postal_code_match",
    "landmark_overlap",
    # Cross & provenance features
    "same_country",
    "candidate_source",
]


def train_lightgbm_classifier(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str] = FEATURE_COLUMNS,
    target_col: str = "label",
    seed: int = RANDOM_SEED,
) -> Tuple[lgb.LGBMClassifier, Dict[str, Any]]:
    """Train LightGBM binary classifier and evaluate on validation features."""
    print("=" * 80)
    print("         STAGE 4: TRAINING LIGHTGBM PAIRWISE MATCH CLASSIFIER")
    print("=" * 80)
    print(f"Number of Features:  {len(feature_cols)}")
    print(f"Training Pairs:      {len(train_df):,}")
    print(f"Validation Pairs:    {len(val_df):,}")
    print(f"Random Seed:         {seed}\n")

    # Prepare feature matrices
    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].values

    X_val = val_df[feature_cols].copy()
    y_val = val_df[target_col].values

    pos_train = int(np.sum(y_train == 1))
    neg_train = int(np.sum(y_train == 0))
    scale_pos = (neg_train / max(1, pos_train))

    print(f"Training Label Distribution:")
    print(f"  - Positive (1): {pos_train:,} ({pos_train / len(y_train) * 100:.2f}%)")
    print(f"  - Negative (0): {neg_train:,} ({neg_train / len(y_train) * 100:.2f}%)")
    print(f"  - Class Weight Scale (scale_pos_weight): {scale_pos:.2f}\n")

    # LightGBM Classifier Configuration (MIT Licensed, classical ML)
    clf = lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=min(scale_pos, 10.0),  # Moderated scale_pos_weight to preserve precision
        random_state=seed,
        n_jobs=-1,
        importance_type="gain",
    )

    print("Fitting LightGBM model with early stopping on validation fold...", flush=True)
    t0 = time.time()
    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_names=["train", "val"],
        eval_metric=["binary_logloss", "auc"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=50),
        ],
    )
    train_duration = time.time() - t0
    print(f"\nModel training finished in {train_duration:.2f}s (Best iteration: {clf.best_iteration_}).", flush=True)

    # Predictions on Validation Fold
    print("\n--- Evaluating Model Performance on Validation Fold ---", flush=True)
    val_probs = clf.predict_proba(X_val)[:, 1]
    val_preds_default = (val_probs >= 0.5).astype(int)

    precision_05 = precision_score(y_val, val_preds_default, zero_division=0)
    recall_05 = recall_score(y_val, val_preds_default, zero_division=0)
    f1_05 = f1_score(y_val, val_preds_default, zero_division=0)
    auc_score = roc_auc_score(y_val, val_probs)

    print(f"Validation Metrics (Default Threshold = 0.50):")
    print(f"  - Precision: {precision_05 * 100:.2f}%")
    print(f"  - Recall:    {recall_05 * 100:.2f}%")
    print(f"  - F1-Score:  {f1_05 * 100:.2f}%")
    print(f"  - ROC-AUC:   {auc_score:.4f}")

    print("\nClassification Report (Threshold = 0.50):")
    print(classification_report(y_val, val_preds_default, digits=4))

    cm = confusion_matrix(y_val, val_preds_default)
    print("Confusion Matrix:")
    print(f"  [TN: {cm[0,0]:,}\tFP: {cm[0,1]:,}]")
    print(f"  [FN: {cm[1,0]:,}\tTP: {cm[1,1]:,}]")

    # Feature Importances Analysis
    print("\n" + "=" * 80)
    print("                     FEATURE IMPORTANCES (GAIN)")
    print("=" * 80)
    importances_gain = clf.booster_.feature_importance(importance_type="gain")
    importances_split = clf.booster_.feature_importance(importance_type="split")

    imp_df = pd.DataFrame({
        "Feature": feature_cols,
        "Gain Importance": importances_gain,
        "Split Count": importances_split,
    }).sort_values(by="Gain Importance", ascending=False)

    # Normalize gain to percentages
    total_gain = imp_df["Gain Importance"].sum()
    imp_df["Relative Gain %"] = (imp_df["Gain Importance"] / total_gain * 100).map("{:.2f}%".format)
    print(imp_df.to_string(index=False))
    print("=" * 80)

    # Save Trained Model to Disk
    print(f"\nSaving model to disk...")
    clf.booster_.save_model(str(MODEL_SAVE_PATH))
    joblib.dump(clf, MODEL_PKL_PATH)
    print(f"  - Saved LightGBM booster: {MODEL_SAVE_PATH}")
    print(f"  - Saved scikit-learn model: {MODEL_PKL_PATH}")

    metrics = {
        "precision_05": precision_05,
        "recall_05": recall_05,
        "f1_05": f1_05,
        "roc_auc": auc_score,
        "best_iteration": clf.best_iteration_,
        "train_time_s": train_duration,
        "feature_importances": imp_df,
    }

    return clf, metrics


def main():
    print("=" * 80)
    print("         LIGHTGBM TRAINING PIPELINE INITIALIZATION")
    print("=" * 80)

    if not VAL_FEATURES.exists():
        print(f"Validation feature table not found at {VAL_FEATURES}.")
        print("Please ensure candidate generation (src/blocking.py) and feature computation (src/features.py) have completed.")
        return

    print("Loading feature tables...", flush=True)
    val_feat = pd.read_parquet(VAL_FEATURES)
    print(f"Loaded {len(val_feat):,} validation candidate feature rows.", flush=True)

    # If TRAIN_FIT_FEATURES exists, use it; otherwise create a stratified 80/20 train/val split of labeled candidate pairs
    if TRAIN_FIT_FEATURES.exists():
        train_feat = pd.read_parquet(TRAIN_FIT_FEATURES)
        print(f"Loaded {len(train_feat):,} train-fit candidate feature rows.", flush=True)
    else:
        print("Splitting available labeled candidate pairs into train-fit fold (75%) and evaluation fold (25%)...", flush=True)
        from sklearn.model_selection import train_test_split
        train_feat, val_eval = train_test_split(
            val_feat,
            test_size=0.25,
            random_state=RANDOM_SEED,
            stratify=val_feat["label"],
        )
        val_feat = val_eval
        print(f"Train-fit pairs: {len(train_feat):,}, Validation pairs: {len(val_feat):,}", flush=True)

    # Train model
    clf, metrics = train_lightgbm_classifier(train_feat, val_feat)

    print("\nTraining and evaluation successfully completed.")
    print("=" * 80)


if __name__ == "__main__":
    main()
