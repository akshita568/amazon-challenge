"""Validation Split Module for Business Entity Resolution ML Challenge.

Creates a reproducible, country-stratified train/validation split from Source 1 entities:
- 80% train_ground_truth_fit (and train_source1_fit)
- 20% val_ground_truth (and val_source1)
Saves all splits to disk under output/splits/.
"""

import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import (
    TRAIN_SOURCE1,
    TRAIN_GROUND_TRUTH,
    TRAIN_GROUND_TRUTH_FIT,
    VAL_GROUND_TRUTH,
    TRAIN_SOURCE1_FIT,
    VAL_SOURCE1,
    RANDOM_SEED,
    SPLITS_DIR,
)


def create_validation_split(val_fraction: float = 0.20, seed: int = RANDOM_SEED):
    """Create reproducible country-stratified train_fit and val splits.

    Args:
        val_fraction: Proportion of Source 1 entities to hold out (default: 0.20 / 20%).
        seed: Random seed for reproducibility (default: 42).
    """
    print("=" * 80)
    print("      CREATING VALIDATION SPLIT (STRATIFIED BY COUNTRY)")
    print("=" * 80)
    print(f"Validation Fraction: {val_fraction * 100:.1f}%")
    print(f"Random Seed:         {seed}")
    print(f"Output Directory:    {SPLITS_DIR}\n")

    # 1. Load Source 1
    print("Loading Source 1 reference entities...")
    df_s1 = pd.read_csv(TRAIN_SOURCE1, sep="\t", low_memory=False)
    total_s1 = len(df_s1)
    print(f"Total Source 1 entities: {total_s1:,}")

    # Check country distribution for stratification
    country_counts = df_s1["country"].value_counts(dropna=False)
    print("Source 1 Country Distribution:")
    for country, count in country_counts.items():
        print(f"  - {country}: {count:,} ({count / total_s1 * 100:.2f}%)")

    # Handle any null countries for stratification by filling a placeholder
    stratify_target = df_s1["country"].fillna("UNKNOWN")

    # 2. Perform stratified train/val split on Source 1 entities
    print(f"\nPerforming stratified split (test_size={val_fraction}, random_state={seed})...")
    s1_fit, s1_val = train_test_split(
        df_s1,
        test_size=val_fraction,
        random_state=seed,
        stratify=stratify_target,
    )

    print(f"Split results for Source 1:")
    print(f"  - Train Fit Source 1: {len(s1_fit):,} entities ({len(s1_fit) / total_s1 * 100:.2f}%)")
    print(f"  - Val Source 1:       {len(s1_val):,} entities ({len(s1_val) / total_s1 * 100:.2f}%)")

    # Verify country distribution in both splits
    print("\nCountry Distribution in Train Fit:")
    for country, count in s1_fit["country"].value_counts().items():
        print(f"  - {country}: {count:,} ({count / len(s1_fit) * 100:.2f}%)")

    print("\nCountry Distribution in Validation:")
    for country, count in s1_val["country"].value_counts().items():
        print(f"  - {country}: {count:,} ({count / len(s1_val) * 100:.2f}%)")

    # 3. Load and Split Ground Truth
    print("\nLoading train ground truth...")
    df_gt = pd.read_csv(TRAIN_GROUND_TRUTH, sep="\t", low_memory=False)
    total_gt = len(df_gt)
    print(f"Total ground truth rows: {total_gt:,}")

    # Index ground truth by source1_entity_id for fast lookup
    df_gt.set_index("source1_entity_id", inplace=True)

    # Align with splits
    val_ids = set(s1_val["entity_id"])
    fit_ids = set(s1_fit["entity_id"])

    print("Aligning ground truth records with Source 1 splits...")
    gt_val = df_gt.loc[df_gt.index.isin(val_ids)].reset_index()
    gt_fit = df_gt.loc[df_gt.index.isin(fit_ids)].reset_index()

    # 4. Rigorous integrity verifications
    print("\nPerforming integrity verifications:")
    assert len(s1_val) + len(s1_fit) == total_s1, "Source 1 count mismatch!"
    assert len(gt_val) + len(gt_fit) == total_gt, "Ground truth count mismatch!"
    assert len(val_ids.intersection(fit_ids)) == 0, "Data leakage detected: overlapping entity IDs!"
    print("  [PASSED] Counts match exactly.")
    print("  [PASSED] Zero entity ID leakage between Train Fit and Val splits.")

    # Check singleton rates in both splits
    val_singletons = (gt_val["matched_entity_ids"].isna() | (gt_val["matched_entity_ids"].str.strip() == "")).sum()
    fit_singletons = (gt_fit["matched_entity_ids"].isna() | (gt_fit["matched_entity_ids"].str.strip() == "")).sum()
    print(f"  Singleton rate in Train Fit: {fit_singletons:,} ({fit_singletons / len(gt_fit) * 100:.2f}%)")
    print(f"  Singleton rate in Val:       {val_singletons:,} ({val_singletons / len(gt_val) * 100:.2f}%)")

    # 5. Save splits to disk
    print("\nSaving split files to disk...")
    # Ground truth splits
    gt_fit.to_csv(TRAIN_GROUND_TRUTH_FIT, sep="\t", index=False)
    print(f"  - Saved train_ground_truth_fit: {TRAIN_GROUND_TRUTH_FIT} ({len(gt_fit):,} rows)")

    gt_val.to_csv(VAL_GROUND_TRUTH, sep="\t", index=False)
    print(f"  - Saved val_ground_truth:       {VAL_GROUND_TRUTH} ({len(gt_val):,} rows)")

    # Source 1 entity splits
    s1_fit.to_csv(TRAIN_SOURCE1_FIT, sep="\t", index=False)
    print(f"  - Saved train_source1_fit:      {TRAIN_SOURCE1_FIT} ({len(s1_fit):,} rows)")

    s1_val.to_csv(VAL_SOURCE1, sep="\t", index=False)
    print(f"  - Saved val_source1:            {VAL_SOURCE1} ({len(s1_val):,} rows)")

    print("\nValidation split successfully created and saved.")
    print("=" * 80)


if __name__ == "__main__":
    create_validation_split(val_fraction=0.20, seed=RANDOM_SEED)
