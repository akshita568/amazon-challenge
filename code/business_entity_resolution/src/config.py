"""Configuration module defining dataset and output paths for the Business Entity Resolution ML challenge."""

from pathlib import Path

# Base directories
SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
CODE_DIR = PROJECT_DIR.parent
ROOT_DIR = CODE_DIR.parent

OUTPUT_DIR = PROJECT_DIR / "output"
DATASET_DIR = ROOT_DIR / "dataset"

TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

# Train file paths
TRAIN_SOURCE1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3 = TRAIN_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"

# Test file paths
TEST_SOURCE1 = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2 = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3 = TEST_DIR / "test_source3.tsv"

# File path mapping dictionaries for easy programmatic access
TRAIN_FILES = {
    "source1": TRAIN_SOURCE1,
    "source2": TRAIN_SOURCE2,
    "source3": TRAIN_SOURCE3,
    "ground_truth": TRAIN_GROUND_TRUTH,
}

TEST_FILES = {
    "source1": TEST_SOURCE1,
    "source2": TEST_SOURCE2,
    "source3": TEST_SOURCE3,
}

ALL_FILES = {
    **{f"train_{k}": v for k, v in TRAIN_FILES.items()},
    **{f"test_{k}": v for k, v in TEST_FILES.items()},
}

# Validation and train fit split paths
SPLITS_DIR = OUTPUT_DIR / "splits"
TRAIN_GROUND_TRUTH_FIT = SPLITS_DIR / "train_ground_truth_fit.tsv"
VAL_GROUND_TRUTH = SPLITS_DIR / "val_ground_truth.tsv"
TRAIN_SOURCE1_FIT = SPLITS_DIR / "train_source1_fit.tsv"
VAL_SOURCE1 = SPLITS_DIR / "val_source1.tsv"

# Normalized datasets directory & file paths
NORMALIZED_DIR = OUTPUT_DIR / "normalized"
TRAIN_SOURCE1_NORM = NORMALIZED_DIR / "train_source1_norm.parquet"
TRAIN_SOURCE2_NORM = NORMALIZED_DIR / "train_source2_norm.parquet"
TRAIN_SOURCE3_NORM = NORMALIZED_DIR / "train_source3_norm.parquet"
TEST_SOURCE1_NORM = NORMALIZED_DIR / "test_source1_norm.parquet"
TEST_SOURCE2_NORM = NORMALIZED_DIR / "test_source2_norm.parquet"
TEST_SOURCE3_NORM = NORMALIZED_DIR / "test_source3_norm.parquet"

NORMALIZED_FILES = {
    "train_source1": TRAIN_SOURCE1_NORM,
    "train_source2": TRAIN_SOURCE2_NORM,
    "train_source3": TRAIN_SOURCE3_NORM,
    "test_source1": TEST_SOURCE1_NORM,
    "test_source2": TEST_SOURCE2_NORM,
    "test_source3": TEST_SOURCE3_NORM,
}

# Candidate pairs paths
CANDIDATES_DIR = OUTPUT_DIR / "candidates"
VAL_CANDIDATES = CANDIDATES_DIR / "val_candidates.parquet"
TRAIN_FIT_CANDIDATES = CANDIDATES_DIR / "train_fit_candidates.parquet"

# Feature tables paths
FEATURES_DIR = OUTPUT_DIR / "features"
TRAIN_FIT_FEATURES = FEATURES_DIR / "train_fit_features.parquet"
VAL_FEATURES = FEATURES_DIR / "val_features.parquet"

RANDOM_SEED = 42

# Ensure directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR.mkdir(parents=True, exist_ok=True)
NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)
CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_DIR.mkdir(parents=True, exist_ok=True)
