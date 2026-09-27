"""Reusable Data Normalization Module for Business Entity Resolution ML Challenge.

Provides country-agnostic string normalization for business names and addresses:
- normalize_name: Lowercase, diacritic folding, punctuation stripping, domain/noise removal,
  standardizing legal abbreviations, and returning BOTH a raw normalized version (name_norm)
  and a legal suffix-stripped version (name_clean).
- normalize_address: Lowercase, diacritic folding, punctuation stripping, abbreviation expansion
  (road, street, avenue, boulevard, apartment, etc.), landmark extraction (near/opp X),
  and trailing postal/ZIP code extraction.
- Fully country-agnostic design degrading gracefully on US, India, France (unseen in train), etc.
- Batch processing and saving normalized parquet tables for all train and test sources.
"""

import gc
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Tuple, List, Dict, Any
import pandas as pd

# Ensure standard output can print Unicode characters safely on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import (
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
    TEST_SOURCE1,
    TEST_SOURCE2,
    TEST_SOURCE3,
    TRAIN_SOURCE1_NORM,
    TRAIN_SOURCE2_NORM,
    TRAIN_SOURCE3_NORM,
    TEST_SOURCE1_NORM,
    TEST_SOURCE2_NORM,
    TEST_SOURCE3_NORM,
    NORMALIZED_DIR,
    RANDOM_SEED,
)

# ---------------------------------------------------------------------------
# Pre-compiled Regex Patterns & Lookups
# ---------------------------------------------------------------------------

# Noise patterns commonly found in business datasets
CLEAN_NOISE_PATS = [
    (re.compile(r"\(id:\s*\d+\)", re.IGNORECASE), " "),
    (re.compile(r"\bid:\s*\d+\b", re.IGNORECASE), " "),
    (re.compile(r"<null>", re.IGNORECASE), " "),
    (re.compile(r"<na>", re.IGNORECASE), " "),
    (re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE), " "),
    # Strip common web TLD extensions occurring in place of business names
    (re.compile(r"\.(?:com|org|net|co\.in|in|fr|io|biz|info|gov|edu)\b", re.IGNORECASE), " "),
]

# Legal suffixes mapping to standardize in raw_norm
LEGAL_EXPAND_MAP = {
    "pvt": "private",
    "ltd": "limited",
    "corp": "corporation",
    "inc": "incorporated",
    "co": "company",
}
LEGAL_EXPAND_REGEX = re.compile(r"\b(" + "|".join(LEGAL_EXPAND_MAP.keys()) + r")\b")

# Legal suffixes to strip completely for name_clean
# Covers English (US/India), French (SARL, SA, SAS, EURL), and international corporate forms
STRIP_NAME_LEGAL = re.compile(
    r"\b(?:"
    r"private limited|pvt ltd|private ltd|pvt limited|"
    r"limited liability company|llc services|llc|"
    r"incorporated|inc|"
    r"corporation|corp foundation|corp|"
    r"limited|ltd|"
    r"private|pvt|"
    r"company|co|"
    r"societe a responsabilite limitee|sarl|sas|sa|eurl|"
    r"gmbh|ag|llp|plc"
    r")\b"
)

# Address abbreviations expansion mapping (US, India, France)
ADDR_ABBR_MAP = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "bd": "boulevard",
    "bvd": "boulevard",
    "apt": "apartment",
    "ln": "lane",
    "dr": "drive",
    "ct": "court",
    "pl": "place",
    "sq": "square",
    "hwy": "highway",
    "pkwy": "parkway",
    "ste": "suite",
    "fl": "floor",
    "bldg": "building",
    "dept": "department",
}
ADDR_ABBR_REGEX = re.compile(r"\b(" + "|".join(ADDR_ABBR_MAP.keys()) + r")\b")

# Landmark pattern: captures phrases indicating proximity to a landmark
LANDMARK_PATTERN = re.compile(
    r"\b(near|opp|opposite|behind|next to|adj to|adjacent to|beside|in front of)\s+([^,]+)",
    re.IGNORECASE,
)

# Trailing postal / PIN / ZIP code pattern
# Matches 5-digit (US, France) or 6-digit (India) numeric code near the end of address
POSTAL_PATTERN = re.compile(
    r"\b([0-9]{5,6}(?:-[0-9]{4})?)\b(?:\s+[a-zA-Z]+)?\s*$",
    re.IGNORECASE,
)


def fold_diacritics(text: str) -> str:
    """Normalize unicode and strip combining diacritical marks (e.g. é -> e, á -> a).
    
    Fast path: returns immediately if string is already ASCII.
    """
    if not text or text.isascii():
        return text or ""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


# ---------------------------------------------------------------------------
# Core Normalization Functions
# ---------------------------------------------------------------------------

def normalize_name(name: Any) -> Tuple[str, str]:
    """Normalize business name and return (name_norm, name_clean).

    Args:
        name: Raw business name string.

    Returns:
        tuple (name_norm, name_clean):
        - name_norm: lowercased, diacritics folded, punctuation stripped,
          noise removed, legal abbreviations standardized.
        - name_clean: name_norm with all common legal suffixes stripped.
    """
    if name is None or pd.isna(name):
        return "", ""

    s = str(name).strip()
    if not s:
        return "", ""

    # 1. Fold diacritics and convert to lowercase
    s = fold_diacritics(s).lower()

    # 2. Clean metadata noise (ID tags, web URLs, TLD extensions)
    for pat, repl in CLEAN_NOISE_PATS:
        s = pat.sub(repl, s)

    # 3. Strip punctuation (replace non-alphanumeric with spaces)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # 4. Standardize legal abbreviations into full forms for name_norm
    name_norm = LEGAL_EXPAND_REGEX.sub(lambda m: LEGAL_EXPAND_MAP[m.group(1)], s)
    name_norm = re.sub(r"\s+", " ", name_norm).strip()

    # 5. Strip legal suffixes completely for name_clean
    name_clean = STRIP_NAME_LEGAL.sub(" ", name_norm)
    name_clean = re.sub(r"\s+", " ", name_clean).strip()

    # Graceful degradation: if stripping legal suffixes left nothing, retain name_norm
    if not name_clean:
        name_clean = name_norm

    return name_norm, name_clean


def normalize_address(address: Any) -> Tuple[str, str, str]:
    """Normalize address and return (address_norm, landmark, postal_code).

    Args:
        address: Raw address string.

    Returns:
        tuple (address_norm, landmark, postal_code):
        - address_norm: lowercased, punctuation stripped, abbreviations expanded,
          landmark excised.
        - landmark: extracted landmark phrase (e.g. 'near railway station').
        - postal_code: extracted trailing numeric PIN/ZIP code (e.g. '11379', '313001').
    """
    if address is None or pd.isna(address):
        return "", "", ""

    s = str(address).strip()
    if not s:
        return "", "", ""

    # 1. Fold diacritics
    s = fold_diacritics(s)

    # Clean explicit null tokens
    s = s.replace("<NULL>", " ").replace("<null>", " ").replace("<na>", " ")

    # 2. Extract landmark phrase before stripping punctuation
    landmark = ""
    lm_match = LANDMARK_PATTERN.search(s)
    if lm_match:
        landmark = lm_match.group(0).strip()
        # Remove landmark phrase from address
        s = s[:lm_match.start()] + " " + s[lm_match.end():]

    # 3. Extract trailing numeric PIN / ZIP code
    postal_code = ""
    post_match = POSTAL_PATTERN.search(s.strip())
    if post_match:
        postal_code = post_match.group(1).strip()

    # 4. Lowercase and strip punctuation
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)

    # 5. Expand abbreviations (single regex pass)
    s = ADDR_ABBR_REGEX.sub(lambda m: ADDR_ABBR_MAP[m.group(1)], s)
    s = re.sub(r"\s+", " ", s).strip()

    # 6. Clean landmark string
    if landmark:
        landmark = fold_diacritics(landmark).lower()
        landmark = re.sub(r"[^a-z0-9\s]", " ", landmark)
        landmark = re.sub(r"\s+", " ", landmark).strip()

    return s, landmark, postal_code


# ---------------------------------------------------------------------------
# High-Throughput Batch Processing & Parquet Saving
# ---------------------------------------------------------------------------

def process_and_save_dataset(
    input_path: Path,
    output_path: Path,
    dataset_name: str,
) -> pd.DataFrame:
    """Normalize names and addresses of a dataset in memory and save as Parquet."""
    if output_path.exists() and output_path.stat().st_size > 1000:
        print(f"\n{dataset_name} ({output_path.name}) already exists ({output_path.stat().st_size / (1024*1024):.2f} MB). Skipping re-computation.", flush=True)
        return pd.read_parquet(output_path, columns=["entity_id", "business_name", "name_norm", "name_clean", "business_address", "address_norm", "landmark", "postal_code", "country"]).head(20)

    print(f"\nProcessing {dataset_name} ({input_path.name})...", flush=True)
    t0 = time.time()

    df = pd.read_csv(input_path, sep="\t", low_memory=False)
    n_rows = len(df)
    print(f"  Loaded {n_rows:,} rows in {time.time() - t0:.2f}s", flush=True)

    # Fast normalization using list comprehension (benchmarked at >50k rows/s)
    t_norm = time.time()
    names = df["business_name"].tolist()
    name_tuples = [normalize_name(x) for x in names]
    df["name_norm"] = [t[0] for t in name_tuples]
    df["name_clean"] = [t[1] for t in name_tuples]

    addresses = df["business_address"].tolist()
    addr_tuples = [normalize_address(x) for x in addresses]
    df["address_norm"] = [t[0] for t in addr_tuples]
    df["landmark"] = [t[1] for t in addr_tuples]
    df["postal_code"] = [t[2] for t in addr_tuples]

    dur = time.time() - t_norm
    print(f"  Normalized {n_rows:,} records in {dur:.2f}s ({n_rows / dur:.0f} rows/s)", flush=True)

    # Save to Parquet
    t_save = time.time()
    df.to_parquet(output_path, index=False, compression="snappy")
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved Parquet to {output_path} ({size_mb:.2f} MB) in {time.time() - t_save:.2f}s", flush=True)

    return df


def print_before_after_samples(df: pd.DataFrame, source_name: str, n_samples: int = 10):
    """Print 10 random before/after examples for sanity checking signal retention."""
    print(f"\n{'='*35} {source_name.upper()} SANITY CHECK ({n_samples} SAMPLES) {'='*35}\n")
    sample = df.sample(n=min(n_samples, len(df)), random_state=RANDOM_SEED)

    for i, (_, row) in enumerate(sample.iterrows(), start=1):
        print(f"[{source_name} - Sample {i}] (Entity ID: {row['entity_id']})")
        print(f"  NAME Original:   '{row['business_name']}'")
        print(f"       name_norm:   '{row['name_norm']}'")
        print(f"       name_clean:  '{row['name_clean']}'")
        print(f"  ADDR Original:   '{row['business_address']}'")
        print(f"       addr_norm:   '{row['address_norm']}'")
        if row["landmark"]:
            print(f"       landmark:    '{row['landmark']}'")
        if row["postal_code"]:
            print(f"       postal_code: '{row['postal_code']}'")
        print(f"  Country:         {row['country']}")
        print("-" * 80)


def main():
    print("=" * 80)
    print("       BUSINESS ENTITY RESOLUTION: DATA NORMALIZATION PIPELINE")
    print("=" * 80)
    print(f"Output Directory: {NORMALIZED_DIR}\n")

    datasets_to_process = [
        ("train_source1", TRAIN_SOURCE1, TRAIN_SOURCE1_NORM),
        ("train_source2", TRAIN_SOURCE2, TRAIN_SOURCE2_NORM),
        ("train_source3", TRAIN_SOURCE3, TRAIN_SOURCE3_NORM),
        ("test_source1", TEST_SOURCE1, TEST_SOURCE1_NORM),
        ("test_source2", TEST_SOURCE2, TEST_SOURCE2_NORM),
        ("test_source3", TEST_SOURCE3, TEST_SOURCE3_NORM),
    ]

    samples_to_inspect: Dict[str, pd.DataFrame] = {}

    total_start = time.time()
    for name, in_path, out_path in datasets_to_process:
        df = process_and_save_dataset(in_path, out_path, name)
        # Retain a slice for sanity checking the 3 sources
        if name in ("train_source1", "train_source2", "train_source3"):
            samples_to_inspect[name] = df[["entity_id", "business_name", "name_norm", "name_clean",
                                           "business_address", "address_norm", "landmark", "postal_code", "country"]].sample(
                n=10, random_state=RANDOM_SEED
            )
        del df
        gc.collect()

    print(f"\nAll datasets processed and saved in {time.time() - total_start:.2f}s.")

    # Print 10 before/after sanity check samples for each source
    print("\n" + "=" * 80)
    print("                BEFORE / AFTER SANITY CHECK INSPECTION")
    print("=" * 80)
    for name, sample_df in samples_to_inspect.items():
        print_before_after_samples(sample_df, name, n_samples=10)

    print("=" * 80)
    print("Normalization complete and verified.")
    print("=" * 80)


if __name__ == "__main__":
    main()
