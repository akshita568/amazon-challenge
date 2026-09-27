"""Exploratory Data Analysis (EDA) on Training Datasets for Business Entity Resolution.

Performs:
1. Row counts for Source 1, Source 2, Source 3, and Ground Truth.
2. Null / empty rate per column per source.
3. Country distributions across train sources (and reminder/verification of France in test).
4. Ground truth analysis: singletons vs matched, match list length distribution,
   and source provenance (Source 2 only, Source 3 only, or both).
5. Side-by-side inspection of matched triples (Source 1 + matched Source 2/3)
   to eyeball text variations, typos, abbreviations, and address formats.
"""

import gc
import sys
from pathlib import Path
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
    TRAIN_GROUND_TRUTH,
    TEST_SOURCE1,
    TEST_SOURCE2,
    TEST_SOURCE3,
)


def analyze_ground_truth(n_samples_for_inspection: int = 6):
    """Analyze train_ground_truth.tsv for singletons, match length, and source composition.
    
    Also selects sample triples with matches across both S2 and S3 for visual noise inspection.
    """
    print(f"\n{'='*30} GROUND TRUTH ANALYSIS {'='*30}")
    gt = pd.read_csv(TRAIN_GROUND_TRUTH, sep="\t", low_memory=False)
    total_gt = len(gt)
    print(f"Total Ground Truth Entities (Source 1 reference): {total_gt:,}")

    # Process matched_entity_ids
    # Handle NaN as empty string
    raw_matches = gt["matched_entity_ids"].fillna("").astype(str)
    
    # Split matches
    match_lists = raw_matches.apply(lambda x: [m.strip() for m in x.split(",") if m.strip()])
    match_counts = match_lists.apply(len)

    # 1. Singletons vs Matched
    singletons_mask = (match_counts == 0)
    num_singletons = singletons_mask.sum()
    num_matched = total_gt - num_singletons

    print("\n--- 1. Singletons vs Matched Entities ---")
    print(f"Singletons (0 matches in S2/S3): {num_singletons:,} ({num_singletons / total_gt * 100:.2f}%)")
    print(f"Matched Entities (>= 1 match):    {num_matched:,} ({num_matched / total_gt * 100:.2f}%)")

    # 2. Match Length Distribution
    print("\n--- 2. Match List Length Distribution ---")
    length_counts = match_counts.value_counts().sort_index()
    length_df = pd.DataFrame({
        "Match Count": length_counts.index,
        "Entities": length_counts.values,
        "Percentage": (length_counts.values / total_gt * 100).round(2),
    })
    print(length_df.head(12).to_string(index=False))

    # 3. Source Breakdown (S2 only, S3 only, Both S2 and S3)
    print("\n--- 3. Source Provenance of Matches ---")
    matched_lists = match_lists[~singletons_mask]

    def get_source_breakdown(m_list):
        has_s2 = any(m.startswith("S2-") for m in m_list)
        has_s3 = any(m.startswith("S3-") for m in m_list)
        if has_s2 and has_s3:
            return "Both S2 and S3"
        elif has_s2:
            return "Source 2 only"
        elif has_s3:
            return "Source 3 only"
        else:
            return "Unknown"

    source_provenance = matched_lists.apply(get_source_breakdown)
    prov_counts = source_provenance.value_counts()
    prov_df = pd.DataFrame({
        "Entities": prov_counts.values,
        "% of Matched": (prov_counts.values / num_matched * 100).round(2),
        "% of Total S1": (prov_counts.values / total_gt * 100).round(2),
    }, index=prov_counts.index)
    print(prov_df.to_string())

    # Select candidates that have matches in BOTH S2 and S3
    def has_both(m_list):
        return any(m.startswith("S2-") for m in m_list) and any(m.startswith("S3-") for m in m_list)

    candidates = gt[match_lists.apply(has_both)].copy()
    sampled = candidates.sample(n=n_samples_for_inspection, random_state=42)

    s1_ids = sampled["source1_entity_id"].tolist()
    s2_ids_needed = set()
    s3_ids_needed = set()

    for idx, row in sampled.iterrows():
        matches = match_lists.loc[idx]
        for m in matches:
            if m.startswith("S2-"):
                s2_ids_needed.add(m)
            elif m.startswith("S3-"):
                s3_ids_needed.add(m)

    sample_meta = {
        "sampled_df": sampled,
        "match_lists": match_lists,
        "s1_ids": set(s1_ids),
        "s2_ids": s2_ids_needed,
        "s3_ids": s3_ids_needed,
    }

    del gt, raw_matches, match_lists, match_counts, matched_lists
    gc.collect()

    return sample_meta


def analyze_source_table(name: str, path: Path, ids_to_extract: set = None):
    """Analyze row counts, null rates, and country distribution of a source table.
    
    Optionally extracts target sample rows for inspection without re-reading the file.
    """
    print(f"\n{'='*30} {name} {'='*30}")
    print(f"File Path: {path}")

    df = pd.read_csv(path, sep="\t", low_memory=False)
    total_rows = len(df)
    print(f"Total Rows:    {total_rows:,}")
    print(f"Total Columns: {df.shape[1]}")

    print("\n--- Column Null / Empty Rates ---")
    null_stats = []
    for col in df.columns:
        null_count = df[col].isna().sum()
        empty_count = 0
        if df[col].dtype == object:
            empty_count = (df[col].fillna("").astype(str).str.strip() == "").sum() - null_count
            if empty_count < 0:
                empty_count = 0
        total_missing = null_count + empty_count
        null_stats.append({
            "Column": col,
            "Null Count": null_count,
            "Empty Str Count": empty_count,
            "Total Missing": total_missing,
            "Missing %": f"{(total_missing / total_rows) * 100:.4f}%",
        })
    stats_df = pd.DataFrame(null_stats)
    print(stats_df.to_string(index=False))

    if "country" in df.columns:
        print("\n--- Country Distribution ---")
        counts = df["country"].value_counts(dropna=False)
        percentages = (counts / total_rows) * 100
        country_df = pd.DataFrame({
            "Count": counts,
            "Percentage": percentages.map("{:.2f}%".format),
        })
        print(country_df.to_string())

    extracted_records = {}
    if ids_to_extract:
        subset = df[df["entity_id"].isin(ids_to_extract)]
        for _, r in subset.iterrows():
            extracted_records[r["entity_id"]] = r.to_dict()

    del df
    gc.collect()

    return total_rows, extracted_records


def analyze_test_countries():
    """Quickly check country distributions in test sets to verify unseen countries."""
    print(f"\n{'='*25} TEST SET COUNTRY DISTRIBUTION CHECK {'='*25}")
    for name, path in [
        ("test_source1", TEST_SOURCE1),
        ("test_source2", TEST_SOURCE2),
        ("test_source3", TEST_SOURCE3),
    ]:
        df = pd.read_csv(path, sep="\t", usecols=["country"], low_memory=False)
        counts = df["country"].value_counts(dropna=False)
        percentages = (counts / len(df)) * 100
        dist_df = pd.DataFrame({
            "Count": counts,
            "Percentage": percentages.map("{:.2f}%".format),
        })
        print(f"\n{name} Country Breakdown (Total: {len(df):,}):")
        print(dist_df.to_string())
        del df
        gc.collect()

    print("\n[IMPORTANT REMINDER]: 'France' is present in test (~14.4 - 15.0%) but completely UNSEEN")
    print("in the training set! Blocking and matching models must be country-agnostic or generalize")
    print("without relying on France-specific training examples.")


def display_matched_examples(sample_meta: dict, s1_records: dict, s2_records: dict, s3_records: dict):
    """Print side-by-side inspection of matched triples to eyeball noise."""
    print(f"\n{'='*30} MATCHED TRIPLES INSPECTION (NOISE / VARIATIONS) {'='*30}")
    sampled_df = sample_meta["sampled_df"]
    match_lists = sample_meta["match_lists"]

    print(f"Inspecting {len(sampled_df)} Matched Entity Triples/Groups:\n")
    for sample_idx, (orig_idx, row) in enumerate(sampled_df.iterrows(), start=1):
        s1_id = row["source1_entity_id"]
        matches = match_lists.loc[orig_idx]
        s1_row = s1_records.get(s1_id)

        print(f"--- [Example {sample_idx}] Reference ID: {s1_id} ---")
        if s1_row:
            print(f"  [SOURCE 1]  Name:    {s1_row.get('business_name')}")
            print(f"              Address: {s1_row.get('business_address')}")
            print(f"              Country: {s1_row.get('country')}")
        else:
            print(f"  [SOURCE 1]  Record not found")

        # Matched S2
        s2_matches = [m for m in matches if m.startswith("S2-")]
        for s2_id in s2_matches:
            if s2_id in s2_records:
                r = s2_records[s2_id]
                print(f"  -> [MATCH S2] ({s2_id})")
                print(f"              Name:    {r.get('business_name')}")
                print(f"              Address: {r.get('business_address')}")
                print(f"              Country: {r.get('country')}")

        # Matched S3
        s3_matches = [m for m in matches if m.startswith("S3-")]
        for s3_id in s3_matches:
            if s3_id in s3_records:
                r = s3_records[s3_id]
                print(f"  -> [MATCH S3] ({s3_id})")
                print(f"              Name:    {r.get('business_name')}")
                print(f"              Address: {r.get('business_address')}")
                print(f"              Country: {r.get('country')}")
        print()


def main():
    print("=" * 80)
    print("       EXPLORATORY DATA ANALYSIS (BUSINESS ENTITY RESOLUTION)")
    print("=" * 80)

    # Step 1: Ground truth analysis + sample triple selection
    sample_meta = analyze_ground_truth(n_samples_for_inspection=6)

    # Step 2: Source tables analysis + sample row extraction in single pass
    _, s1_records = analyze_source_table("train_source1", TRAIN_SOURCE1, sample_meta["s1_ids"])
    _, s2_records = analyze_source_table("train_source2", TRAIN_SOURCE2, sample_meta["s2_ids"])
    _, s3_records = analyze_source_table("train_source3", TRAIN_SOURCE3, sample_meta["s3_ids"])

    # Step 3: Test set countries check & reminder
    analyze_test_countries()

    # Step 4: Display sample matched triples
    display_matched_examples(sample_meta, s1_records, s2_records, s3_records)

    print("=" * 80)
    print("EDA Complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
