"""End-to-End Test Inference & Submission Generation Pipeline.

Runs the complete pipeline on test datasets:
1. Normalize test_source1, test_source2, and test_source3 (country-agnostic, handling France).
2. Generate candidate pairs via hierarchical bucket-then-similarity blocking.
3. Compute vectorized pairwise features for all generated test candidate pairs.
4. Score candidate pairs using the trained LightGBM model.
5. Apply optimal decision threshold and mutual top candidate tie-breaking.
6. Write required competition deliverables:
   - output/candidate_pairs.tsv (source1_entity_id, candidate_entity_ids)
   - output/matching_results.tsv (source1_entity_id, matched_entity_ids)
7. Report singleton vs matched entity ratio as a sanity check against training distribution.
"""

import gc
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
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
    TEST_SOURCE1,
    TEST_SOURCE2,
    TEST_SOURCE3,
    TEST_SOURCE1_NORM,
    TEST_SOURCE2_NORM,
    TEST_SOURCE3_NORM,
    OUTPUT_DIR,
    RANDOM_SEED,
)
from src.normalize import process_and_save_dataset
from src.blocking import fit_global_vectorizer, get_blocking_key
from src.features import compute_pairwise_features
from src.train import FEATURE_COLUMNS, MODEL_SAVE_PATH, MODEL_PKL_PATH

# Output deliverable paths
CANDIDATE_PAIRS_FILE = OUTPUT_DIR / "candidate_pairs.tsv"
MATCHING_RESULTS_FILE = OUTPUT_DIR / "matching_results.tsv"


def ensure_test_data_normalized():
    """Ensure test sources are normalized and saved as Parquet."""
    print("\nVerifying test data normalization...", flush=True)
    datasets = [
        ("test_source1", TEST_SOURCE1, TEST_SOURCE1_NORM),
        ("test_source2", TEST_SOURCE2, TEST_SOURCE2_NORM),
        ("test_source3", TEST_SOURCE3, TEST_SOURCE3_NORM),
    ]
    for name, in_p, out_p in datasets:
        if not out_p.exists() or out_p.stat().st_size < 1000:
            print(f"Normalizing {name}...", flush=True)
            process_and_save_dataset(in_p, out_p, name)
        else:
            print(f"  {name} normalized Parquet ready: {out_p.name} ({out_p.stat().st_size / (1024*1024):.2f} MB)", flush=True)


def generate_test_candidates(
    df_s1: pd.DataFrame,
    df_s2: pd.DataFrame,
    df_s3: pd.DataFrame,
    vectorizer: Any,
    top_k: int = 20,
    sim_floor: float = 0.12,
    coarse_threshold: int = 500,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Generate candidate pairs for test entities using hierarchical blocking."""
    print("\n--- Generating Candidate Pairs for Test Set ---", flush=True)
    t0 = time.time()

    text_s1 = (df_s1["name_clean"].fillna("") + " " + df_s1["address_norm"].fillna("")).tolist()
    text_s2 = (df_s2["name_clean"].fillna("") + " " + df_s2["address_norm"].fillna("")).tolist()
    text_s3 = (df_s3["name_clean"].fillna("") + " " + df_s3["address_norm"].fillna("")).tolist()

    s1_ids = df_s1["entity_id"].tolist()
    s2_ids = df_s2["entity_id"].tolist()
    s3_ids = df_s3["entity_id"].tolist()

    # Initial 3-char prefix grouping
    b_s1 = defaultdict(list)
    b_cand = defaultdict(list)

    for i, (nc, nn) in enumerate(zip(df_s1["name_clean"], df_s1["name_norm"])):
        k = get_blocking_key(nc, nn, 3)
        b_s1[k].append(i)

    for i, (nc, nn) in enumerate(zip(df_s2["name_clean"], df_s2["name_norm"])):
        k = get_blocking_key(nc, nn, 3)
        b_cand[k].append(("S2", i))

    for i, (nc, nn) in enumerate(zip(df_s3["name_clean"], df_s3["name_norm"])):
        k = get_blocking_key(nc, nn, 3)
        b_cand[k].append(("S3", i))

    # Refine coarse buckets with 4-char prefix
    sub_buckets_s1 = defaultdict(list)
    sub_buckets_cand = defaultdict(list)

    nc_s1 = df_s1["name_clean"].fillna("").tolist()
    nn_s1 = df_s1["name_norm"].fillna("").tolist()
    nc_s2 = df_s2["name_clean"].fillna("").tolist()
    nn_s2 = df_s2["name_norm"].fillna("").tolist()
    nc_s3 = df_s3["name_clean"].fillna("").tolist()
    nn_s3 = df_s3["name_norm"].fillna("").tolist()

    for k, idxs1 in b_s1.items():
        cand_list = b_cand.get(k, [])
        if len(idxs1) + len(cand_list) <= coarse_threshold:
            sub_buckets_s1[k] = idxs1
            sub_buckets_cand[k] = cand_list
        else:
            for i in idxs1:
                sub_k = get_blocking_key(nc_s1[i], nn_s1[i], 4)
                sub_buckets_s1[sub_k].append(i)
            for src, i in cand_list:
                if src == "S2":
                    sub_k = get_blocking_key(nc_s2[i], nn_s2[i], 4)
                else:
                    sub_k = get_blocking_key(nc_s3[i], nn_s3[i], 4)
                sub_buckets_cand[sub_k].append((src, i))

    del b_s1, b_cand
    gc.collect()

    print(f"Bucketing complete. Evaluating {len(sub_buckets_s1):,} sub-buckets...", flush=True)

    candidate_records: Dict[str, Dict[str, float]] = defaultdict(dict)
    # Ensure every S1 has an entry
    for s1_id in s1_ids:
        candidate_records[s1_id] = {}

    for sub_k, s1_idx_list in sub_buckets_s1.items():
        cand_items = sub_buckets_cand.get(sub_k, [])
        if not cand_items:
            continue
        if len(cand_items) > 400:
            cand_items = cand_items[:400]

        s1_texts_b = [text_s1[i] for i in s1_idx_list]
        cand_texts_b = []
        cand_ids_b = []
        for src, i in cand_items:
            if src == "S2":
                cand_texts_b.append(text_s2[i])
                cand_ids_b.append(s2_ids[i])
            else:
                cand_texts_b.append(text_s3[i])
                cand_ids_b.append(s3_ids[i])

        X1 = vectorizer.transform(s1_texts_b)
        Xcand = vectorizer.transform(cand_texts_b)
        sim_mat = (X1 @ Xcand.T).toarray()

        for row_i, s1_idx in enumerate(s1_idx_list):
            s1_id = s1_ids[s1_idx]
            sims = sim_mat[row_i]
            valid_idx = np.where(sims >= sim_floor)[0]
            if len(valid_idx) > 0:
                valid_sims = sims[valid_idx]
                if len(valid_idx) > top_k:
                    top_idx = np.argpartition(-valid_sims, top_k)[:top_k]
                    valid_idx = valid_idx[top_idx]
                    valid_sims = valid_sims[top_idx]

                for ci, c_sim in zip(valid_idx, valid_sims):
                    cid = cand_ids_b[ci]
                    if cid not in candidate_records[s1_id] or c_sim > candidate_records[s1_id][cid]:
                        candidate_records[s1_id][cid] = float(c_sim)

    # Flatten candidate pairs
    rows = []
    candidates_by_s1: Dict[str, List[str]] = {}
    for s1_id in s1_ids:
        cands = candidate_records.get(s1_id, {})
        sorted_cands = [c for c, _ in sorted(cands.items(), key=lambda x: x[1], reverse=True)[:top_k]]
        candidates_by_s1[s1_id] = sorted_cands
        for cid in sorted_cands:
            rows.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": cid,
                "tfidf_similarity": cands[cid],
            })

    df_pairs = pd.DataFrame(rows)
    print(f"Generated {len(df_pairs):,} candidate pairs for {len(s1_ids):,} test entities in {time.time() - t0:.2f}s.", flush=True)
    return df_pairs, candidates_by_s1


def write_candidate_pairs_file(
    candidates_by_s1: Dict[str, List[str]],
    all_s1_ids: List[str],
    output_path: Path = CANDIDATE_PAIRS_FILE,
):
    """Write output/candidate_pairs.tsv with exact candidate lists."""
    print(f"\nWriting candidate pairs to {output_path}...", flush=True)
    rows = []
    for s1 in all_s1_ids:
        cands = candidates_by_s1.get(s1, [])
        # Deduplicate preserving order, ensuring S2-/S3- only
        seen = set()
        clean_cands = []
        for c in cands:
            if c not in seen and (c.startswith("S2-") or c.startswith("S3-")):
                seen.add(c)
                clean_cands.append(c)
        rows.append({
            "source1_entity_id": s1,
            "candidate_entity_ids": ",".join(clean_cands),
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_path, sep="\t", index=False)
    print(f"  Saved candidate_pairs.tsv ({len(df_out):,} rows, {output_path.stat().st_size / (1024*1024):.2f} MB).", flush=True)


def write_matching_results_file(
    predicted_matches_by_s1: Dict[str, List[str]],
    candidates_by_s1: Dict[str, List[str]],
    all_s1_ids: List[str],
    output_path: Path = MATCHING_RESULTS_FILE,
):
    """Write output/matching_results.tsv with predicted matches (subset of candidates)."""
    print(f"\nWriting final matching results to {output_path}...", flush=True)
    rows = []
    num_matched = 0
    num_singletons = 0

    for s1 in all_s1_ids:
        matches = predicted_matches_by_s1.get(s1, [])
        allowed_cands = set(candidates_by_s1.get(s1, []))

        # Enforce constraints: subset of candidates, no duplicates, S2/S3 only
        seen = set()
        valid_matches = []
        for m in matches:
            if m in allowed_cands and m not in seen and (m.startswith("S2-") or m.startswith("S3-")):
                seen.add(m)
                valid_matches.append(m)

        if valid_matches:
            num_matched += 1
        else:
            num_singletons += 1

        rows.append({
            "source1_entity_id": s1,
            "matched_entity_ids": ",".join(valid_matches),
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_path, sep="\t", index=False)
    total_entities = len(df_out)

    print(f"  Saved matching_results.tsv ({total_entities:,} rows, {output_path.stat().st_size / (1024*1024):.2f} MB).", flush=True)
    print("\n" + "=" * 80)
    print("                 TEST PREDICTION SANITY CHECK")
    print("=" * 80)
    print(f"Total Source 1 Test Entities:  {total_entities:,}")
    print(f"Matched Entities (>= 1 match): {num_matched:,} ({num_matched / total_entities * 100:.2f}%)")
    print(f"Singleton Entities (0 matches): {num_singletons:,} ({num_singletons / total_entities * 100:.2f}%)")
    print(f"Training Singleton Benchmark:  ~5.58%")
    print("=" * 80)


def main(optimal_threshold: float = 0.52, use_mutual_tiebreak: bool = True):
    print("=" * 80)
    print("       STAGE 6: END-TO-END TEST INFERENCE & DELIVERABLES GENERATION")
    print("=" * 80)

    # 1. Normalization check
    ensure_test_data_normalized()

    # 2. Load normalized test datasets
    print("\nLoading normalized test datasets (selective columns)...", flush=True)
    cols = ["entity_id", "name_norm", "name_clean", "business_name", "business_address", "address_norm", "postal_code", "landmark", "country"]
    df_s1_test = pd.read_parquet(TEST_SOURCE1_NORM, columns=cols)
    df_s2_test = pd.read_parquet(TEST_SOURCE2_NORM, columns=cols)
    df_s3_test = pd.read_parquet(TEST_SOURCE3_NORM, columns=cols)

    test_s1_ids = df_s1_test["entity_id"].tolist()
    print(f"Loaded {len(df_s1_test):,} Test S1, {len(df_s2_test):,} S2, {len(df_s3_test):,} S3.", flush=True)

    # 3. Fit global vectorizer on combined sample
    sample_texts = (
        df_s1_test["name_clean"].sample(n=min(25000, len(df_s1_test)), random_state=RANDOM_SEED).tolist() +
        df_s2_test["name_clean"].sample(n=min(25000, len(df_s2_test)), random_state=RANDOM_SEED).tolist() +
        df_s3_test["name_clean"].sample(n=min(25000, len(df_s3_test)), random_state=RANDOM_SEED).tolist()
    )
    vectorizer = fit_global_vectorizer(sample_texts, max_features=35000)

    # 4. Generate candidate pairs
    df_test_pairs, candidates_by_s1 = generate_test_candidates(
        df_s1_test, df_s2_test, df_s3_test, vectorizer,
        top_k=20, sim_floor=0.12
    )

    # Write Deliverable 1: output/candidate_pairs.tsv
    write_candidate_pairs_file(candidates_by_s1, test_s1_ids)

    # 5. Compute pairwise features
    feat_df = compute_pairwise_features(df_test_pairs, df_s1_test, df_s2_test, df_s3_test)

    # 6. Score candidate pairs with trained LightGBM model
    print("\nScoring test candidate pairs with LightGBM...", flush=True)
    if MODEL_PKL_PATH.exists():
        clf = joblib.load(MODEL_PKL_PATH)
    else:
        clf = lgb.Booster(model_file=str(MODEL_SAVE_PATH))

    X_test = feat_df[FEATURE_COLUMNS]
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X_test)[:, 1]
    else:
        probs = clf.predict(X_test)

    feat_df["prob"] = probs

    # 7. Apply threshold and tie-breaking strategy
    print(f"\nApplying threshold ({optimal_threshold:.2f}) and mutual tie-breaking ({use_mutual_tiebreak})...", flush=True)
    best_s1_for_cand: Dict[str, Tuple[str, float]] = {}
    if use_mutual_tiebreak:
        for s1, cand, p in zip(feat_df["source1_entity_id"], feat_df["candidate_entity_id"], feat_df["prob"]):
            if cand not in best_s1_for_cand or p > best_s1_for_cand[cand][1]:
                best_s1_for_cand[cand] = (s1, p)

    predicted_matches_by_s1: Dict[str, List[str]] = defaultdict(list)
    for s1, cand, p in zip(feat_df["source1_entity_id"], feat_df["candidate_entity_id"], feat_df["prob"]):
        if p >= optimal_threshold:
            if not use_mutual_tiebreak or (best_s1_for_cand.get(cand, ("", 0))[0] == s1):
                predicted_matches_by_s1[s1].append(cand)

    # Write Deliverable 2: output/matching_results.tsv
    write_matching_results_file(predicted_matches_by_s1, candidates_by_s1, test_s1_ids)

    print("\nEnd-to-end inference complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
