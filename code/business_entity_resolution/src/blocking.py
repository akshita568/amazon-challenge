"""Candidate Generation & Blocking Module for Business Entity Resolution ML Challenge.

Implements an efficient O(n) bucket-then-similarity candidate generation approach:
1. Compute cheap blocking keys per record:
   - Primary: first 3 characters of suffix-stripped name (prefix3_name).
   - Finer refinement: if a bucket exceeds 500 records, split by first 4 characters (prefix4_name)
     to prevent coarse bucket bottlenecks.
2. Group Source 1, Source 2, and Source 3 records into buckets using defaultdict.
3. Fit TF-IDF character n-gram cosine vectorizer once globally.
4. Compute similarity strictly within buckets and keep top-K candidates above a similarity floor.
5. Country-agnostic: does NOT block on country (ensuring generalization to France in test).
6. Evaluates recall and reduction ratio against val_ground_truth from Stage 1.
7. Saves generated candidate pairs for downstream feature engineering and model training.
"""

import gc
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure standard output can print Unicode characters safely on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import (
    TRAIN_SOURCE1_NORM,
    TRAIN_SOURCE2_NORM,
    TRAIN_SOURCE3_NORM,
    VAL_GROUND_TRUTH,
    VAL_SOURCE1,
    TRAIN_SOURCE1_FIT,
    VAL_CANDIDATES,
    TRAIN_FIT_CANDIDATES,
    CANDIDATES_DIR,
    RANDOM_SEED,
)


def get_blocking_key(name_clean: str, name_norm: str, length: int = 3) -> str:
    """Extract character prefix blocking key."""
    s = (name_clean or name_norm or "").strip()
    if len(s) >= length:
        return s[:length]
    return (s + "___")[:length]


def fit_global_vectorizer(sample_texts: List[str], max_features: int = 35000) -> TfidfVectorizer:
    """Fit character n-gram TF-IDF vectorizer once globally."""
    print(f"Fitting global TF-IDF vectorizer on {len(sample_texts):,} sample records...", flush=True)
    t0 = time.time()
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 4),
        min_df=3,
        max_features=max_features,
        sublinear_tf=True,
    )
    vec.fit(sample_texts)
    print(f"  Vectorizer fitted in {time.time() - t0:.2f}s (Vocabulary: {len(vec.vocabulary_):,} features)", flush=True)
    return vec


def generate_candidates_for_fold(
    df_s1: pd.DataFrame,
    df_s2: pd.DataFrame,
    df_s3: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    top_k: int = 15,
    sim_floor: float = 0.15,
    coarse_threshold: int = 500,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Generate candidate pairs using hierarchical bucket-then-similarity matching."""
    t_start = time.time()
    print(f"\n--- Generating Candidates (Top-K: {top_k}, Sim Floor: {sim_floor}) ---", flush=True)

    text_s1 = (df_s1["name_clean"].fillna("") + " " + df_s1["address_norm"].fillna("")).tolist()
    text_s2 = (df_s2["name_clean"].fillna("") + " " + df_s2["address_norm"].fillna("")).tolist()
    text_s3 = (df_s3["name_clean"].fillna("") + " " + df_s3["address_norm"].fillna("")).tolist()

    s1_ids = df_s1["entity_id"].tolist()
    s2_ids = df_s2["entity_id"].tolist()
    s3_ids = df_s3["entity_id"].tolist()

    # Step 1: Initial 3-char bucketing
    t_buck = time.time()
    print("Grouping records into 3-character prefix buckets...", flush=True)
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

    print(f"Bucketing complete in {time.time() - t_buck:.2f}s. Unique S1 buckets: {len(b_s1):,}", flush=True)

    # Inspect bucket sizes and identify coarse buckets
    bucket_sizes = []
    coarse_buckets = []
    for k, idxs1 in b_s1.items():
        cand_list = b_cand.get(k, [])
        tot = len(idxs1) + len(cand_list)
        bucket_sizes.append(tot)
        if tot > coarse_threshold:
            coarse_buckets.append((k, len(idxs1), len(cand_list), tot))

    avg_bucket_size = float(np.mean(bucket_sizes)) if bucket_sizes else 0.0
    median_bucket_size = float(np.median(bucket_sizes)) if bucket_sizes else 0.0
    print(f"Bucket Statistics (Initial prefix3_name key):")
    print(f"  - Average Records per Bucket: {avg_bucket_size:.1f}")
    print(f"  - Median Records per Bucket:  {median_bucket_size:.1f}")
    print(f"  - Coarse Buckets (>{coarse_threshold} records): {len(coarse_buckets):,}")

    if coarse_buckets:
        print(f"  [COARSE BUCKET NOTICE]: Top largest buckets before refinement:")
        for k, n1, nc, tot in sorted(coarse_buckets, key=lambda x: x[3], reverse=True)[:3]:
            print(f"    Bucket '{k}': {n1:,} S1 + {nc:,} Candidates = {tot:,} records")
        print(f"  Refining coarse buckets with 4-character prefix (finer key) to optimize speed and precision...")

    # Step 2: Hierarchical Refinement into Sub-Buckets for Evaluation
    # If a bucket > coarse_threshold, sub-divide into 4-char prefix
    sub_buckets_s1 = defaultdict(list)
    # Step 2: Hierarchical Refinement into Sub-Buckets for Evaluation
    # If a bucket > coarse_threshold, sub-divide into 4-char prefix
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
            # Keep as-is
            sub_buckets_s1[k] = idxs1
            sub_buckets_cand[k] = cand_list
        else:
            # Sub-divide by 4-char prefix (using fast list indexing)
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

    refined_sizes = [len(idxs) + len(sub_buckets_cand.get(k, [])) for k, idxs in sub_buckets_s1.items()]
    print(f"Post-Refinement Statistics:")
    print(f"  - Total Sub-Buckets: {len(sub_buckets_s1):,}")
    print(f"  - Refined Average Records per Bucket: {np.mean(refined_sizes):.1f}")
    print(f"  - Refined Median Records per Bucket:  {np.median(refined_sizes):.1f}")

    # Step 3: Compute TF-IDF Cosine Similarity within Sub-Buckets
    t_sim = time.time()
    print("Computing TF-IDF cosine similarities within buckets...", flush=True)

    candidate_records: Dict[str, Dict[str, float]] = defaultdict(dict)
    active_eval_count = 0

    for sub_k, s1_idx_list in sub_buckets_s1.items():
        cand_items = sub_buckets_cand.get(sub_k, [])
        if not cand_items:
            continue
        # Cap candidates per sub-bucket to 400 to prevent pathological bottlenecks
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

        # Vectorize and compute dot product
        X1 = vectorizer.transform(s1_texts_b)
        Xcand = vectorizer.transform(cand_texts_b)
        sim_mat = (X1 @ Xcand.T).toarray()

        for row_i, s1_idx in enumerate(s1_idx_list):
            s1_id = s1_ids[s1_idx]
            sims = sim_mat[row_i]
            valid_mask = sims >= sim_floor
            valid_idx = np.where(valid_mask)[0]

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

        active_eval_count += 1

    sim_time = time.time() - t_sim
    total_time = time.time() - t_start
    print(f"Evaluated {active_eval_count:,} active sub-buckets in {sim_time:.2f}s (Total blocking time: {total_time:.2f}s).", flush=True)

    # Flatten candidate pairs
    rows = []
    for s1_id, cands in candidate_records.items():
        sorted_cands = sorted(cands.items(), key=lambda x: x[1], reverse=True)[:top_k]
        for cid, sim in sorted_cands:
            rows.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": cid,
                "tfidf_similarity": sim,
            })

    df_pairs = pd.DataFrame(rows)
    stats = {
        "top_k": top_k,
        "sim_floor": sim_floor,
        "total_time_s": total_time,
        "avg_bucket_size": avg_bucket_size,
        "refined_avg_bucket_size": float(np.mean(refined_sizes)) if refined_sizes else 0.0,
        "num_coarse_buckets": len(coarse_buckets),
        "total_candidate_pairs": len(df_pairs),
    }

    return df_pairs, stats


def evaluate_blocking_recall(
    df_candidates: pd.DataFrame,
    val_gt: pd.DataFrame,
    total_val_s1_count: int,
) -> Tuple[float, float, Dict[str, Any]]:
    """Compute recall of true matches in val_ground_truth and candidate reduction ratio."""
    print("\n--- Evaluating Blocking Recall on Validation Split ---", flush=True)

    cand_map = defaultdict(set)
    for _, row in df_candidates.iterrows():
        cand_map[row["source1_entity_id"]].add(row["candidate_entity_id"])

    total_true_matches = 0
    recalled_matches = 0
    entities_with_matches = 0
    entities_fully_recalled = 0

    for _, row in val_gt.iterrows():
        s1_id = row["source1_entity_id"]
        matches_str = str(row["matched_entity_ids"]).strip()
        if not matches_str or matches_str == "nan":
            continue

        true_matches = [m.strip() for m in matches_str.split(",") if m.strip()]
        if not true_matches:
            continue

        entities_with_matches += 1
        total_true_matches += len(true_matches)
        cands = cand_map.get(s1_id, set())

        hits = sum(1 for m in true_matches if m in cands)
        recalled_matches += hits
        if hits == len(true_matches):
            entities_fully_recalled += 1

    recall = (recalled_matches / total_true_matches) if total_true_matches > 0 else 0.0
    entity_coverage = (entities_fully_recalled / entities_with_matches) if entities_with_matches > 0 else 0.0
    avg_candidates_per_entity = len(df_candidates) / total_val_s1_count if total_val_s1_count > 0 else 0.0

    print(f"Validation Evaluation Metrics:")
    print(f"  - Total Validation S1 Entities:     {total_val_s1_count:,}")
    print(f"  - Total True Matches to Recover:    {total_true_matches:,}")
    print(f"  - True Matches Recalled in Top-K:   {recalled_matches:,}")
    print(f"  =======================================================")
    print(f"  >>> BLOCKING RECALL:                {recall * 100:.2f}% <<<")
    print(f"  - Perfect Entity Recall Rate:       {entity_coverage * 100:.2f}% ({entities_fully_recalled:,}/{entities_with_matches:,})")
    print(f"  - Average Candidates per S1 Entity: {avg_candidates_per_entity:.2f}")
    print(f"  - Reduction Ratio:                  ~{avg_candidates_per_entity / 10320000 * 100:.6f}%")
    print(f"  - Total Generated Candidate Pairs:  {len(df_candidates):,}")
    print(f"  =======================================================")

    metrics = {
        "blocking_recall": recall,
        "avg_candidates_per_entity": avg_candidates_per_entity,
        "total_true_matches": total_true_matches,
        "recalled_matches": recalled_matches,
    }
    return recall, avg_candidates_per_entity, metrics


def main():
    print("=" * 80)
    print("           STAGE 2: CANDIDATE GENERATION & BLOCKING")
    print("=" * 80)

    # 1. Load validation split (selective columns for speed)
    print("Loading normalized datasets (selective columns)...", flush=True)
    t0 = time.time()
    cols = ["entity_id", "name_clean", "name_norm", "address_norm"]
    df_s1_val = pd.read_parquet(TRAIN_SOURCE1_NORM, columns=cols)
    val_gt = pd.read_csv(VAL_GROUND_TRUTH, sep="\t")
    val_s1_ids = set(val_gt["source1_entity_id"])
    df_s1_val = df_s1_val[df_s1_val["entity_id"].isin(val_s1_ids)].copy()

    # Load Source 2 and Source 3
    df_s2 = pd.read_parquet(TRAIN_SOURCE2_NORM, columns=cols)
    df_s3 = pd.read_parquet(TRAIN_SOURCE3_NORM, columns=cols)
    print(f"Loaded {len(df_s1_val):,} Val S1, {len(df_s2):,} S2, {len(df_s3):,} S3 in {time.time() - t0:.2f}s.", flush=True)

    # 2. Fit global TF-IDF vectorizer
    sample_corpus = (
        df_s1_val["name_clean"].sample(n=min(25000, len(df_s1_val)), random_state=RANDOM_SEED).tolist() +
        df_s2["name_clean"].sample(n=min(25000, len(df_s2)), random_state=RANDOM_SEED).tolist() +
        df_s3["name_clean"].sample(n=min(25000, len(df_s3)), random_state=RANDOM_SEED).tolist()
    )
    vectorizer = fit_global_vectorizer(sample_corpus, max_features=35000)

    # 3. Experiment on a representative 10,000 entity validation slice for fast benchmarking
    eval_slice_size = 10000
    val_sample_gt = val_gt.head(eval_slice_size).copy()
    sample_s1_ids = set(val_sample_gt["source1_entity_id"])
    df_s1_slice = df_s1_val[df_s1_val["entity_id"].isin(sample_s1_ids)].copy()

    print("\n" + "=" * 80)
    print("RUNNING CANDIDATE GENERATION: prefix3 with prefix4 refinement (K=15, Floor=0.15)")
    print("=" * 80)

    cands_df, stats = generate_candidates_for_fold(
        df_s1_slice, df_s2, df_s3, vectorizer,
        top_k=15, sim_floor=0.15, coarse_threshold=500
    )
    recall, red_ratio, _ = evaluate_blocking_recall(cands_df, val_sample_gt, len(df_s1_slice))

    # If recall is below 90%, test adjustment with higher K (K=25) and lower floor (0.10)
    if recall < 0.90:
        print("\n" + "=" * 80)
        print("RECALL ADJUSTMENT: Increasing K to 25 and lowering similarity floor to 0.10")
        print("=" * 80)
        cands_df_adj, stats_adj = generate_candidates_for_fold(
            df_s1_slice, df_s2, df_s3, vectorizer,
            top_k=25, sim_floor=0.10, coarse_threshold=500
        )
        recall_adj, red_adj, _ = evaluate_blocking_recall(cands_df_adj, val_sample_gt, len(df_s1_slice))

        print("\n" + "=" * 80)
        print("                  BLOCKING STRATEGY COMPARISON")
        print("=" * 80)
        comp_df = pd.DataFrame([
            {
                "Configuration": "K=15, Floor=0.15",
                "Blocking Time": f"{stats['total_time_s']:.1f}s",
                "Avg Bucket Size": f"{stats['refined_avg_bucket_size']:.1f}",
                "Blocking Recall": f"{recall * 100:.2f}%",
                "Avg Candidates/S1": f"{red_ratio:.2f}",
            },
            {
                "Configuration": "K=25, Floor=0.10 (Adjusted)",
                "Blocking Time": f"{stats_adj['total_time_s']:.1f}s",
                "Avg Bucket Size": f"{stats_adj['refined_avg_bucket_size']:.1f}",
                "Blocking Recall": f"{recall_adj * 100:.2f}%",
                "Avg Candidates/S1": f"{red_adj:.2f}",
            },
        ])
        print(comp_df.to_string(index=False))
        # Keep the higher recall candidate set
        cands_df = cands_df_adj

    # Save candidates for Stage 3 & Stage 4
    out_file = VAL_CANDIDATES
    cands_df.to_parquet(out_file, index=False)
    print(f"\nSaved validation candidate pairs to {out_file} ({len(cands_df):,} rows).", flush=True)

    print("=" * 80)
    print("Blocking & Candidate Generation Complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
