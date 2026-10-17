#!/usr/bin/env python3
"""
Module M4: Data Preparation
=============================
Aggregates raw benchmark data (Dataset A) into the ML-ready
Dataset B with one row per network condition.

Process:
  1. Load raw CSV (Dataset A)
  2. Compute median throughput per (condition, variant) across repetitions
  3. Pivot to one row per condition
  4. Label best_variant as the highest-throughput variant
  5. Save aggregated CSV (Dataset B)

Author: Shreyans Bohra (24BCT0013) – Data & ML
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RAW_CSV     = "data/dataset_a_raw.csv"
ML_CSV      = "data/dataset_b_ml.csv"
CONDITION_COLS = ["bandwidth_mbps", "delay_ms", "loss_pct"]
TCP_VARIANTS   = ["reno", "cubic", "bbr"]


# ---------------------------------------------------------------------------
# Data loading and validation
# ---------------------------------------------------------------------------
def load_raw_data(filepath=RAW_CSV):
    """Load and validate the raw benchmark CSV."""
    if not os.path.exists(filepath):
        print(f"[ERROR] Raw CSV not found: {filepath}")
        return None
    
    df = pd.read_csv(filepath)
    print(f"[LOAD] Raw dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Validate required columns
    required = ["bandwidth_mbps", "delay_ms", "loss_pct", "tcp_variant",
                "throughput_mbps", "retransmissions", "run_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing columns: {missing}")
        return None

    if "baseline_rtt_ms" not in df.columns and "rtt_ms" not in df.columns:
        print("[ERROR] Missing baseline_rtt_ms (or legacy rtt_ms) column.")
        return None
    if "baseline_rtt_ms" not in df.columns:
        print("[WARNING] Using legacy rtt_ms as baseline RTT. This is not valid "
              "for new Mininet experiments because it may be post-selection.")
    
    # Basic data quality checks
    print(f"  Unique conditions: {df.groupby(CONDITION_COLS).ngroups}")
    print(f"  Variants: {sorted(df['tcp_variant'].unique())}")
    print(f"  Run IDs: {sorted(df['run_id'].unique())}")
    print(f"  Missing values: {df.isnull().sum().sum()}")
    print(f"  Throughput range: [{df['throughput_mbps'].min():.2f}, "
          f"{df['throughput_mbps'].max():.2f}] Mbps")

    if not validate_complete_conditions(df):
        return None
    
    return df


def validate_complete_conditions(df):
    """Reject incomplete or failed conditions before ML aggregation."""
    invalid = df[(df["throughput_mbps"] <= 0) | (df["retransmissions"] < 0)]
    if not invalid.empty:
        print(f"[ERROR] Found {len(invalid)} failed/invalid measurement rows.")
        return False

    variants_by_condition = df.groupby(CONDITION_COLS)["tcp_variant"].agg(set)
    incomplete = variants_by_condition[variants_by_condition != set(TCP_VARIANTS)]
    if not incomplete.empty:
        print(f"[ERROR] {len(incomplete)} conditions do not contain exactly Reno, Cubic, and BBR.")
        return False

    repetitions = df.groupby(CONDITION_COLS + ["tcp_variant"]).size()
    if repetitions.nunique() != 1:
        print("[ERROR] Unequal repetition counts across condition/variant groups.")
        return False

    baseline_column = "baseline_rtt_ms" if "baseline_rtt_ms" in df.columns else "rtt_ms"
    if (df[baseline_column] <= 0).any():
        print("[ERROR] Found invalid baseline RTT values.")
        return False
    print(f"  Complete conditions: {len(variants_by_condition)}; repetitions per variant: {repetitions.iloc[0]}")
    return True


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate_to_ml_dataset(df_raw):
    """
    Aggregate raw data to one row per network condition.
    
    Steps:
    1. Group by (condition, variant)
    2. Compute median throughput and retransmissions across repetitions
    3. Pivot variants to columns
    4. Compute baseline_rtt from pre-selection ping measurements
    5. Label best_variant
    
    Parameters
    ----------
    df_raw : pd.DataFrame – Dataset A
    
    Returns
    -------
    pd.DataFrame – Dataset B (ML-ready)
    """
    # Step 1: Median throughput per (condition, variant) across repetitions
    agg = df_raw.groupby(CONDITION_COLS + ["tcp_variant"]).agg(
        median_throughput=("throughput_mbps", "median"),
        median_retransmissions=("retransmissions", "median"),
        num_runs=("run_id", "count")
    ).reset_index()
    
    print(f"\n[AGG] Aggregated to {len(agg)} (condition, variant) groups")
    
    # Step 2: Pivot throughput by variant
    tp_pivot = agg.pivot_table(
        index=CONDITION_COLS,
        columns="tcp_variant",
        values="median_throughput"
    ).reset_index()
    
    # Rename throughput columns
    tp_pivot.columns = [
        f"{c}_tp" if c in TCP_VARIANTS else c
        for c in tp_pivot.columns
    ]
    
    # Step 3: Compute baseline RTT from ping, before a TCP variant is selected.
    # ``rtt_ms`` is supported only for legacy/synthetic data and emits a warning
    # in load_raw_data; new Mininet datasets use baseline_rtt_ms.
    baseline_column = "baseline_rtt_ms" if "baseline_rtt_ms" in df_raw.columns else "rtt_ms"
    baseline_rtt = df_raw.groupby(CONDITION_COLS)[baseline_column].median().reset_index()
    baseline_rtt.columns = CONDITION_COLS + ["baseline_rtt"]
    
    # Step 4: Merge
    df_ml = tp_pivot.merge(baseline_rtt, on=CONDITION_COLS, how="left")
    
    # Step 5: Add retransmission data per variant
    retrans_pivot = agg.pivot_table(
        index=CONDITION_COLS,
        columns="tcp_variant",
        values="median_retransmissions"
    ).reset_index()
    retrans_pivot.columns = [
        f"{c}_retrans" if c in TCP_VARIANTS else c
        for c in retrans_pivot.columns
    ]
    df_ml = df_ml.merge(retrans_pivot, on=CONDITION_COLS, how="left")
    
    # Step 6: Label best_variant (highest median throughput)
    tp_cols = [f"{v}_tp" for v in TCP_VARIANTS]
    df_ml["best_variant"] = df_ml[tp_cols].idxmax(axis=1).str.replace("_tp", "")
    
    # Step 7: Add throughput advantage metric
    df_ml["best_throughput"] = df_ml[tp_cols].max(axis=1)
    df_ml["second_best_throughput"] = df_ml[tp_cols].apply(
        lambda row: sorted(row, reverse=True)[1], axis=1
    )
    df_ml["throughput_advantage_pct"] = (
        (df_ml["best_throughput"] - df_ml["second_best_throughput"])
        / df_ml["second_best_throughput"] * 100
    ).round(2)
    
    print(f"[ML] Created Dataset B: {df_ml.shape[0]} rows, {df_ml.shape[1]} columns")
    print(f"\n  Best variant distribution:")
    print(f"  {df_ml['best_variant'].value_counts().to_dict()}")
    print(f"\n  Average throughput advantage: "
          f"{df_ml['throughput_advantage_pct'].mean():.1f}%")
    
    return df_ml


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
def save_ml_dataset(df_ml, filepath=ML_CSV):
    """Save Dataset B to CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df_ml.to_csv(filepath, index=False)
    print(f"[SAVED] ML dataset → {filepath}")


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def prepare_data(raw_path=RAW_CSV, ml_path=ML_CSV):
    """Run the complete data preparation pipeline."""
    print("=" * 60)
    print("  Module M4: Data Preparation Pipeline")
    print("=" * 60)
    
    # Load
    df_raw = load_raw_data(raw_path)
    if df_raw is None:
        return None
    
    # Aggregate
    df_ml = aggregate_to_ml_dataset(df_raw)
    
    # Save
    save_ml_dataset(df_ml, ml_path)
    
    # Summary statistics
    print(f"\n--- Dataset B Summary ---")
    print(df_ml.describe().round(2).to_string())
    
    return df_ml


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate one experiment dataset for ML.")
    parser.add_argument("--input", default=RAW_CSV, help="Raw CSV input path")
    parser.add_argument("--output", default=ML_CSV, help="Aggregated CSV output path")
    args = parser.parse_args()
    df = prepare_data(args.input, args.output)
    if df is not None:
        print(f"\n\nFirst 5 rows of Dataset B:")
        print(df.head().to_string(index=False))
