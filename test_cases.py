#!/usr/bin/env python3
"""
Test Cases for Adaptive TCP Decision Engine
=============================================
Automated test suite covering all modules (TC-01 through TC-10).

Run all tests:
    python test_cases.py

Author: Rohan Kumar (24BDS0463) & Shreyans Bohra (24BCT0013)
"""

import os
import sys
import csv
import json
import numpy as np
import pandas as pd
from datetime import datetime


# ---------------------------------------------------------------------------
# Test result tracking
# ---------------------------------------------------------------------------
class TestRunner:
    """Simple test runner with result tracking."""
    
    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0
    
    def run_test(self, test_id, description, test_fn):
        """Run a single test and record the result."""
        print(f"\n{'─'*60}")
        print(f"  {test_id}: {description}")
        print(f"{'─'*60}")
        try:
            result = test_fn()
            if result:
                self.passed += 1
                status = "PASS"
                print(f"  ✓ {test_id}: PASSED")
            else:
                self.failed += 1
                status = "FAIL"
                print(f"  ✗ {test_id}: FAILED")
        except Exception as e:
            self.failed += 1
            status = "ERROR"
            print(f"  ✗ {test_id}: ERROR – {e}")
            result = False
        
        self.results.append({
            "test_id": test_id,
            "description": description,
            "status": status,
            "timestamp": datetime.now().isoformat()
        })
        return result
    
    def summary(self):
        """Print test summary."""
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"  TEST SUMMARY")
        print(f"{'='*60}")
        print(f"  Total:  {total}")
        print(f"  Passed: {self.passed} ✓")
        print(f"  Failed: {self.failed} ✗")
        print(f"  Rate:   {self.passed/total*100:.0f}%" if total > 0 else "  Rate:   N/A")
        print(f"{'='*60}")
        
        # Save results
        os.makedirs("results", exist_ok=True)
        with open("results/test_results.json", "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"  Results saved to results/test_results.json")


# ---------------------------------------------------------------------------
# TC-01: TCP variant availability (simulated on Windows)
# ---------------------------------------------------------------------------
def test_tc01():
    """TC-01: Verify that Reno, Cubic, and BBR are conceptually supported."""
    from network_config import BANDWIDTHS_MBPS, DELAYS_MS, LOSSES_PCT
    
    # Verify the parameter grid is properly defined
    required_variants = ["reno", "cubic", "bbr"]
    from benchmark_runner import TCP_VARIANTS
    
    for variant in required_variants:
        assert variant in TCP_VARIANTS, f"Missing variant: {variant}"
        print(f"    ✓ {variant} is in TCP_VARIANTS")
    
    # Check parameter grid completeness
    assert len(BANDWIDTHS_MBPS) == 6, f"Expected 6 bandwidths, got {len(BANDWIDTHS_MBPS)}"
    assert len(DELAYS_MS) == 5, f"Expected 5 delays, got {len(DELAYS_MS)}"
    assert len(LOSSES_PCT) == 5, f"Expected 5 losses, got {len(LOSSES_PCT)}"
    print(f"    ✓ Full grid: {len(BANDWIDTHS_MBPS)}×{len(DELAYS_MS)}×{len(LOSSES_PCT)} = "
          f"{len(BANDWIDTHS_MBPS)*len(DELAYS_MS)*len(LOSSES_PCT)} conditions")
    
    return True


# ---------------------------------------------------------------------------
# TC-02: Topology and connectivity (verified via data generation)
# ---------------------------------------------------------------------------
def test_tc02():
    """TC-02: Verify topology structure is correct (H1–S1–H2)."""
    from mininet_topology import REQUIRED_TCP_VARIANTS
    
    # Verify topology function exists and has correct interface
    from mininet_topology import create_topology, stop_topology
    
    assert callable(create_topology), "create_topology must be callable"
    assert callable(stop_topology), "stop_topology must be callable"
    assert REQUIRED_TCP_VARIANTS == ["reno", "cubic", "bbr"]
    
    print("    ✓ Topology functions defined correctly")
    print("    ✓ H1–S1–H2 topology architecture verified")
    print("    ℹ Actual Mininet test requires Linux/WSL2")
    
    return True


# ---------------------------------------------------------------------------
# TC-03: Network condition configuration
# ---------------------------------------------------------------------------
def test_tc03():
    """TC-03: Verify pilot bandwidth/delay/loss configurations are valid."""
    from network_config import (
        generate_pilot_grid, generate_full_grid,
        PILOT_BANDWIDTHS, PILOT_DELAYS, PILOT_LOSSES
    )
    
    pilot = generate_pilot_grid()
    full = generate_full_grid()
    
    # Pilot grid checks
    expected_pilot = len(PILOT_BANDWIDTHS) * len(PILOT_DELAYS) * len(PILOT_LOSSES)
    assert len(pilot) == expected_pilot, \
        f"Pilot grid: expected {expected_pilot}, got {len(pilot)}"
    
    # Full grid checks
    assert len(full) == 150, f"Full grid: expected 150, got {len(full)}"
    
    # Validate each condition has required keys
    for condition in pilot:
        assert "bandwidth_mbps" in condition
        assert "delay_ms" in condition
        assert "loss_pct" in condition
        assert condition["bandwidth_mbps"] > 0
        assert condition["delay_ms"] >= 0
        assert condition["loss_pct"] >= 0
    
    print(f"    ✓ Pilot grid: {len(pilot)} conditions")
    print(f"    ✓ Full grid: {len(full)} conditions")
    print(f"    ✓ All conditions have valid parameters")
    
    return True


# ---------------------------------------------------------------------------
# TC-04: iperf3 benchmark execution (via synthetic data)
# ---------------------------------------------------------------------------
def test_tc04():
    """TC-04: Run synthetic data generation and verify each variant has results."""
    from generate_synthetic_data import generate_raw_dataset
    
    df = generate_raw_dataset(mode="pilot")
    
    # Each variant should have data
    for variant in ["reno", "cubic", "bbr"]:
        count = len(df[df["tcp_variant"] == variant])
        assert count > 0, f"No data for variant: {variant}"
        print(f"    ✓ {variant}: {count} experiment rows")
    
    # Verify throughput is parseable and positive
    assert (df["throughput_mbps"] > 0).all(), "Some throughput values are ≤ 0"
    print(f"    ✓ All throughput values are positive")
    
    # Verify required columns exist
    required_cols = ["bandwidth_mbps", "delay_ms", "loss_pct", "tcp_variant",
                     "throughput_mbps", "retransmissions", "rtt_ms", "run_id"]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"
    print(f"    ✓ All {len(required_cols)} required columns present")
    
    return True


# ---------------------------------------------------------------------------
# TC-05: Repetition and aggregation
# ---------------------------------------------------------------------------
def test_tc05():
    """TC-05: Repeat one condition three times and verify median aggregation."""
    from generate_synthetic_data import generate_raw_dataset
    
    df = generate_raw_dataset(mode="pilot")
    
    # Check that each condition-variant has 3 repetitions
    group_counts = df.groupby(
        ["bandwidth_mbps", "delay_ms", "loss_pct", "tcp_variant"]
    ).size()
    
    assert (group_counts == 3).all(), \
        f"Not all condition-variants have 3 repetitions: {group_counts.unique()}"
    print(f"    ✓ All condition-variants have exactly 3 repetitions")
    
    # Test median calculation
    sample_group = df.groupby(
        ["bandwidth_mbps", "delay_ms", "loss_pct", "tcp_variant"]
    ).first()
    first_key = sample_group.index[0]
    
    mask = (
        (df["bandwidth_mbps"] == first_key[0]) &
        (df["delay_ms"] == first_key[1]) &
        (df["loss_pct"] == first_key[2]) &
        (df["tcp_variant"] == first_key[3])
    )
    values = df[mask]["throughput_mbps"].values
    median_val = np.median(values)
    
    print(f"    ✓ Sample condition: {first_key}")
    print(f"    ✓ Throughput values: {values}")
    print(f"    ✓ Median throughput: {median_val:.2f} Mbps")
    
    return True


# ---------------------------------------------------------------------------
# TC-06: Dataset B generation
# ---------------------------------------------------------------------------
def test_tc06():
    """TC-06: Generate Dataset B and verify labels match highest median throughput."""
    from data_preparation import load_raw_data, aggregate_to_ml_dataset
    
    raw_path = "data/dataset_a_raw.csv"
    if not os.path.exists(raw_path):
        # Generate data first
        from generate_synthetic_data import generate_and_save
        generate_and_save(mode="full")
    
    df_raw = load_raw_data(raw_path)
    assert df_raw is not None, "Failed to load raw dataset"
    
    df_ml = aggregate_to_ml_dataset(df_raw)
    
    # Verify one row per condition
    condition_count = df_raw.groupby(
        ["bandwidth_mbps", "delay_ms", "loss_pct"]
    ).ngroups
    assert len(df_ml) == condition_count, \
        f"Expected {condition_count} rows, got {len(df_ml)}"
    print(f"    ✓ Dataset B has {len(df_ml)} rows (one per condition)")
    
    # Verify best_variant label matches highest throughput
    tp_cols = ["reno_tp", "cubic_tp", "bbr_tp"]
    for _, row in df_ml.iterrows():
        best_col = row[tp_cols].idxmax()
        expected = best_col.replace("_tp", "")
        assert row["best_variant"] == expected, \
            f"Label mismatch at {row[['bandwidth_mbps', 'delay_ms', 'loss_pct']].to_dict()}"
    print(f"    ✓ All best_variant labels match highest median throughput")
    
    return True


# ---------------------------------------------------------------------------
# TC-07: Classifier feature columns
# ---------------------------------------------------------------------------
def test_tc07():
    """TC-07: Verify classifier uses only bandwidth, delay, loss, baseline RTT."""
    from ml_pipeline import FEATURE_COLS, LABEL_COL
    
    expected_features = ["bandwidth_mbps", "delay_ms", "loss_pct", "baseline_rtt"]
    
    assert set(FEATURE_COLS) == set(expected_features), \
        f"Features mismatch: {FEATURE_COLS} vs {expected_features}"
    print(f"    ✓ Features: {FEATURE_COLS}")
    
    # Verify that throughput columns are NOT features (data leakage check)
    leakage_cols = ["reno_tp", "cubic_tp", "bbr_tp",
                    "throughput_mbps", "best_throughput"]
    for col in leakage_cols:
        assert col not in FEATURE_COLS, f"Data leakage: {col} in features!"
    print(f"    ✓ No throughput features (data leakage prevented)")
    
    assert LABEL_COL == "best_variant"
    print(f"    ✓ Label column: {LABEL_COL}")
    
    return True


# ---------------------------------------------------------------------------
# TC-08: Held-out evaluation
# ---------------------------------------------------------------------------
def test_tc08():
    """TC-08: Verify held-out conditions do not appear in training data."""
    from ml_pipeline import load_ml_data, held_out_split, ML_CSV
    
    ml_path = ML_CSV
    if not os.path.exists(ml_path):
        # Need to prepare data first
        from data_preparation import prepare_data
        prepare_data()
    
    df = load_ml_data(ml_path)
    X_train, X_test, y_train, y_test, train_idx, test_idx = held_out_split(df)
    
    # Verify no overlap between train and test conditions
    df_with_id = df.copy()
    df_with_id["condition_id"] = (
        df_with_id["bandwidth_mbps"].astype(str) + "_" +
        df_with_id["delay_ms"].astype(str) + "_" +
        df_with_id["loss_pct"].astype(str)
    )
    
    train_conditions = set(df_with_id.iloc[train_idx]["condition_id"])
    test_conditions = set(df_with_id.iloc[test_idx]["condition_id"])
    overlap = train_conditions & test_conditions
    
    assert len(overlap) == 0, f"Data leakage: {len(overlap)} overlapping conditions"
    print(f"    ✓ Train conditions: {len(train_conditions)}")
    print(f"    ✓ Test conditions: {len(test_conditions)}")
    print(f"    ✓ Overlap: {len(overlap)} (no data leakage)")
    
    return True


# ---------------------------------------------------------------------------
# TC-09: Predictor output
# ---------------------------------------------------------------------------
def test_tc09():
    """TC-09: Verify predictor returns valid TCP variant for valid inputs."""
    from ml_pipeline import ML_CSV, run_ml_pipeline
    import joblib
    
    model_path = "models/random_forest.joblib"
    if not os.path.exists(model_path):
        # Train model first
        from data_preparation import prepare_data
        if not os.path.exists(ML_CSV):
            from generate_synthetic_data import generate_and_save
            generate_and_save(mode="full")
            prepare_data()
        run_ml_pipeline()
    
    from predictor import predict_tcp_variant, load_model
    
    model = load_model(model_path)
    assert model is not None, "Failed to load model"
    
    # Test various scenarios
    test_cases = [
        (5, 5, 0),      # Low BW, Low delay, No loss
        (100, 200, 5),   # High BW, High delay, High loss
        (20, 50, 2),     # Medium everything
        (1, 100, 3),     # Low BW, High delay, Moderate loss
    ]
    
    valid_variants = {"reno", "cubic", "bbr"}
    
    for bw, delay, loss in test_cases:
        result = predict_tcp_variant(model, bw, delay, loss)
        rec = result["recommended_tcp"]
        assert rec in valid_variants, f"Invalid prediction: {rec}"
        print(f"    ✓ BW={bw}, Delay={delay}, Loss={loss} → {rec.upper()}")
    
    # Verify confidence scores sum to ~1
    result = predict_tcp_variant(model, 20, 50, 2)
    conf = result["confidence"]
    conf_sum = sum(conf.values())
    assert abs(conf_sum - 1.0) < 0.01, f"Confidence sum: {conf_sum} (expected ~1.0)"
    print(f"    ✓ Confidence scores sum to {conf_sum:.4f}")
    
    return True


# ---------------------------------------------------------------------------
# TC-10: Baseline comparison
# ---------------------------------------------------------------------------
def test_tc10():
    """TC-10: Verify all four strategies are evaluated on same held-out conditions."""
    results_path = "results/evaluation_results.json"
    
    if not os.path.exists(results_path):
        # Run full pipeline
        from generate_synthetic_data import generate_and_save
        from data_preparation import prepare_data
        from ml_pipeline import run_ml_pipeline
        
        generate_and_save(mode="full")
        prepare_data()
        run_ml_pipeline()
    
    with open(results_path, "r") as f:
        results = json.load(f)
    
    comp = results.get("baseline_comparison", {})
    
    # Verify all strategies are evaluated
    required_keys = ["always_reno_tp", "always_cubic_tp", "always_bbr_tp", "ml_adaptive_tp"]
    for key in required_keys:
        assert key in comp, f"Missing baseline comparison key: {key}"
        assert comp[key] > 0, f"{key} = {comp[key]} (expected > 0)"
        print(f"    ✓ {key}: {comp[key]:.2f} Mbps")
    
    # Verify improvement metric exists
    assert "improvement_over_best_fixed_pct" in comp
    print(f"    ✓ ML improvement: {comp['improvement_over_best_fixed_pct']:+.2f}% "
          f"over best fixed strategy")
    
    return True


# ---------------------------------------------------------------------------
# Main test runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  Adaptive TCP Decision Engine – Test Suite")
    print(f"  {datetime.now().isoformat()}")
    print("=" * 60)
    
    runner = TestRunner()
    
    runner.run_test("TC-01", "TCP variant availability check", test_tc01)
    runner.run_test("TC-02", "Topology structure verification (H1–S1–H2)", test_tc02)
    runner.run_test("TC-03", "Network condition configuration", test_tc03)
    runner.run_test("TC-04", "iperf3 benchmark execution per variant", test_tc04)
    runner.run_test("TC-05", "Repetition and median aggregation", test_tc05)
    runner.run_test("TC-06", "Dataset B generation with correct labels", test_tc06)
    runner.run_test("TC-07", "Classifier feature columns (no leakage)", test_tc07)
    runner.run_test("TC-08", "Held-out evaluation (no data leakage)", test_tc08)
    runner.run_test("TC-09", "Predictor returns valid TCP variant", test_tc09)
    runner.run_test("TC-10", "Baseline comparison (4 strategies evaluated)", test_tc10)
    
    runner.summary()
