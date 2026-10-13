#!/usr/bin/env python3
"""
Main Pipeline Runner
=====================
Runs the complete end-to-end pipeline on Windows:
  1. Generate synthetic TCP experiment data
  2. Prepare and aggregate data
  3. Train and evaluate ML models
  4. Generate all visualizations
  5. Run the predictor demo
  6. Execute test suite

Usage:
    python main.py              # Run full pipeline
    python main.py --skip-tests # Skip test suite

Author: Rohan Kumar (24BDS0463) & Shreyans Bohra (24BCT0013)
"""

import argparse
import sys
import os
import time
from datetime import datetime


def main(skip_tests=False):
    start_time = time.time()
    
    print("\n" + "+" + "=" * 68 + "+")
    print("|" + " " * 8 + "ADAPTIVE TCP CONGESTION CONTROL DECISION ENGINE" + " " * 13 + "|")
    print("|" + " " * 8 + "Complete Pipeline Runner" + " " * 37 + "|")
    print("|" + " " * 8 + f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}" + " " * 27 + "|")
    print("+" + "=" * 68 + "+\n")
    
    # ===========================================================
    # STEP 1: Generate Synthetic Data
    # ===========================================================
    print("\n" + "=" * 70)
    print("  STEP 1/6: Generating Synthetic Experiment Data")
    print("=" * 70)
    from generate_synthetic_data import generate_and_save
    df_raw = generate_and_save(mode="full")
    print(f"  → Generated {len(df_raw)} raw experiment rows")
    
    # ===========================================================
    # STEP 2: Data Preparation
    # ===========================================================
    print("\n" + "=" * 70)
    print("  STEP 2/6: Data Preparation & Aggregation")
    print("=" * 70)
    from data_preparation import prepare_data
    df_ml = prepare_data()
    if df_ml is None:
        print("[FATAL] Data preparation failed!")
        sys.exit(1)
    print(f"  → Aggregated to {len(df_ml)} network conditions")
    
    # ===========================================================
    # STEP 3: ML Pipeline
    # ===========================================================
    print("\n" + "=" * 70)
    print("  STEP 3/6: Machine Learning Pipeline")
    print("=" * 70)
    from ml_pipeline import run_ml_pipeline
    results, model = run_ml_pipeline()
    
    # ===========================================================
    # STEP 4: Visualizations
    # ===========================================================
    print("\n" + "=" * 70)
    print("  STEP 4/6: Generating Visualizations")
    print("=" * 70)
    from visualizations import generate_all_plots
    generate_all_plots()
    
    # ===========================================================
    # STEP 5: Predictor Demo
    # ===========================================================
    print("\n" + "=" * 70)
    print("  STEP 5/6: Running Predictor Demo")
    print("=" * 70)
    from predictor import load_model, run_demo
    model = load_model()
    if model is not None:
        run_demo(model)
    
    # ===========================================================
    # STEP 6: Test Suite
    # ===========================================================
    if not skip_tests:
        print("\n" + "=" * 70)
        print("  STEP 6/6: Running Test Suite")
        print("=" * 70)
        from test_cases import TestRunner
        from test_cases import (
            test_tc01, test_tc02, test_tc03, test_tc04, test_tc05,
            test_tc06, test_tc07, test_tc08, test_tc09, test_tc10
        )
        
        runner = TestRunner()
        tests = [
            ("TC-01", "TCP variant availability", test_tc01),
            ("TC-02", "Topology structure", test_tc02),
            ("TC-03", "Network condition config", test_tc03),
            ("TC-04", "Benchmark per variant", test_tc04),
            ("TC-05", "Repetition & aggregation", test_tc05),
            ("TC-06", "Dataset B generation", test_tc06),
            ("TC-07", "Feature columns (no leakage)", test_tc07),
            ("TC-08", "Held-out evaluation", test_tc08),
            ("TC-09", "Predictor output", test_tc09),
            ("TC-10", "Baseline comparison", test_tc10),
        ]
        for tid, desc, fn in tests:
            runner.run_test(tid, desc, fn)
        runner.summary()
    
    # ===========================================================
    # SUMMARY
    # ===========================================================
    elapsed = time.time() - start_time
    
    print("\n" + "+" + "=" * 68 + "+")
    print("|" + " " * 18 + "PIPELINE COMPLETE" + " " * 33 + "|")
    print("+" + "=" * 68 + "+")
    print(f"|  Total time: {elapsed:.1f} seconds" + " " * (54 - len(f"{elapsed:.1f}")) + "|")
    print(f"|  Data:       data/dataset_a_raw.csv, data/dataset_b_ml.csv" + " " * 8 + "|")
    print(f"|  Models:     models/random_forest.joblib" + " " * 28 + "|")
    print(f"|  Plots:      plots/ (9 visualization files)" + " " * 24 + "|")
    print(f"|  Results:    results/evaluation_results.json" + " " * 24 + "|")
    print(f"|  Tests:      results/test_results.json" + " " * 29 + "|")
    print("+" + "=" * 68 + "+")
    
    # Print key results
    if results:
        rf = results.get("random_forest", {})
        comp = results.get("baseline_comparison", {})
        print(f"\n  Key Results:")
        print(f"  +- Random Forest Accuracy: {rf.get('accuracy', 'N/A')}")
        print(f"  +- Random Forest F1 Score: {rf.get('f1_score', 'N/A')}")
        print(f"  +- Always Reno:  {comp.get('always_reno_tp', 'N/A')} Mbps")
        print(f"  +- Always Cubic: {comp.get('always_cubic_tp', 'N/A')} Mbps")
        print(f"  +- Always BBR:   {comp.get('always_bbr_tp', 'N/A')} Mbps")
        print(f"  +- ML Adaptive:  {comp.get('ml_adaptive_tp', 'N/A')} Mbps")
        print(f"  +- Improvement:  {comp.get('improvement_over_best_fixed_pct', 'N/A')}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the complete TCP Decision Engine pipeline")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running test suite")
    args = parser.parse_args()
    
    main(skip_tests=args.skip_tests)
