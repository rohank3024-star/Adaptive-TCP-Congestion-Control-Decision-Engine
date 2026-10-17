#!/usr/bin/env python3
"""
Module: Experiment Orchestrator
================================
End-to-end pipeline runner that ties together M1–M3.
Supports pilot mode (54 runs) and full mode (1350 runs).

Usage (requires root on Linux / WSL2):
    sudo python3 run_experiment.py --mode pilot
    sudo python3 run_experiment.py --mode full
    sudo python3 run_experiment.py --mode pilot --repetitions 3

Author: Rohan Kumar (24BDS0463) – Networking & Experimentation
"""

import argparse
import time
import sys
import os
from datetime import datetime

from network_config import (
    generate_smoke_grid, generate_pilot_grid, generate_full_grid,
    apply_condition, clear_condition, measure_baseline_rtt
)
from benchmark_runner import (
    init_csv, benchmark_condition,
    start_iperf_server, stop_iperf_server, RAW_CSV_FILE
)
from mininet_topology import create_topology, stop_topology, verify_tcp_variants


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_experiments(mode="pilot", repetitions=3, csv_path=RAW_CSV_FILE):
    """
    Run the complete experiment pipeline.

    Parameters
    ----------
    mode        : str – 'smoke' (1 condition), 'pilot' (18), or 'full' (150)
    repetitions : int – number of repetitions per condition (default 3)
    csv_path    : str – output CSV path
    """
    print("=" * 70)
    print(f"  Adaptive TCP Decision Engine – Experiment Runner")
    print(f"  Mode: {mode.upper()} | Repetitions: {repetitions}")
    print(f"  Started: {datetime.now().isoformat()}")
    print("=" * 70)

    # --- Step 1: Verify TCP variants ---
    print("\n[STEP 1] Verifying TCP variant availability...")
    if not verify_tcp_variants():
        print("[FATAL] Cannot proceed without all TCP variants.")
        sys.exit(1)

    # --- Step 2: Generate parameter grid ---
    print(f"\n[STEP 2] Generating {mode} parameter grid...")
    grid_generators = {
        "smoke": generate_smoke_grid,
        "pilot": generate_pilot_grid,
        "full": generate_full_grid,
    }
    conditions = grid_generators[mode]()
    total_runs = len(conditions) * 3 * repetitions  # 3 TCP variants
    print(f"         {len(conditions)} conditions × 3 variants × {repetitions} reps = {total_runs} runs")
    est_time_min = total_runs * 15 / 60  # ~15s per run
    print(f"         Estimated time: {est_time_min:.0f}–{est_time_min * 1.3:.0f} minutes")

    # --- Step 3: Create topology ---
    print("\n[STEP 3] Creating Mininet topology (H1 – S1 – H2)...")
    result = create_topology()
    if result is None:
        print("[FATAL] Failed to create topology.")
        sys.exit(1)
    net, h1, h2 = result

    # --- Step 4: Initialize CSV ---
    print(f"\n[STEP 4] Initializing CSV at {csv_path}...")
    init_csv(csv_path)

    # --- Step 5: Start iperf3 server ---
    print("\n[STEP 5] Starting iperf3 server on h2...")
    start_iperf_server(h2)

    # --- Step 6: Run experiments ---
    print(f"\n[STEP 6] Running experiments ({total_runs} total)...")
    completed = 0
    failed = 0
    start_time = time.time()

    for i, condition in enumerate(conditions, 1):
        bw = condition["bandwidth_mbps"]
        delay = condition["delay_ms"]
        loss = condition["loss_pct"]

        print(f"\n--- Condition {i}/{len(conditions)}: "
              f"BW={bw}Mbps, Delay={delay}ms, Loss={loss}% ---")

        # Apply network condition
        apply_condition(h1, h2, bw, delay, loss)
        time.sleep(0.5)

        # Measure baseline RTT
        baseline_rtt = measure_baseline_rtt(h1, h2.IP())

        for rep in range(1, repetitions + 1):
            print(f"  [Rep {rep}/{repetitions}]")
            try:
                # Restart iperf server between reps for clean state
                stop_iperf_server(h2)
                time.sleep(0.3)
                start_iperf_server(h2)
                time.sleep(0.3)

                rows = benchmark_condition(
                    h1, h2, bw, delay, loss,
                    baseline_rtt, run_id=rep, csv_path=csv_path
                )
                completed += len(rows)
                failed += 3 - len(rows)
            except Exception as e:
                print(f"  [ERROR] Experiment failed: {e}")
                failed += 3

        # Clear conditions before next
        clear_condition(h1, h2)

    elapsed = time.time() - start_time

    # --- Step 7: Cleanup ---
    print("\n[STEP 7] Cleaning up...")
    stop_iperf_server(h2)
    stop_topology(net)

    # --- Summary ---
    print("\n" + "=" * 70)
    print("  EXPERIMENT COMPLETE")
    print(f"  Total time:     {elapsed / 60:.1f} minutes")
    print(f"  Completed runs: {completed}")
    print(f"  Failed runs:    {failed}")
    print(f"  Output CSV:     {csv_path}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run TCP congestion control experiments"
    )
    parser.add_argument(
        "--mode", choices=["smoke", "pilot", "full"], default="pilot",
        help="Experiment mode: smoke (3 runs), pilot (54 runs), or full (1350 runs)"
    )
    parser.add_argument(
        "--repetitions", type=int, default=3,
        help="Number of repetitions per condition (default: 3)"
    )
    parser.add_argument(
        "--output", type=str, default=RAW_CSV_FILE,
        help=f"Output CSV file path (default: {RAW_CSV_FILE})"
    )
    args = parser.parse_args()

    run_experiments(
        mode=args.mode,
        repetitions=args.repetitions,
        csv_path=args.output
    )
