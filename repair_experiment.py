#!/usr/bin/env python3
"""Repair incomplete Mininet observations without altering the source CSV.

The script copies all valid rows from an existing experiment CSV into a new
CSV, then reruns only rows with missing receiver-side iperf metrics. It is
intended for transient benchmark failures, not to discard unfavorable results.
"""

import argparse
import csv
import os
import sys
import time
from collections import defaultdict

from benchmark_runner import RAW_CSV_HEADERS, benchmark_condition
from mininet_topology import create_topology, stop_topology, verify_tcp_variants
from network_config import apply_condition, clear_condition, measure_baseline_rtt
from benchmark_runner import start_iperf_server, stop_iperf_server

CONDITION_COLS = ["bandwidth_mbps", "delay_ms", "loss_pct"]
LEGACY_RAW_CSV_HEADERS = [column for column in RAW_CSV_HEADERS if column != "throughput_source"]


def observation_key(row):
    """Canonical key tolerant of CSV spellings such as 1 and 1.0."""
    return (float(row["bandwidth_mbps"]), float(row["delay_ms"]),
            float(row["loss_pct"]), row["tcp_variant"].lower(), int(row["run_id"]))


def is_invalid(row):
    """Whether a row lacks usable receiver-side iperf measurements."""
    return (float(row["throughput_mbps"]) <= 0
            or float(row["retransmissions"]) < 0
            or float(row["baseline_rtt_ms"]) <= 0)


def write_valid_rows(rows, output_path):
    """Create the repair output containing only valid original observations."""
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_CSV_HEADERS)
        writer.writeheader()
        for row in rows:
            output_row = dict(row)
            output_row.setdefault("throughput_source", "iperf_client_json_receiver")
            writer.writerow(output_row)


def repair(input_path, output_path, attempts=3, max_missing=None):
    with open(input_path, newline="") as handle:
        reader = csv.DictReader(handle)
        input_headers = reader.fieldnames or []
        if input_headers not in (RAW_CSV_HEADERS, LEGACY_RAW_CSV_HEADERS):
            missing = [column for column in LEGACY_RAW_CSV_HEADERS if column not in input_headers]
            raise ValueError(f"Input CSV is missing columns: {missing}")
        rows = list(reader)

    expected = {observation_key(row) for row in rows}
    if len(expected) != len(rows):
        raise ValueError("Input CSV contains duplicate condition/variant/run keys.")

    existing = []
    if os.path.exists(output_path):
        with open(output_path, newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != RAW_CSV_HEADERS:
                raise ValueError(f"Existing output has an incompatible schema: {output_path}")
            existing = list(reader)
        if any(is_invalid(row) for row in existing):
            raise ValueError("Existing output contains invalid rows; move it aside before resuming.")

    existing_keys = [observation_key(row) for row in existing]
    if len(existing_keys) != len(set(existing_keys)):
        raise ValueError("Existing repair output has duplicate condition/variant/run keys.")
    if not set(existing_keys).issubset(expected):
        raise ValueError("Existing repair output contains keys not present in the input.")

    valid_input = [row for row in rows if not is_invalid(row)]
    if not existing:
        write_valid_rows(valid_input, output_path)
        existing = valid_input
        existing_keys = [observation_key(row) for row in existing]
        print(f"[CSV] Copied {len(existing)} valid source rows to {output_path}")

    missing_keys = expected - set(existing_keys)
    bad = [row for row in rows if observation_key(row) in missing_keys]
    if max_missing is not None:
        bad = bad[:max_missing]
    print(f"[LOAD] {len(rows)} expected rows: {len(valid_input)} valid in source, "
          f"{len(missing_keys)} missing or invalid, {len(existing)} already present")
    if max_missing is not None:
        print(f"[DIAGNOSTIC] Attempting at most {len(bad)} missing observations")
    if not bad:
        print(f"[DONE] Complete dataset: {len(existing)} rows in {output_path}")
        return

    if not verify_tcp_variants():
        raise RuntimeError("Required TCP variants are unavailable")

    topology = create_topology()
    if topology is None:
        raise RuntimeError("Could not create Mininet topology")
    net, h1, h2 = topology
    start_iperf_server(h2)
    repaired = 0

    try:
        grouped = defaultdict(list)
        for row in bad:
            condition = tuple(float(row[column]) for column in CONDITION_COLS)
            grouped[condition].append(row)

        for (bandwidth, delay, loss), group in sorted(grouped.items()):
            print(f"\n[REPAIR] BW={bandwidth}Mbps, Delay={delay}ms, Loss={loss}%")
            apply_condition(h1, h2, bandwidth, delay, loss)
            time.sleep(0.5)
            baseline_rtt = measure_baseline_rtt(h1, h2.IP())
            if baseline_rtt <= 0:
                print("[ERROR] Invalid baseline RTT; this condition remains unrepaired.")
                clear_condition(h1, h2)
                continue

            by_run = defaultdict(list)
            for row in group:
                by_run[int(row["run_id"])].append(row["tcp_variant"])
            for run_id, variants in sorted(by_run.items()):
                for variant in variants:
                    key = (str(int(bandwidth)) if bandwidth.is_integer() else str(bandwidth),
                           str(int(delay)) if delay.is_integer() else str(delay),
                           str(int(loss)) if loss.is_integer() else str(loss),
                           variant, str(run_id))
                    for attempt in range(1, attempts + 1):
                        stop_iperf_server(h2)
                        start_iperf_server(h2)
                        rows_written = benchmark_condition(
                            h1, h2, bandwidth, delay, loss, baseline_rtt,
                            run_id=run_id, csv_path=output_path, variants=[variant],
                        )
                        repaired += len(rows_written)
                        if rows_written:
                            break
                        print(f"[RETRY] {variant} run {run_id}: attempt {attempt}/{attempts} failed")
            clear_condition(h1, h2)
    finally:
        stop_iperf_server(h2)
        stop_topology(net)

    with open(output_path, newline="") as handle:
        repaired_rows = list(csv.DictReader(handle))
    output_keys = {observation_key(row) for row in repaired_rows}
    remaining = expected - output_keys
    print(f"\n[REPAIR] Newly recorded rows: {repaired}. Output: {output_path}")
    print(f"         Present rows: {len(output_keys)}/{len(expected)}; missing rows: {len(remaining)}")
    if remaining:
        print("         Rerun this command to resume after resolving the iperf failure.")
        sys.exit(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rerun only incomplete Mininet observations.")
    parser.add_argument("--input", required=True, help="Original experiment CSV (preserved)")
    parser.add_argument("--output", required=True, help="New repaired experiment CSV")
    parser.add_argument("--attempts", type=int, default=3,
                        help="Attempts per missing observation (default: 3)")
    parser.add_argument("--max-missing", type=int, default=None,
                        help="Limit repaired observations (diagnostics; default: all)")
    args = parser.parse_args()
    if args.attempts < 1:
        parser.error("--attempts must be at least 1")
    if args.max_missing is not None and args.max_missing < 1:
        parser.error("--max-missing must be at least 1")
    repair(args.input, args.output, args.attempts, args.max_missing)
