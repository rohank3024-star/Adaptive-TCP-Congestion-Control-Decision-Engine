#!/usr/bin/env python3
"""
Module M3: Benchmark Automation
================================
Runs iperf3 benchmarks with TCP Reno, Cubic, and BBR under configured
network conditions. Parses throughput and retransmission metrics and
logs results to CSV.

Usage (requires root on Linux / WSL2):
    Imported by run_experiment.py.
    sudo python3 benchmark_runner.py   # standalone smoke test

Author: Rohan Kumar (24BDS0463) – Networking & Experimentation
"""

import subprocess
import json
import csv
import os
import re
import time
from datetime import datetime

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TCP_VARIANTS       = ["reno", "cubic", "bbr"]
IPERF_DURATION_SEC = 10          # seconds per iperf3 run
IPERF_PORT         = 5201
RAW_CSV_FILE       = "data/dataset_a_raw.csv"
RAW_CSV_HEADERS    = [
    "timestamp", "bandwidth_mbps", "delay_ms", "loss_pct",
    "tcp_variant", "throughput_mbps", "retransmissions",
    "rtt_ms", "run_id"
]


# ---------------------------------------------------------------------------
# iperf3 server management
# ---------------------------------------------------------------------------
def start_iperf_server(host, port=IPERF_PORT):
    """Start iperf3 in server mode on the receiver host."""
    host.cmd(f"iperf3 -s -p {port} -D")  # -D = daemon mode
    time.sleep(0.5)  # Let it start up
    print(f"[IPERF] Server started on {host.name}:{port}")


def stop_iperf_server(host):
    """Kill iperf3 server on the receiver host."""
    host.cmd("pkill -f 'iperf3 -s' 2>/dev/null")
    time.sleep(0.3)


# ---------------------------------------------------------------------------
# Run single iperf3 benchmark
# ---------------------------------------------------------------------------
def run_iperf_benchmark(sender, receiver_ip, tcp_variant,
                        duration=IPERF_DURATION_SEC, port=IPERF_PORT):
    """
    Run a single iperf3 test with the specified TCP variant.

    Parameters
    ----------
    sender       : Mininet host object
    receiver_ip  : str – IP of receiver
    tcp_variant  : str – 'reno', 'cubic', or 'bbr'
    duration     : int – test duration in seconds
    port         : int – iperf3 port

    Returns
    -------
    dict with keys: throughput_mbps, retransmissions, rtt_ms
    or None on failure
    """
    # Set congestion control algorithm on sender
    sender.cmd(f"sysctl -w net.ipv4.tcp_congestion_control={tcp_variant}")
    time.sleep(0.2)

    # Run iperf3 in JSON mode
    cmd = (f"iperf3 -c {receiver_ip} -p {port} -t {duration} "
           f"-C {tcp_variant} -J")
    output = sender.cmd(cmd)

    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        print(f"[ERROR] Failed to parse iperf3 JSON for {tcp_variant}")
        print(f"        Raw output (first 200 chars): {output[:200]}")
        return None

    # Extract metrics from iperf3 JSON
    try:
        end = data.get("end", {})
        sum_sent = end.get("sum_sent", {})

        throughput_bps = sum_sent.get("bits_per_second", 0)
        throughput_mbps = round(throughput_bps / 1_000_000, 2)
        retransmissions = sum_sent.get("retransmits", 0)

        # RTT from streams (mean RTT in microseconds)
        streams = end.get("streams", [{}])
        if streams and "sender" in streams[0]:
            mean_rtt_us = streams[0]["sender"].get("mean_rtt", 0)
            rtt_ms = round(mean_rtt_us / 1000, 2)
        else:
            rtt_ms = -1.0

        result = {
            "throughput_mbps": throughput_mbps,
            "retransmissions": retransmissions,
            "rtt_ms": rtt_ms
        }
        print(f"    [{tcp_variant.upper():5s}] Throughput: {throughput_mbps:8.2f} Mbps | "
              f"Retransmits: {retransmissions:5d} | RTT: {rtt_ms:.2f} ms")
        return result

    except (KeyError, IndexError, TypeError) as e:
        print(f"[ERROR] Failed to extract metrics for {tcp_variant}: {e}")
        return None


# ---------------------------------------------------------------------------
# CSV logging
# ---------------------------------------------------------------------------
def init_csv(filepath=RAW_CSV_FILE):
    """Create the raw CSV file with headers if it doesn't exist."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if not os.path.exists(filepath):
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(RAW_CSV_HEADERS)
        print(f"[CSV] Created {filepath}")


def append_to_csv(row_dict, filepath=RAW_CSV_FILE):
    """Append a single experiment result row to the raw CSV."""
    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_CSV_HEADERS)
        writer.writerow(row_dict)


# ---------------------------------------------------------------------------
# Run benchmarks for one network condition
# ---------------------------------------------------------------------------
def benchmark_condition(h1, h2, bandwidth_mbps, delay_ms, loss_pct,
                        baseline_rtt, run_id=1, csv_path=RAW_CSV_FILE):
    """
    Benchmark all three TCP variants under one network condition.

    Parameters
    ----------
    h1, h2         : Mininet host objects
    bandwidth_mbps : int
    delay_ms       : int
    loss_pct       : float
    baseline_rtt   : float – measured before benchmarking
    run_id         : int – repetition number (1, 2, or 3)
    csv_path       : str – path to raw CSV

    Returns
    -------
    list of dicts – one per variant with all metrics
    """
    results = []
    timestamp = datetime.now().isoformat()

    for variant in TCP_VARIANTS:
        result = run_iperf_benchmark(h1, h2.IP(), variant)
        if result is None:
            result = {
                "throughput_mbps": 0.0,
                "retransmissions": -1,
                "rtt_ms": -1.0
            }

        row = {
            "timestamp": timestamp,
            "bandwidth_mbps": bandwidth_mbps,
            "delay_ms": delay_ms,
            "loss_pct": loss_pct,
            "tcp_variant": variant,
            "throughput_mbps": result["throughput_mbps"],
            "retransmissions": result["retransmissions"],
            "rtt_ms": result["rtt_ms"],
            "run_id": run_id
        }
        append_to_csv(row, csv_path)
        results.append(row)

        # Brief pause between variants to let the network settle
        time.sleep(1)

    return results


# ---------------------------------------------------------------------------
# Main – standalone smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  Module M3: Benchmark Runner (Standalone Test)")
    print("=" * 60)
    print()
    print("This module is designed to be imported by run_experiment.py.")
    print("It requires Mininet and must be run with sudo on Linux/WSL2.")
    print()
    print("CSV Headers:", RAW_CSV_HEADERS)
    print("TCP Variants:", TCP_VARIANTS)
    print(f"iperf3 Duration: {IPERF_DURATION_SEC}s per run")
    print(f"Output file: {RAW_CSV_FILE}")
    
    # Initialize CSV (creates file with headers)
    init_csv()
    print("[OK] CSV initialized.")
