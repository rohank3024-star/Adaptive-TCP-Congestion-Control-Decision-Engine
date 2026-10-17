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
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TCP_VARIANTS       = ["reno", "cubic", "bbr"]
IPERF_DURATION_SEC = 10          # seconds per iperf3 run
IPERF_PORT         = 5201
IPERF_SERVER_LOG   = "/tmp/cn_project_iperf3_server.log"
RAW_CSV_FILE       = "data/dataset_a_mininet.csv"
RAW_CSV_HEADERS    = [
    "timestamp", "bandwidth_mbps", "delay_ms", "loss_pct",
    "tcp_variant", "throughput_mbps", "retransmissions",
    "baseline_rtt_ms", "iperf_rtt_ms", "throughput_source", "run_id"
]


# ---------------------------------------------------------------------------
# iperf3 server management
# ---------------------------------------------------------------------------
def start_iperf_server(host, port=IPERF_PORT):
    """Start iperf3 in server mode on the receiver host."""
    host.cmd(f"iperf3 -s -p {port} -D --logfile {IPERF_SERVER_LOG}")
    time.sleep(0.5)  # Let it start up
    print(f"[IPERF] Server started on {host.name}:{port}")


def stop_iperf_server(host):
    """Kill iperf3 server on the receiver host."""
    host.cmd("pkill -f 'iperf3 -s' 2>/dev/null")
    time.sleep(0.3)


def _read_receiver_log_since(offset):
    """Read iperf server output written after a client's start time."""
    try:
        with open(IPERF_SERVER_LOG, "rb") as handle:
            handle.seek(offset)
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _receiver_rate_bps(server_log):
    """Parse the receiver-side total line from iperf3 server text output."""
    matches = re.findall(
        r"\b([0-9]+(?:\.[0-9]+)?)\s*([KMGT]?)bits/sec\s+receiver\b",
        server_log,
        flags=re.IGNORECASE,
    )
    if not matches:
        return None
    amount, unit = matches[-1]
    multiplier = {"": 1, "k": 1_000, "m": 1_000_000,
                  "g": 1_000_000_000, "t": 1_000_000_000_000}[unit.lower()]
    return float(amount) * multiplier


def _receiver_interval_rate_bps(server_log, minimum_duration=8.0):
    """Aggregate server-side iperf interval bytes over observed interval time."""
    interval_pattern = re.compile(
        r"\]\s*([0-9]+(?:\.[0-9]+)?)-([0-9]+(?:\.[0-9]+)?)\s+sec\s+"
        r"([0-9]+(?:\.[0-9]+)?)\s*(Bytes|KBytes|MBytes|GBytes)\s+"
        r"[0-9]+(?:\.[0-9]+)?\s*[KMGT]?bits/sec",
        flags=re.IGNORECASE,
    )
    byte_multiplier = {"bytes": 1, "kbytes": 1024, "mbytes": 1024**2,
                       "gbytes": 1024**3}
    total_bytes = total_seconds = 0.0
    for start, end, amount, unit in interval_pattern.findall(server_log):
        duration = float(end) - float(start)
        if duration <= 0:
            continue
        total_seconds += duration
        total_bytes += float(amount) * byte_multiplier[unit.lower()]
    if total_seconds < minimum_duration or total_bytes <= 0:
        return None
    return total_bytes * 8 / total_seconds


def _last_receiver_session(server_log):
    """Discard earlier iperf server sessions when parsing a shared logfile."""
    sessions = re.split(r"Accepted connection from .*", server_log)
    return sessions[-1] if len(sessions) > 1 else server_log


# ---------------------------------------------------------------------------
# Run single iperf3 benchmark
# ---------------------------------------------------------------------------
def run_iperf_benchmark(sender, receiver_ip, tcp_variant,
                        duration=IPERF_DURATION_SEC, port=IPERF_PORT,
                        server_host=None):
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
    dict with keys: throughput_mbps, retransmissions, iperf_rtt_ms.
    ``iperf_rtt_ms`` is an outcome measured after selecting the variant; it
    must not be used as a predictor input.
    or None on failure
    """
    # Set congestion control algorithm on sender
    sender.cmd(f"sysctl -w net.ipv4.tcp_congestion_control={tcp_variant}")
    time.sleep(0.2)

    try:
        server_log_offset = Path(IPERF_SERVER_LOG).stat().st_size
    except OSError:
        server_log_offset = 0

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
        intervals = data.get("intervals", [])
        sum_sent = end.get("sum_sent", {})
        sum_received = end.get("sum_received", {})

        # Use receiver-side throughput (more accurate than sender)
        throughput_bps = sum_received.get("bits_per_second", 0)
        throughput_mbps = round(throughput_bps / 1_000_000, 2)
        retransmissions = sum_sent.get("retransmits", 0)

        # RTT from streams (mean RTT in microseconds)
        streams = end.get("streams", [{}])
        if streams and "sender" in streams[0]:
            mean_rtt_us = streams[0]["sender"].get("mean_rtt", 0)
            iperf_rtt_ms = round(mean_rtt_us / 1000, 2)
        else:
            iperf_rtt_ms = -1.0

        if throughput_bps > 0:
            result = {
                "throughput_mbps": throughput_mbps,
                "retransmissions": retransmissions,
                "iperf_rtt_ms": iperf_rtt_ms,
                "throughput_source": "iperf_client_json_receiver",
            }
            print(f"    [{tcp_variant.upper():5s}] Throughput: {throughput_mbps:8.2f} Mbps | "
                  f"Retransmits: {retransmissions:5d} | iperf RTT: {iperf_rtt_ms:.2f} ms")
            return result

        server_log = _read_receiver_log_since(server_log_offset)
        server_log = _last_receiver_session(server_log)
        receiver_bps = _receiver_rate_bps(server_log)
        source = "iperf_server_log_receiver_summary"
        if receiver_bps is None and server_host is not None:
            server_log = server_host.cmd(f"tail -n 200 {IPERF_SERVER_LOG}")
            server_log = _last_receiver_session(server_log)
            receiver_bps = _receiver_rate_bps(server_log)
            if receiver_bps is None:
                receiver_bps = _receiver_interval_rate_bps(server_log)
                source = "iperf_server_log_receiver_intervals"
        interval_retransmissions = [
            item["sum"]["retransmits"] for item in intervals
            if isinstance(item.get("sum"), dict) and "retransmits" in item["sum"]
        ]
        if receiver_bps and interval_retransmissions:
            recovered_throughput = round(receiver_bps / 1_000_000, 2)
            recovered_retransmissions = int(sum(interval_retransmissions))
            print(f"[RECOVERED] {tcp_variant.upper()} receiver throughput from server log: "
                  f"{recovered_throughput:.2f} Mbps; RTT summary unavailable.")
            return {
                "throughput_mbps": recovered_throughput,
                "retransmissions": recovered_retransmissions,
                "iperf_rtt_ms": iperf_rtt_ms,
                "throughput_source": source,
            }

        if throughput_mbps <= 0:
            interval_sums = [item.get("sum", {}) for item in intervals]
            print(f"[ERROR] iperf3 returned incomplete receiver metrics for {tcp_variant}; "
                  f"the run will not be recorded (top-level keys={list(data)}, "
                  f"iperf error={data.get('error')!r}, end keys={list(end)}, "
                  f"sent={sum_sent}, received={sum_received}, "
                  f"intervals={len(intervals)}, first interval sum="
                  f"{interval_sums[0] if interval_sums else None}).")
            if server_host is not None:
                server_log = server_host.cmd(f"tail -n 25 {IPERF_SERVER_LOG}")
                print(f"[IPERF SERVER LOG - LAST 25 LINES]\n{server_log}")
            return None

        result = {
            "throughput_mbps": throughput_mbps,
            "retransmissions": retransmissions,
            "iperf_rtt_ms": iperf_rtt_ms,
            "throughput_source": "iperf_client_json_receiver",
        }
        print(f"    [{tcp_variant.upper():5s}] Throughput: {throughput_mbps:8.2f} Mbps | "
              f"Retransmits: {retransmissions:5d} | iperf RTT: {iperf_rtt_ms:.2f} ms")
        return result

    except (KeyError, IndexError, TypeError) as e:
        print(f"[ERROR] Failed to extract metrics for {tcp_variant}: {e}")
        return None


# ---------------------------------------------------------------------------
# CSV logging
# ---------------------------------------------------------------------------
def init_csv(filepath=RAW_CSV_FILE):
    """Create a Mininet-results CSV, refusing incompatible existing files."""
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
    if not os.path.exists(filepath):
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(RAW_CSV_HEADERS)
        print(f"[CSV] Created {filepath}")
        return

    with open(filepath, newline="") as f:
        existing_headers = next(csv.reader(f), [])
    if existing_headers != RAW_CSV_HEADERS:
        raise ValueError(
            f"Refusing to append to {filepath}: its schema does not match "
            "the validated Mininet schema. Choose a new --output path."
        )


def append_to_csv(row_dict, filepath=RAW_CSV_FILE):
    """Append a single experiment result row to the raw CSV."""
    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_CSV_HEADERS)
        writer.writerow(row_dict)


# ---------------------------------------------------------------------------
# Run benchmarks for one network condition
# ---------------------------------------------------------------------------
def benchmark_condition(h1, h2, bandwidth_mbps, delay_ms, loss_pct,
                        baseline_rtt, run_id=1, csv_path=RAW_CSV_FILE,
                        variants=None):
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
    variants       : iterable of TCP variants; defaults to all project variants

    Returns
    -------
    list of dicts – one per variant with all metrics
    """
    results = []
    timestamp = datetime.now().isoformat()

    if baseline_rtt <= 0:
        print("[ERROR] Invalid baseline ping RTT; condition was not recorded.")
        return results

    if variants is None:
        variants = TCP_VARIANTS

    for variant in variants:
        result = run_iperf_benchmark(h1, h2.IP(), variant, server_host=h2)
        if result is None:
            print(f"[ERROR] {variant.upper()} failed; no synthetic zero row was recorded.")
            continue

        row = {
            "timestamp": timestamp,
            "bandwidth_mbps": bandwidth_mbps,
            "delay_ms": delay_ms,
            "loss_pct": loss_pct,
            "tcp_variant": variant,
            "throughput_mbps": result["throughput_mbps"],
            "retransmissions": result["retransmissions"],
            "baseline_rtt_ms": baseline_rtt,
            "iperf_rtt_ms": result["iperf_rtt_ms"],
            "throughput_source": result["throughput_source"],
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
