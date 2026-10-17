#!/usr/bin/env python3
"""
Pantheon Real-World Dataset Fetcher
=====================================
Downloads and parses TCP congestion control performance data from
Stanford's Pantheon platform (pantheon.stanford.edu).

Pantheon is a peer-reviewed benchmarking platform (Best Paper, USENIX ATC '18)
that tests multiple congestion control algorithms across real-world network
paths globally.

This script:
  1. Fetches performance JSON files from Pantheon's S3 archive
  2. Parses throughput (Kbps), delay (ms), and loss rate per algorithm
  3. Structures data into a CSV for our ML pipeline
  4. Filters for algorithms relevant to our project (cubic, bbr, vegas/reno-like)

Output: data/dataset_pantheon_real.csv

Author: Rohan Kumar (24BDS0463) – Networking & Experimentation
"""

import json
import csv
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUTPUT_CSV = "data/dataset_pantheon_real.csv"
CSV_HEADERS = [
    "source", "network_path", "run_id", "flow_id",
    "algorithm", "throughput_kbps", "delay_ms", "loss_rate",
    "throughput_mbps"
]

# Algorithms we care about for our project
TARGET_ALGORITHMS = [
    "cubic", "bbr", "vegas",       # Direct matches to our study
    "reno",                         # If available
    "copa", "vivace", "pcc",        # Other well-known algorithms
    "indigo", "ledbat", "sprout",   # ML/delay-based algorithms
    "fillp", "fillp_sheep",         # High-throughput protocols
    "verus", "scream", "quic",      # Other transport protocols
    "taova",                        # Adaptive algorithm
]

# Pantheon experiment results on S3
# Format: s3.amazonaws.com/stanford-pantheon/{type}/{node}/reports/{date}-{path}-pantheon-perf.json
PANTHEON_EXPERIMENTS = [
    # GCE (Google Cloud) paths — controlled datacenter environments
    {
        "name": "GCE-London-to-Iowa",
        "url": "https://s3.amazonaws.com/stanford-pantheon/real-world/GCE-Iowa/reports/2020-04-17T10-16-GCE-London-to-GCE-Iowa-5-runs-3-flows-pantheon-perf.json",
        "path_type": "datacenter",
        "description": "Google Cloud London to Iowa (transatlantic)"
    },
    {
        "name": "GCE-London-to-Sydney",
        "url": "https://s3.amazonaws.com/stanford-pantheon/real-world/GCE-Sydney/reports/2020-04-17T07-04-GCE-London-to-GCE-Sydney-5-runs-3-flows-pantheon-perf.json",
        "path_type": "datacenter",
        "description": "Google Cloud London to Sydney (intercontinental)"
    },
    {
        "name": "GCE-London-to-Tokyo",
        "url": "https://s3.amazonaws.com/stanford-pantheon/real-world/GCE-Tokyo/reports/2020-04-17T07-06-GCE-London-to-GCE-Tokyo-5-runs-3-flows-pantheon-perf.json",
        "path_type": "datacenter",
        "description": "Google Cloud London to Tokyo (intercontinental)"
    },
    {
        "name": "GCE-London-to-Brazil",
        "url": "https://s3.amazonaws.com/stanford-pantheon/real-world/GCE-Brazil/reports/2020-04-17T07-20-GCE-London-to-GCE-Brazil-5-runs-3-flows-pantheon-perf.json",
        "path_type": "datacenter",
        "description": "Google Cloud London to Brazil (intercontinental)"
    },
    # AWS paths
    {
        "name": "AWS-California-to-India",
        "url": "https://s3.amazonaws.com/stanford-pantheon/real-world/AWS-India/reports/2020-04-17T07-05-AWS-California-to-AWS-India-5-runs-3-flows-pantheon-perf.json",
        "path_type": "datacenter",
        "description": "AWS California to India (long-haul)"
    },
    {
        "name": "AWS-California-to-Korea",
        "url": "https://s3.amazonaws.com/stanford-pantheon/real-world/AWS-Korea/reports/2020-04-17T06-57-AWS-California-to-AWS-Korea-5-runs-3-flows-pantheon-perf.json",
        "path_type": "datacenter",
        "description": "AWS California to Korea (trans-Pacific)"
    },
    # Cellular/Wired paths — real end-user conditions
    {
        "name": "Stanford-to-Brazil-cellular",
        "url": "https://s3.amazonaws.com/stanford-pantheon/real-world/Brazil/reports/2020-04-15T04-36-Stanford-to-Brazil-1-runs-1-flows-pantheon-perf.json",
        "path_type": "cellular",
        "description": "Stanford to Brazil cellular (real ISP conditions)"
    },
    {
        "name": "Stanford-to-Colombia-cellular",
        "url": "https://s3.amazonaws.com/stanford-pantheon/real-world/Colombia/reports/2020-04-15T07-02-Stanford-to-Colombia-1-runs-1-flows-pantheon-perf.json",
        "path_type": "cellular",
        "description": "Stanford to Colombia cellular (real ISP conditions)"
    },
    {
        "name": "Stanford-to-India-cellular",
        "url": "https://s3.amazonaws.com/stanford-pantheon/real-world/India/reports/2020-04-15T06-09-Stanford-to-India-1-runs-1-flows-pantheon-perf.json",
        "path_type": "cellular",
        "description": "Stanford to India cellular (real ISP conditions)"
    },
    {
        "name": "Stanford-to-Mexico-cellular",
        "url": "https://s3.amazonaws.com/stanford-pantheon/real-world/Mexico/reports/2020-04-06T07-00-Stanford-to-Mexico-1-runs-1-flows-pantheon-perf.json",
        "path_type": "cellular",
        "description": "Stanford to Mexico cellular (real ISP conditions)"
    },
    {
        "name": "Stanford-to-China-cellular",
        "url": "https://s3.amazonaws.com/stanford-pantheon/real-world/China/reports/2020-03-01T04-00-Stanford-to-China-1-runs-1-flows-pantheon-perf.json",
        "path_type": "cellular",
        "description": "Stanford to China cellular (real ISP conditions)"
    },
]


# ---------------------------------------------------------------------------
# Fetch and parse
# ---------------------------------------------------------------------------
def fetch_json(url, timeout=30):
    """Download JSON from a URL."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"  [WARN] HTTP {e.code} for {url}")
        return None
    except urllib.error.URLError as e:
        print(f"  [WARN] URL error for {url}: {e.reason}")
        return None
    except Exception as e:
        print(f"  [WARN] Failed to fetch {url}: {e}")
        return None


def parse_pantheon_json(data, experiment_name, path_type):
    """
    Parse Pantheon performance JSON into structured rows.

    Pantheon JSON format:
    {
        "algorithm_name": {
            "run_id": {
                "flow_id": {
                    "tput": throughput_kbps,
                    "delay": delay_ms,
                    "loss": loss_fraction
                },
                "all": { aggregated across flows }
            }
        }
    }

    Returns list of dicts matching CSV_HEADERS.
    """
    rows = []

    for algorithm, runs in data.items():
        # Normalize algorithm name
        alg_lower = algorithm.lower().strip()

        for run_id, flows in runs.items():
            for flow_id, metrics in flows.items():
                if not isinstance(metrics, dict):
                    continue

                tput_kbps = metrics.get("tput", 0)
                delay_ms = metrics.get("delay", -1)
                loss_rate = metrics.get("loss", 0)

                rows.append({
                    "source": experiment_name,
                    "network_path": path_type,
                    "run_id": run_id,
                    "flow_id": flow_id,
                    "algorithm": alg_lower,
                    "throughput_kbps": round(tput_kbps, 2),
                    "delay_ms": round(delay_ms, 2),
                    "loss_rate": round(loss_rate, 6),
                    "throughput_mbps": round(tput_kbps / 1000, 4)
                })

    return rows


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def fetch_all_experiments():
    """Fetch all Pantheon experiment data and save to CSV."""
    print("=" * 70)
    print("  Pantheon Real-World Dataset Fetcher")
    print("  Stanford University — USENIX ATC '18 Best Paper")
    print("=" * 70)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    all_rows = []
    successful = 0
    failed = 0

    for i, experiment in enumerate(PANTHEON_EXPERIMENTS, 1):
        name = experiment["name"]
        url = experiment["url"]
        path_type = experiment["path_type"]
        desc = experiment["description"]

        print(f"\n[{i}/{len(PANTHEON_EXPERIMENTS)}] {name}")
        print(f"  Description: {desc}")
        print(f"  URL: {url[:80]}...")

        data = fetch_json(url)
        if data is None:
            failed += 1
            continue

        rows = parse_pantheon_json(data, name, path_type)
        all_rows.extend(rows)
        successful += 1

        # Count algorithms in this experiment
        algorithms = set(r["algorithm"] for r in rows)
        print(f"  ✓ Parsed {len(rows)} records, {len(algorithms)} algorithms")
        print(f"    Algorithms: {', '.join(sorted(algorithms))}")

        # Be polite to the server
        time.sleep(0.5)

    # Write CSV
    print(f"\n{'=' * 70}")
    print(f"  Writing CSV...")

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"  ✓ Saved {len(all_rows)} total records to {OUTPUT_CSV}")

    # Summary statistics
    if all_rows:
        import pandas as pd
        df = pd.DataFrame(all_rows)

        print(f"\n{'=' * 70}")
        print(f"  DATASET SUMMARY")
        print(f"{'=' * 70}")
        print(f"  Total records:     {len(df)}")
        print(f"  Experiments:       {successful} succeeded, {failed} failed")
        print(f"  Network paths:     {df['source'].nunique()}")
        print(f"  Algorithms:        {df['algorithm'].nunique()}")
        print(f"\n  Records per algorithm:")
        algo_counts = df.groupby("algorithm").size().sort_values(ascending=False)
        for alg, count in algo_counts.items():
            marker = " ◄" if alg in ["cubic", "bbr", "vegas"] else ""
            print(f"    {alg:20s}: {count:5d} records{marker}")

        # Key algorithms summary
        key_algs = df[df["algorithm"].isin(["cubic", "bbr", "vegas"])]
        if len(key_algs) > 0:
            print(f"\n  Key Algorithms Performance Summary:")
            summary = key_algs.groupby("algorithm").agg(
                avg_throughput_mbps=("throughput_mbps", "mean"),
                avg_delay_ms=("delay_ms", "mean"),
                avg_loss=("loss_rate", "mean"),
                count=("throughput_mbps", "count")
            ).round(4)
            print(summary.to_string())

        print(f"\n  Throughput range: [{df['throughput_mbps'].min():.2f}, "
              f"{df['throughput_mbps'].max():.2f}] Mbps")
        print(f"  Delay range:     [{df['delay_ms'].min():.1f}, "
              f"{df['delay_ms'].max():.1f}] ms")

    print(f"\n{'=' * 70}")
    print(f"  DONE — Real-world dataset saved to {OUTPUT_CSV}")
    print(f"{'=' * 70}")

    return all_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fetch_all_experiments()
