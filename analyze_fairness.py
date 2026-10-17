#!/usr/bin/env python3
"""Summarize paired-flow fairness experiments without mixing other datasets."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
       6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
       11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
       16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
       21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
       26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}


def mean_ci95(values):
    """Mean and two-sided 95% Student-t confidence interval (small n aware)."""
    if not values:
        return {"mean": None, "ci95_low": None, "ci95_high": None, "n": 0}
    avg = sum(values) / len(values)
    if len(values) == 1:
        low = high = None
    else:
        variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
        margin = T95.get(len(values) - 1, 1.96) * math.sqrt(variance / len(values))
        low, high = avg - margin, avg + margin
    return {"mean": avg, "ci95_low": low, "ci95_high": high, "n": len(values)}


def summarize(path, max_loaded_rtt_increase_ms=None):
    """Summarize only complete two-flow runs; skip malformed/incomplete pairs."""
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    runs = defaultdict(list)
    for row in rows:
        key = (row["timestamp"], row["run_id"], row["queue_discipline"])
        runs[key].append(row)

    grouped = defaultdict(lambda: defaultdict(list))
    variant_grouped = defaultdict(lambda: defaultdict(list))
    paired_qdisc_runs = defaultdict(dict)
    incomplete = 0
    for (_, _, qdisc), flow_rows in runs.items():
        if len(flow_rows) != 2:
            incomplete += 1
            continue
        ordered = sorted(flow_rows, key=lambda row: int(row["flow_id"]))
        if [int(row["flow_id"]) for row in ordered] != [1, 2]:
            incomplete += 1
            continue
        pair = " vs ".join(sorted((ordered[0]["tcp_variant"],
                                   ordered[1]["tcp_variant"])))
        key = (qdisc, pair)
        run_metrics = {
            "aggregate_throughput_mbps": sum(
                float(row["throughput_mbps"]) for row in ordered
            ),
            "loaded_rtt_avg_ms": sum(
                float(row["loaded_rtt_avg_ms"]) for row in ordered
            ) / 2,
            "loaded_rtt_increase_ms": sum(
                float(row["loaded_rtt_increase_ms"]) for row in ordered
            ) / 2,
            "jain_fairness": float(ordered[0]["jain_fairness"]),
        }
        for name, value in run_metrics.items():
            grouped[key][name].append(value)
        paired_qdisc_runs[(pair, ordered[0]["run_id"])][qdisc] = run_metrics
        for row in ordered:
            variant_key = (qdisc, row["tcp_variant"])
            throughput = float(row["throughput_mbps"])
            rtt = float(row["loaded_rtt_avg_ms"])
            increase = float(row["loaded_rtt_increase_ms"])
            variant_grouped[variant_key]["throughput_mbps"].append(throughput)
            variant_grouped[variant_key]["loaded_rtt_avg_ms"].append(rtt)
            variant_grouped[variant_key]["loaded_rtt_increase_ms"].append(increase)
            variant_grouped[variant_key].setdefault(
                "throughput_within_rtt_limit_mbps", []
            )
            if (max_loaded_rtt_increase_ms is not None and
                    increase <= max_loaded_rtt_increase_ms):
                variant_grouped[variant_key]["throughput_within_rtt_limit_mbps"].append(
                    throughput
                )

    summary = []
    for (qdisc, pair), metrics in sorted(grouped.items()):
        summary.append({
            "queue_discipline": qdisc,
            "variant_pair": pair,
            "complete_paired_runs": len(metrics["jain_fairness"]),
            **{name: mean_ci95(values) for name, values in metrics.items()},
        })
    per_variant = []
    for (qdisc, variant), metrics in sorted(variant_grouped.items()):
        metric_names = ("throughput_mbps", "loaded_rtt_avg_ms",
                        "loaded_rtt_increase_ms")
        entry = {
            "queue_discipline": qdisc,
            "tcp_variant": variant,
            "complete_flow_observations": len(metrics["throughput_mbps"]),
            **{name: mean_ci95(metrics[name]) for name in metric_names},
        }
        if max_loaded_rtt_increase_ms is not None:
            entry["throughput_within_rtt_limit_mbps"] = mean_ci95(
                metrics["throughput_within_rtt_limit_mbps"]
            )
            entry["max_loaded_rtt_increase_ms"] = max_loaded_rtt_increase_ms
            entry["latency_constraint"] = "loaded RTT increase <= configured limit"
        per_variant.append(entry)
    paired_differences = defaultdict(lambda: defaultdict(list))
    unpaired_comparisons = 0
    for (pair, _run_id), queue_metrics in paired_qdisc_runs.items():
        if not {"pfifo", "fq_codel"}.issubset(queue_metrics):
            unpaired_comparisons += 1
            continue
        for metric in ("aggregate_throughput_mbps", "loaded_rtt_avg_ms",
                       "loaded_rtt_increase_ms", "jain_fairness"):
            paired_differences[pair][metric].append(
                queue_metrics["fq_codel"][metric] - queue_metrics["pfifo"][metric]
            )
    paired_summary = [
        {
            "queue_comparison": "fq_codel_minus_pfifo",
            "variant_pair": pair,
            "paired_repetitions": len(metrics["jain_fairness"]),
            **{name: mean_ci95(values) for name, values in metrics.items()},
        }
        for pair, metrics in sorted(paired_differences.items())
    ]
    return {
        "input": str(path),
        "incomplete_pairs_skipped": incomplete,
        "groups": summary,
        "per_variant": per_variant,
        "paired_queue_comparisons": paired_summary,
        "unpaired_queue_comparisons_skipped": unpaired_comparisons,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="dataset_c_fairness_mininet.csv")
    parser.add_argument("--output", help="optional JSON summary path")
    parser.add_argument(
        "--max-loaded-rtt-increase-ms", type=float,
        help="also report throughput samples meeting this loaded-RTT increase ceiling",
    )
    args = parser.parse_args()
    if args.max_loaded_rtt_increase_ms is not None and args.max_loaded_rtt_increase_ms < 0:
        parser.error("--max-loaded-rtt-increase-ms must be nonnegative")
    result = summarize(args.input, args.max_loaded_rtt_increase_ms)
    rendered = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
