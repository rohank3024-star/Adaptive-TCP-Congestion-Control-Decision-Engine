"""Unit tests for the isolated paired-flow experiment and its analyzer."""

import json
import tempfile
import unittest
from pathlib import Path

from analyze_fairness import summarize
from dashboard import load_fairness_summaries
from fairness_experiment import (
    build_queue_commands,
    jains_fairness,
    parse_iperf_result,
    parse_ping_summary,
)


class FairnessExperimentTests(unittest.TestCase):
    def test_jain_index_equal_and_unequal(self):
        self.assertAlmostEqual(jains_fairness([10, 10]), 1.0)
        self.assertAlmostEqual(jains_fairness([10, 0]), 0.5)

    def test_jain_rejects_invalid_rates(self):
        with self.assertRaises(ValueError):
            jains_fairness([])
        with self.assertRaises(ValueError):
            jains_fairness([1, -1])

    def test_ping_parser_reports_loaded_rtt_and_loss(self):
        output = ("5 packets transmitted, 4 received, 20% packet loss, time 4000ms\n"
                  "rtt min/avg/max/mdev = 20.1/25.2/31.3/4.0 ms")
        self.assertEqual(parse_ping_summary(output), {
            "ping_loss_pct": 20.0,
            "loaded_rtt_min_ms": 20.1,
            "loaded_rtt_avg_ms": 25.2,
            "loaded_rtt_max_ms": 31.3,
        })
        self.assertIsNone(parse_ping_summary("ping failed"))

    def test_iperf_parser_uses_receiver_metrics(self):
        payload = {"end": {"sum_sent": {"retransmits": 7},
                           "sum_received": {"bits_per_second": 4_200_000}}}
        result = parse_iperf_result(json.dumps(payload))
        self.assertEqual(result["throughput_mbps"], 4.2)
        self.assertEqual(result["retransmissions"], 7)
        self.assertEqual(result["throughput_source"], "iperf_client_json_receiver")
        self.assertIsNone(parse_iperf_result('{"end": {}}'))

    def test_queue_commands_shape_shared_egress_and_select_queue(self):
        fifo = build_queue_commands("s1-eth3", 20, "pfifo")
        fq = build_queue_commands("s1-eth3", 20, "fq_codel")
        self.assertIn("rate 20mbit ceil 20mbit", fifo[2])
        self.assertIn("pfifo limit 1000", fifo[3])
        self.assertIn("fq_codel limit 1000", fq[3])
        with self.assertRaises(ValueError):
            build_queue_commands("s1-eth3", 20, "cake")

    def test_analyzer_uses_complete_pairs_and_confidence_intervals(self):
        headers = [
            "timestamp", "run_id", "queue_discipline", "flow_id",
            "tcp_variant", "throughput_mbps", "loaded_rtt_avg_ms",
            "loaded_rtt_increase_ms",
            "jain_fairness",
        ]
        rows = [
            ["t1", 1, "pfifo", 1, "reno", 4, 50, 30, 0.8],
            ["t1", 1, "pfifo", 2, "bbr", 8, 54, 34, 0.8],
            ["t2", 2, "pfifo", 1, "reno", 6, 55, 35, 0.9],
            ["t2", 2, "pfifo", 2, "bbr", 7, 57, 37, 0.9],
            ["t3", 3, "fq_codel", 1, "reno", 5, 35, 15, 0.95],
        ]
        import csv
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paired.csv"
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(headers)
                writer.writerows(rows)
            result = summarize(path, max_loaded_rtt_increase_ms=33)
        self.assertEqual(result["incomplete_pairs_skipped"], 1)
        group = result["groups"][0]
        self.assertEqual(group["complete_paired_runs"], 2)
        self.assertEqual(group["aggregate_throughput_mbps"]["mean"], 12.5)
        self.assertLess(group["aggregate_throughput_mbps"]["ci95_low"], 12.5)
        self.assertGreater(group["aggregate_throughput_mbps"]["ci95_high"], 12.5)
        self.assertEqual(group["loaded_rtt_avg_ms"]["mean"], 54.0)
        self.assertEqual(group["loaded_rtt_increase_ms"]["mean"], 34.0)
        variants = {(row["queue_discipline"], row["tcp_variant"]): row
                    for row in result["per_variant"]}
        self.assertEqual(
            variants[("pfifo", "reno")]["throughput_within_rtt_limit_mbps"]["n"], 1
        )
        self.assertEqual(
            variants[("pfifo", "bbr")]["throughput_within_rtt_limit_mbps"]["n"], 0
        )

    def test_analyzer_computes_paired_queue_differences(self):
        import csv
        headers = [
            "timestamp", "run_id", "queue_discipline", "flow_id",
            "tcp_variant", "throughput_mbps", "loaded_rtt_avg_ms",
            "loaded_rtt_increase_ms", "jain_fairness",
        ]
        rows = []
        for timestamp, qdisc, rates, rtt, increase, fairness in (
            ("p1", "pfifo", (4, 8), 1000, 900, 0.6),
            ("f1", "fq_codel", (5, 7), 100, 0, 0.95),
            ("p2", "pfifo", (3, 9), 1100, 1000, 0.7),
            ("f2", "fq_codel", (6, 6), 102, 2, 1.0),
        ):
            rows.extend([
                [timestamp, timestamp[-1], qdisc, 1, "reno", rates[0], rtt,
                 increase, fairness],
                [timestamp, timestamp[-1], qdisc, 2, "bbr", rates[1], rtt,
                 increase, fairness],
            ])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paired.csv"
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(headers)
                writer.writerows(rows)
            result = summarize(path)
        self.assertEqual(result["unpaired_queue_comparisons_skipped"], 0)
        comparison = result["paired_queue_comparisons"][0]
        self.assertEqual(comparison["paired_repetitions"], 2)
        self.assertEqual(comparison["aggregate_throughput_mbps"]["mean"], 0.0)
        self.assertEqual(comparison["loaded_rtt_increase_ms"]["mean"], -949.0)

    def test_dashboard_loads_condition_and_paired_comparison(self):
        import csv
        from fairness_experiment import HEADERS
        rows = []
        for timestamp, qdisc, rates, loaded, increase, fairness in (
            ("p1", "pfifo", (4, 8), 1000, 900, 0.6),
            ("f1", "fq_codel", (5, 7), 100, 0, 0.95),
        ):
            for index, variant in enumerate(("reno", "bbr")):
                rows.append({
                    "timestamp": timestamp,
                    "run_id": 1,
                    "bandwidth_mbps": 20,
                    "delay_ms": 50,
                    "loss_pct": 0,
                    "queue_discipline": qdisc,
                    "flow_id": index + 1,
                    "tcp_variant": variant,
                    "competing_variant": ("bbr", "reno")[index],
                    "throughput_mbps": rates[index],
                    "retransmissions": 0,
                    "baseline_rtt_avg_ms": 100,
                    "baseline_rtt_min_ms": 99,
                    "baseline_rtt_max_ms": 101,
                    "baseline_ping_loss_pct": 0,
                    "loaded_rtt_avg_ms": loaded,
                    "loaded_rtt_increase_ms": increase,
                    "loaded_rtt_min_ms": loaded - 1,
                    "loaded_rtt_max_ms": loaded + 1,
                    "loaded_ping_loss_pct": 0,
                    "jain_fairness": fairness,
                    "throughput_source": "iperf_client_json_receiver",
                })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "condition.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=HEADERS)
                writer.writeheader()
                writer.writerows(rows)
            result = load_fairness_summaries([path])
        self.assertEqual(result[0]["condition"], "20 Mbps / 50 ms / 0% loss")
        self.assertEqual(len(result[0]["paired_queue_comparisons"]), 1)


if __name__ == "__main__":
    unittest.main()
