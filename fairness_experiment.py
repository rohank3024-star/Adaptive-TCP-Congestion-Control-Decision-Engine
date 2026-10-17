#!/usr/bin/env python3
"""Controlled two-flow TCP fairness and loaded-latency Mininet experiment.

This is intentionally separate from the original single-flow dataset and ML
pipeline. It compares a FIFO bottleneck queue with FQ-CoDel while two TCP
senders share a shaped egress link. Ping runs concurrently with both transfers
to measure loaded RTT. Results are written only to the requested new CSV.

Example (requires Linux/Mininet, iperf3, and root):
    sudo python3 fairness_experiment.py --mode smoke --output data/fairness_smoke.csv
    sudo python3 fairness_experiment.py --mode pilot --repetitions 5
"""

import argparse
import csv
import json
import math
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from mininet_topology import verify_tcp_variants

VARIANTS = ("reno", "cubic", "bbr")
QUEUE_DISCIPLINES = ("pfifo", "fq_codel")
VARIANT_PAIRS = (("reno", "cubic"), ("reno", "bbr"), ("cubic", "bbr"))
DEFAULT_OUTPUT = "data/dataset_c_fairness_mininet.csv"
HEADERS = [
    "timestamp", "run_id", "bandwidth_mbps", "delay_ms", "loss_pct",
    "queue_discipline", "flow_id", "tcp_variant", "competing_variant",
    "throughput_mbps", "retransmissions", "baseline_rtt_avg_ms",
    "baseline_rtt_min_ms", "baseline_rtt_max_ms", "baseline_ping_loss_pct",
    "loaded_rtt_avg_ms", "loaded_rtt_increase_ms", "loaded_rtt_min_ms",
    "loaded_rtt_max_ms", "loaded_ping_loss_pct",
    "jain_fairness", "throughput_source",
]


def jains_fairness(values):
    """Return Jain's fairness index; reject empty/negative measurements."""
    values = [float(value) for value in values]
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("fairness requires finite, nonnegative flow rates")
    denominator = len(values) * sum(value * value for value in values)
    return (sum(values) ** 2 / denominator) if denominator else 0.0


def parse_ping_summary(output):
    """Parse Linux ping's packet-loss and RTT summary; return None on failure."""
    loss = re.search(r"([\d.]+)%\s+packet loss", output)
    rtt = re.search(
        r"(?:rtt|round-trip)\s+min/avg/max/(?:mdev|stddev)\s*=\s*"
        r"([\d.]+)/([\d.]+)/([\d.]+)", output,
    )
    if not loss or not rtt:
        return None
    return {
        "ping_loss_pct": float(loss.group(1)),
        "loaded_rtt_min_ms": float(rtt.group(1)),
        "loaded_rtt_avg_ms": float(rtt.group(2)),
        "loaded_rtt_max_ms": float(rtt.group(3)),
    }


def parse_iperf_result(output):
    """Extract actual receiver throughput and sender retransmissions from JSON."""
    try:
        data = json.loads(output)
        end = data["end"]
        received = end.get("sum_received", {})
        sent = end.get("sum_sent", {})
        rate = float(received["bits_per_second"])
        if not math.isfinite(rate) or rate <= 0:
            return None
        retransmits = sent.get("retransmits")
        if retransmits is None:
            streams = end.get("streams", [])
            retransmits = sum(
                stream.get("sender", {}).get("retransmits", 0)
                for stream in streams
            )
        return {
            "throughput_mbps": rate / 1_000_000,
            "retransmissions": int(retransmits),
            "throughput_source": "iperf_client_json_receiver",
        }
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def build_queue_commands(interface, bandwidth_mbps, queue_discipline):
    """Build explicit shared-bottleneck HTB + FIFO/FQ-CoDel commands."""
    if queue_discipline not in QUEUE_DISCIPLINES:
        raise ValueError(f"unsupported queue discipline: {queue_discipline}")
    if float(bandwidth_mbps) <= 0:
        raise ValueError("bandwidth must be positive")
    return [
        f"tc qdisc del dev {interface} root 2>/dev/null || true",
        f"tc qdisc add dev {interface} root handle 1: htb default 10",
        f"tc class add dev {interface} parent 1: classid 1:10 htb "
        f"rate {bandwidth_mbps}mbit ceil {bandwidth_mbps}mbit",
        (f"tc qdisc add dev {interface} parent 1:10 handle 10: "
         f"{queue_discipline} limit 1000"),
    ]


def _new_topology():
    """Create two senders and one receiver connected to a single OVS bridge."""
    try:
        from mininet.net import Mininet
        from mininet.topo import Topo
        from mininet.link import TCLink
        from mininet.node import OVSBridge
        from mininet.log import setLogLevel
    except ImportError as exc:
        raise RuntimeError("Mininet is required (install with: sudo apt install mininet)") from exc

    setLogLevel("warning")

    class SharedBottleneckTopo(Topo):
        def build(self):
            switch = self.addSwitch("s1")
            c1 = self.addHost("c1")
            c2 = self.addHost("c2")
            server = self.addHost("server")
            for host in (c1, c2, server):
                self.addLink(host, switch, cls=TCLink)

    network = Mininet(topo=SharedBottleneckTopo(), link=TCLink,
                      switch=OVSBridge, controller=None)
    network.start()
    c1, c2, server, switch = (network.get(name) for name in ("c1", "c2", "server", "s1"))
    if network.ping([c1, c2, server], timeout="3") > 50:
        network.stop()
        raise RuntimeError("new shared-bottleneck Mininet topology failed connectivity check")
    server_link = server.defaultIntf().link
    switch_intf = (server_link.intf1 if server_link.intf1.node == switch
                   else server_link.intf2)
    return network, c1, c2, server, switch, switch_intf.name


def _apply_condition(c1, c2, server, bandwidth_mbps, delay_ms, loss_pct,
                     queue_discipline, switch, bottleneck_interface):
    """Shape one shared egress; apply delay/loss at each endpoint egress."""
    for command in build_queue_commands(
            bottleneck_interface, bandwidth_mbps, queue_discipline):
        switch.cmd(command)
    qdisc_state = switch.cmd(f"tc qdisc show dev {bottleneck_interface}")
    if f"qdisc {queue_discipline}" not in qdisc_state or "qdisc htb" not in qdisc_state:
        raise RuntimeError(
            f"failed to install {queue_discipline} below HTB on "
            f"{bottleneck_interface}: {qdisc_state.strip()}"
        )

    for host in (c1, c2, server):
        interface = host.defaultIntf().name
        host.cmd(f"tc qdisc del dev {interface} root 2>/dev/null || true")
        command = f"tc qdisc add dev {interface} root handle 20: netem delay {delay_ms}ms"
        if float(loss_pct) > 0:
            command += f" loss {loss_pct}%"
        host.cmd(command)


def _start_iperf_servers(server, ports):
    for port in ports:
        server.cmd(f"iperf3 -s -p {port} -D --logfile /tmp/cn_fair_server_{port}.log")
    time.sleep(0.5)


def _stop_iperf_servers(server):
    server.cmd("killall iperf3 2>/dev/null || true")


def _start_client(host, receiver_ip, port, variant, duration):
    """Start iperf in the host namespace and return its process handle."""
    return host.popen(
        ["iperf3", "-c", receiver_ip, "-p", str(port), "-t", str(duration),
         "-C", variant, "-J"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _read_client(host, process, duration):
    """Wait for a completed client process and parse its JSON output."""
    try:
        output, _ = process.communicate(timeout=duration + 20)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        output, _ = process.communicate()
        raise RuntimeError(f"iperf3 timed out on {host.name}: {output[-300:]}") from exc
    parsed = parse_iperf_result(output)
    if parsed is None:
        raise RuntimeError(
            f"iperf3 failed on {host.name} (exit={process.returncode}): {output[-400:]}"
        )
    return parsed


def _run_ping_batch(hosts, receiver_ip, count):
    """Measure ping summaries concurrently from both client namespaces."""
    processes = [
        host.popen(
            ["ping", "-n", "-q", "-c", str(count), "-W", "2", receiver_ip],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for host in hosts
    ]
    results = []
    try:
        for host, process in zip(hosts, processes):
            try:
                output, _ = process.communicate(timeout=count * 2 + 5)
            except subprocess.TimeoutExpired as exc:
                process.kill()
                output, _ = process.communicate()
                raise RuntimeError(f"ping timed out on {host.name}: {output[-300:]}") from exc
            parsed = parse_ping_summary(output)
            if parsed is None:
                raise RuntimeError(f"could not parse ping from {host.name}: {output[-300:]}")
            results.append(parsed)
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate()
    return results


def run_experiment(mode="smoke", repetitions=1, duration=10,
                   bandwidth_mbps=20, delay_ms=50, loss_pct=0,
                   output=DEFAULT_OUTPUT, ping_count=5):
    """Run paired-flow tests; smoke uses Reno-vs-BBR, pilot covers all pairs."""
    if mode not in ("smoke", "pilot"):
        raise ValueError("mode must be smoke or pilot")
    if repetitions < 1 or duration < 3 or ping_count < 1:
        raise ValueError("repetitions/ping count must be positive and duration at least 3s")
    if bandwidth_mbps <= 0 or delay_ms < 0 or not 0 <= loss_pct < 100:
        raise ValueError("bandwidth must be positive; delay nonnegative; loss in [0, 100)")
    pairs = (VARIANT_PAIRS[1],) if mode == "smoke" else VARIANT_PAIRS
    queue_types = QUEUE_DISCIPLINES
    expected = len(pairs) * len(queue_types) * repetitions
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size:
        raise FileExistsError(f"refusing to overwrite existing results: {output_path}")

    if not verify_tcp_variants():
        raise RuntimeError("Reno, Cubic and BBR must be available in the kernel")
    network, c1, c2, server, switch, bottleneck_interface = _new_topology()
    completed = failed = 0
    ports = (5201, 5202)
    try:
        _start_iperf_servers(server, ports)
        with output_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=HEADERS)
            writer.writeheader()
            for run_id in range(1, repetitions + 1):
                queue_order = queue_types if run_id % 2 else tuple(reversed(queue_types))
                pair_order = pairs if run_id % 2 else tuple(reversed(pairs))
                for queue_type in queue_order:
                    _apply_condition(c1, c2, server, bandwidth_mbps, delay_ms,
                                     loss_pct, queue_type, switch, bottleneck_interface)
                    for base_pair in pair_order:
                        pair = base_pair if run_id % 2 else tuple(reversed(base_pair))
                        hosts = (c1, c2)
                        processes = []
                        client_results = []
                        try:
                            baseline_pings = _run_ping_batch(hosts, server.IP(), ping_count)
                            paths = [
                                _start_client(hosts[index], server.IP(), ports[index],
                                              pair[index], duration)
                                for index in range(2)
                            ]
                            processes = paths
                            time.sleep(0.5)
                            loaded_pings = _run_ping_batch(hosts, server.IP(), ping_count)
                            for host, process in zip(hosts, processes):
                                client_results.append(_read_client(host, process, duration))

                            fairness = jains_fairness(
                                [result["throughput_mbps"] for result in client_results]
                            )
                            timestamp = datetime.now().isoformat()
                            for index, result in enumerate(client_results):
                                row = {
                                    "timestamp": timestamp,
                                    "run_id": run_id,
                                    "bandwidth_mbps": bandwidth_mbps,
                                    "delay_ms": delay_ms,
                                    "loss_pct": loss_pct,
                                    "queue_discipline": queue_type,
                                    "flow_id": index + 1,
                                    "tcp_variant": pair[index],
                                    "competing_variant": pair[1 - index],
                                    **result,
                                    "baseline_rtt_avg_ms": baseline_pings[index]["loaded_rtt_avg_ms"],
                                    "baseline_rtt_min_ms": baseline_pings[index]["loaded_rtt_min_ms"],
                                    "baseline_rtt_max_ms": baseline_pings[index]["loaded_rtt_max_ms"],
                                    "baseline_ping_loss_pct": baseline_pings[index]["ping_loss_pct"],
                                    "loaded_rtt_avg_ms": loaded_pings[index]["loaded_rtt_avg_ms"],
                                    "loaded_rtt_min_ms": loaded_pings[index]["loaded_rtt_min_ms"],
                                    "loaded_rtt_max_ms": loaded_pings[index]["loaded_rtt_max_ms"],
                                    "loaded_ping_loss_pct": loaded_pings[index]["ping_loss_pct"],
                                    "loaded_rtt_increase_ms": (
                                        loaded_pings[index]["loaded_rtt_avg_ms"] -
                                        baseline_pings[index]["loaded_rtt_avg_ms"]
                                    ),
                                    "jain_fairness": fairness,
                                }
                                writer.writerow(row)
                            handle.flush()
                            completed += 1
                            print(
                                f"[OK] qdisc={queue_type}, pair={pair[0]}/{pair[1]}, "
                                f"run={run_id}: {client_results[0]['throughput_mbps']:.2f}/"
                                f"{client_results[1]['throughput_mbps']:.2f} Mbps, "
                                f"loaded RTT={loaded_pings[0]['loaded_rtt_avg_ms']:.2f}/"
                                f"{loaded_pings[1]['loaded_rtt_avg_ms']:.2f} ms, "
                                f"Jain={fairness:.3f}"
                            )
                        except Exception as exc:
                            failed += 1
                            print(f"[FAIL] qdisc={queue_type}, pair={pair}, run={run_id}: {exc}")
                            # Do not represent failed measurements as zero-valued rows.
                            for process in processes:
                                if process.poll() is None:
                                    process.kill()
                                    process.communicate()
        print(f"[DONE] Paired tests: {completed}/{expected}; failed={failed}; CSV={output_path}")
        return completed, failed
    finally:
        _stop_iperf_servers(server)
        network.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "pilot"), default="smoke")
    parser.add_argument("--repetitions", type=int, default=1,
                        help="repeat each queue/pair condition (pilot recommendation: 5)")
    parser.add_argument("--duration", type=int, default=10, help="seconds per TCP flow")
    parser.add_argument("--bandwidth", type=float, default=20, help="shared bottleneck Mbps")
    parser.add_argument("--delay", type=float, default=50, help="one-way endpoint delay in ms")
    parser.add_argument("--loss", type=float, default=0, help="endpoint egress loss percentage")
    parser.add_argument("--ping-count", type=int, default=5)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        _, failed = run_experiment(
            mode=args.mode, repetitions=args.repetitions, duration=args.duration,
            bandwidth_mbps=args.bandwidth, delay_ms=args.delay, loss_pct=args.loss,
            output=args.output, ping_count=args.ping_count,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
