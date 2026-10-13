#!/usr/bin/env python3
"""
Module M2: Network Condition Control
=====================================
Applies bandwidth, delay, and packet-loss settings to Mininet links
using Linux tc (traffic control) with netem.

Provides functions to:
  - Apply a specific (bandwidth, delay, loss) configuration
  - Clear / reset link conditions
  - Measure baseline RTT before TCP benchmarking

Usage:
    Imported by benchmark_runner.py and run_experiment.py.
    Can be run standalone for quick testing:
        sudo python3 network_config.py

Author: Rohan Kumar (24BDS0463) – Networking & Experimentation
"""

import subprocess
import re
import time


# ---------------------------------------------------------------------------
# Experiment parameter grid
# ---------------------------------------------------------------------------
BANDWIDTHS_MBPS = [1, 5, 10, 20, 50, 100]
DELAYS_MS       = [5, 20, 50, 100, 200]
LOSSES_PCT      = [0, 1, 2, 3, 5]

# Pilot subset (54-run validation)
PILOT_BANDWIDTHS = [5, 20, 100]
PILOT_DELAYS     = [5, 50, 200]
PILOT_LOSSES     = [0, 2]


# ---------------------------------------------------------------------------
# tc / netem helpers
# ---------------------------------------------------------------------------
def apply_netem(host, interface, bandwidth_mbps, delay_ms, loss_pct):
    """
    Apply tc/netem settings on a Mininet host interface.
    
    Parameters
    ----------
    host          : Mininet host object (has .cmd() method)
    interface     : str – interface name (e.g., 'h1-eth0')
    bandwidth_mbps: int – link bandwidth in Mbps
    delay_ms      : int – one-way delay in ms
    loss_pct      : float – packet loss percentage
    """
    # Clear any existing qdisc first
    host.cmd(f"tc qdisc del dev {interface} root 2>/dev/null")
    
    # Add HTB root qdisc for bandwidth control
    host.cmd(f"tc qdisc add dev {interface} root handle 1: htb default 1")
    host.cmd(
        f"tc class add dev {interface} parent 1: classid 1:1 htb "
        f"rate {bandwidth_mbps}mbit ceil {bandwidth_mbps}mbit"
    )
    
    # Add netem qdisc for delay and loss
    netem_cmd = f"tc qdisc add dev {interface} parent 1:1 handle 10: netem delay {delay_ms}ms"
    if loss_pct > 0:
        netem_cmd += f" loss {loss_pct}%"
    
    host.cmd(netem_cmd)
    
    return True


def clear_netem(host, interface):
    """Remove all tc/netem settings from a host interface."""
    host.cmd(f"tc qdisc del dev {interface} root 2>/dev/null")


def apply_condition(h1, h2, bandwidth_mbps, delay_ms, loss_pct):
    """
    Apply symmetric network conditions to both host interfaces.
    Delay is split across both directions so total RTT = 2 * delay_ms.
    
    Parameters
    ----------
    h1, h2         : Mininet host objects
    bandwidth_mbps : int
    delay_ms       : int – one-way delay (total RTT will be ~2x this)
    loss_pct       : float
    """
    # Get interface names (Mininet convention: hostName-eth0)
    h1_intf = f"{h1.name}-eth0"
    h2_intf = f"{h2.name}-eth0"
    
    apply_netem(h1, h1_intf, bandwidth_mbps, delay_ms, loss_pct)
    apply_netem(h2, h2_intf, bandwidth_mbps, delay_ms, loss_pct)
    
    print(f"[CONFIG] Applied: BW={bandwidth_mbps}Mbps, "
          f"Delay={delay_ms}ms (one-way), Loss={loss_pct}%")
    return True


def clear_condition(h1, h2):
    """Clear tc/netem settings on both hosts."""
    h1_intf = f"{h1.name}-eth0"
    h2_intf = f"{h2.name}-eth0"
    clear_netem(h1, h1_intf)
    clear_netem(h2, h2_intf)


# ---------------------------------------------------------------------------
# Baseline RTT measurement
# ---------------------------------------------------------------------------
def measure_baseline_rtt(h1, h2_ip, count=5):
    """
    Measure baseline RTT using ping before TCP benchmarking.
    
    Parameters
    ----------
    h1     : Mininet host object (sender)
    h2_ip  : str – IP address of h2
    count  : int – number of ping packets
    
    Returns
    -------
    float – average RTT in ms, or -1.0 on failure
    """
    output = h1.cmd(f"ping -c {count} -q {h2_ip}")
    
    # Parse: rtt min/avg/max/mdev = 10.123/10.456/10.789/0.123 ms
    match = re.search(r"rtt min/avg/max/mdev = [\d.]+/([\d.]+)/", output)
    if match:
        avg_rtt = float(match.group(1))
        print(f"[RTT] Baseline RTT to {h2_ip}: {avg_rtt:.2f} ms")
        return avg_rtt
    
    print(f"[WARNING] Could not parse RTT from ping output:\n{output}")
    return -1.0


# ---------------------------------------------------------------------------
# Parameter grid generators
# ---------------------------------------------------------------------------
def generate_full_grid():
    """Generate all 150 network condition combinations."""
    conditions = []
    for bw in BANDWIDTHS_MBPS:
        for delay in DELAYS_MS:
            for loss in LOSSES_PCT:
                conditions.append({
                    "bandwidth_mbps": bw,
                    "delay_ms": delay,
                    "loss_pct": loss
                })
    print(f"[GRID] Generated {len(conditions)} network conditions (full grid)")
    return conditions


def generate_pilot_grid():
    """Generate the 18 pilot network conditions (54 runs with 3 variants)."""
    conditions = []
    for bw in PILOT_BANDWIDTHS:
        for delay in PILOT_DELAYS:
            for loss in PILOT_LOSSES:
                conditions.append({
                    "bandwidth_mbps": bw,
                    "delay_ms": delay,
                    "loss_pct": loss
                })
    print(f"[GRID] Generated {len(conditions)} network conditions (pilot grid)")
    return conditions


# ---------------------------------------------------------------------------
# Main – standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  Module M2: Network Condition Control")
    print("=" * 60)
    
    # Show parameter grids
    pilot = generate_pilot_grid()
    full = generate_full_grid()
    
    print(f"\nPilot conditions: {len(pilot)}")
    print(f"Full conditions:  {len(full)}")
    print(f"Total pilot runs (x3 variants): {len(pilot) * 3}")
    print(f"Total full runs  (x3 variants): {len(full) * 3}")
    print(f"Total full runs  (x3 reps):     {len(full) * 3 * 3}")
    
    print("\nSample pilot conditions:")
    for c in pilot[:5]:
        print(f"  BW={c['bandwidth_mbps']}Mbps, "
              f"Delay={c['delay_ms']}ms, Loss={c['loss_pct']}%")
