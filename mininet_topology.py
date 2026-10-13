#!/usr/bin/env python3
"""
Module M1: Mininet Topology Setup
==================================
Creates the H1 – S1 – H2 linear topology used for all TCP experiments.
Verifies that TCP Reno, Cubic, and BBR are available in the Linux kernel.

Usage (requires root on Linux / WSL2):
    sudo python3 mininet_topology.py

Author: Rohan Kumar (24BDS0463) – Networking & Experimentation
"""

import subprocess
import sys
import os

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REQUIRED_TCP_VARIANTS = ["reno", "cubic", "bbr"]


# ---------------------------------------------------------------------------
# TCP variant availability check
# ---------------------------------------------------------------------------
def get_available_tcp_variants():
    """Query the Linux kernel for available TCP congestion-control algorithms."""
    try:
        result = subprocess.run(
            ["sysctl", "net.ipv4.tcp_available_congestion_control"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            print(f"[ERROR] sysctl failed: {result.stderr.strip()}")
            return []
        # Output format: net.ipv4.tcp_available_congestion_control = reno cubic bbr
        raw = result.stdout.strip().split("=")[-1].strip()
        return raw.split()
    except FileNotFoundError:
        print("[ERROR] sysctl not found. Are you running on Linux / WSL2?")
        return []
    except subprocess.TimeoutExpired:
        print("[ERROR] sysctl timed out.")
        return []


def verify_tcp_variants():
    """Ensure Reno, Cubic, and BBR are all available. Try modprobe for BBR."""
    available = get_available_tcp_variants()
    print(f"[INFO] Available TCP variants: {available}")

    missing = [v for v in REQUIRED_TCP_VARIANTS if v not in available]

    if "bbr" in missing:
        print("[INFO] BBR not found – attempting 'sudo modprobe tcp_bbr' ...")
        subprocess.run(["sudo", "modprobe", "tcp_bbr"], capture_output=True)
        available = get_available_tcp_variants()
        missing = [v for v in REQUIRED_TCP_VARIANTS if v not in available]

    if missing:
        print(f"[FATAL] Missing TCP variants: {missing}")
        print("        Use a native Ubuntu VM or the official Mininet VM image.")
        return False

    print("[OK] All required TCP variants are available.")
    return True


# ---------------------------------------------------------------------------
# Mininet topology creation
# ---------------------------------------------------------------------------
def create_topology(bandwidth="100mbit", delay="5ms", loss=0):
    """
    Create the H1 – S1 – H2 Mininet topology.
    Returns (net, h1, h2) or None on failure.
    
    Parameters
    ----------
    bandwidth : str   – Link bandwidth (e.g., '10mbit')
    delay     : str   – One-way link delay (e.g., '50ms')
    loss      : float – Packet-loss percentage (e.g., 2 for 2%)
    """
    try:
        from mininet.net import Mininet
        from mininet.topo import Topo
        from mininet.link import TCLink
        from mininet.log import setLogLevel
    except ImportError:
        print("[ERROR] Mininet is not installed. Run: sudo apt install mininet")
        return None

    setLogLevel("warning")

    class LinearTopo(Topo):
        """H1 -- S1 -- H2 linear topology."""
        def build(self):
            switch = self.addSwitch("s1")
            h1 = self.addHost("h1")
            h2 = self.addHost("h2")
            # Link parameters set here as defaults; they will be
            # overridden per-experiment by network_config.py
            self.addLink(h1, switch, cls=TCLink,
                         bw=int(bandwidth.replace("mbit", "")),
                         delay=delay, loss=loss)
            self.addLink(h2, switch, cls=TCLink,
                         bw=int(bandwidth.replace("mbit", "")),
                         delay=delay, loss=loss)

    topo = LinearTopo()
    net = Mininet(topo=topo, link=TCLink)
    net.start()

    h1 = net.get("h1")
    h2 = net.get("h2")

    # Quick connectivity check
    print(f"[INFO] h1 IP: {h1.IP()}, h2 IP: {h2.IP()}")
    ping_result = net.ping([h1, h2], timeout="3")
    if ping_result > 50:  # more than 50% loss
        print("[WARNING] High packet loss in ping test – check topology.")
    else:
        print("[OK] Topology is up and hosts can communicate.")

    return net, h1, h2


def stop_topology(net):
    """Cleanly shut down the Mininet topology."""
    if net:
        net.stop()
        print("[INFO] Topology stopped.")


# ---------------------------------------------------------------------------
# Main – standalone verification
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  Module M1: Mininet Topology & TCP Variant Verification")
    print("=" * 60)

    if not verify_tcp_variants():
        sys.exit(1)

    result = create_topology()
    if result is None:
        print("[SKIP] Mininet not available – topology test skipped.")
        sys.exit(0)

    net, h1, h2 = result
    print(f"\n[TEST] Ping from h1 to h2:")
    h1.cmd(f"ping -c 3 {h2.IP()}")
    stop_topology(net)
    print("\n[DONE] M1 verification complete.")
