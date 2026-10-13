#!/usr/bin/env python3
"""
Synthetic Data Generator
=========================
Generates realistic TCP experiment data based on known TCP Reno, Cubic,
and BBR behavior under varying network conditions. This allows the ML
pipeline and visualizations to be demonstrated on Windows without Mininet.

The synthetic data models real-world TCP behavior:
  - BBR excels in high-delay, moderate-loss environments
  - Cubic performs well in high-bandwidth, low-loss conditions
  - Reno is competitive in low-bandwidth, low-delay scenarios
  - All variants degrade under high packet loss
  - Noise is added to simulate measurement variability

Author: Shreyans Bohra (24BCT0013) – Data & ML
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

# Seed for reproducibility
np.random.seed(42)

# ---------------------------------------------------------------------------
# Parameter grid
# ---------------------------------------------------------------------------
BANDWIDTHS_MBPS = [1, 5, 10, 20, 50, 100]
DELAYS_MS       = [5, 20, 50, 100, 200]
LOSSES_PCT      = [0, 1, 2, 3, 5]
TCP_VARIANTS    = ["reno", "cubic", "bbr"]
REPETITIONS     = 3

# Pilot grid
PILOT_BANDWIDTHS = [5, 20, 100]
PILOT_DELAYS     = [5, 50, 200]
PILOT_LOSSES     = [0, 2]


# ---------------------------------------------------------------------------
# TCP throughput models (based on published TCP behavior)
# ---------------------------------------------------------------------------
def compute_throughput_reno(bw, delay, loss):
    """
    TCP Reno throughput model.
    Reno uses AIMD (Additive Increase, Multiplicative Decrease).
    Performance degrades significantly with high delay and loss.
    Mathis formula approximation: throughput ∝ MSS / (RTT * sqrt(loss))
    """
    rtt = 2 * delay  # Round trip in ms
    mss = 1460       # bytes

    if loss == 0:
        # No loss: limited by bandwidth, but Reno is conservative
        base = bw * 0.82  # Reno typically gets ~82% of link capacity
        # Reno struggles more with high BDP (bandwidth-delay product)
        bdp_factor = max(0.4, 1.0 - (bw * rtt / 100000) * 0.3)
        throughput = base * bdp_factor
    else:
        # Mathis formula adapted
        loss_rate = loss / 100.0
        # throughput in Mbps
        mathis = (mss * 8 / 1_000_000) / (rtt / 1000.0 * np.sqrt(loss_rate)) * 0.75
        throughput = min(bw * 0.85, mathis)

    # Add realistic noise (±5-10%)
    noise = np.random.normal(1.0, 0.06)
    throughput = max(0.05, throughput * noise)
    return round(throughput, 2)


def compute_throughput_cubic(bw, delay, loss):
    """
    TCP Cubic throughput model.
    Cubic uses a cubic function for window growth, designed for
    high-bandwidth, long-delay networks. Better than Reno for high BDP.
    """
    rtt = 2 * delay

    if loss == 0:
        # Cubic gets closer to link capacity, especially at high BW
        base = bw * 0.90  # ~90% utilization
        # Cubic handles high BDP better than Reno
        bdp_factor = max(0.55, 1.0 - (bw * rtt / 150000) * 0.15)
        throughput = base * bdp_factor
    else:
        loss_rate = loss / 100.0
        mss = 1460
        # Cubic's cubic function allows faster recovery
        cubic_factor = (mss * 8 / 1_000_000) / (rtt / 1000.0 * (loss_rate ** 0.4)) * 1.1
        throughput = min(bw * 0.88, cubic_factor)

    noise = np.random.normal(1.0, 0.05)
    throughput = max(0.05, throughput * noise)
    return round(throughput, 2)


def compute_throughput_bbr(bw, delay, loss):
    """
    TCP BBR throughput model.
    BBR estimates bottleneck bandwidth and min RTT, operates differently
    from loss-based algorithms. Resilient to moderate packet loss.
    """
    rtt = 2 * delay

    if loss == 0:
        # BBR achieves near-full link utilization
        base = bw * 0.94
        # BBR handles high delay better than Reno/Cubic
        delay_factor = max(0.70, 1.0 - (delay / 1000) * 0.2)
        throughput = base * delay_factor
    elif loss <= 2:
        # BBR is resilient to moderate loss (doesn't interpret loss as congestion)
        loss_penalty = 1.0 - loss * 0.05  # Only 5% penalty per % loss
        base = bw * 0.90 * loss_penalty
        delay_factor = max(0.65, 1.0 - (delay / 800) * 0.15)
        throughput = base * delay_factor
    else:
        # BBR degrades under heavy loss but less than Reno/Cubic
        loss_rate = loss / 100.0
        base = bw * 0.85 * (1.0 - loss_rate * 0.8)
        delay_factor = max(0.55, 1.0 - (delay / 600) * 0.2)
        throughput = base * delay_factor

    noise = np.random.normal(1.0, 0.04)
    throughput = max(0.05, throughput * noise)
    return round(throughput, 2)


def compute_retransmissions(bw, delay, loss, variant):
    """Model retransmissions based on loss rate and TCP variant behavior."""
    if loss == 0:
        base_retrans = np.random.poisson(2)  # Minimal retransmissions
    else:
        # Retransmissions scale with loss and data volume
        data_volume_factor = bw * 10  # 10s test
        expected = data_volume_factor * (loss / 100) * 0.3
        
        # BBR has fewer retransmissions due to pacing
        if variant == "bbr":
            expected *= 0.6
        elif variant == "reno":
            expected *= 1.3  # Reno's bursty behavior
        
        base_retrans = max(0, int(np.random.poisson(max(1, expected))))
    
    return base_retrans


def compute_baseline_rtt(delay):
    """Model baseline RTT measurement (2x one-way delay + jitter)."""
    rtt = 2 * delay
    jitter = np.random.normal(0, delay * 0.05)  # ±5% jitter
    return round(max(0.5, rtt + jitter), 2)


# ---------------------------------------------------------------------------
# Generate datasets
# ---------------------------------------------------------------------------
def generate_raw_dataset(mode="full"):
    """
    Generate Dataset A (raw benchmark data).
    
    Parameters
    ----------
    mode : str – 'pilot' or 'full'
    
    Returns
    -------
    pd.DataFrame – raw benchmark results
    """
    if mode == "pilot":
        bandwidths = PILOT_BANDWIDTHS
        delays = PILOT_DELAYS
        losses = PILOT_LOSSES
    else:
        bandwidths = BANDWIDTHS_MBPS
        delays = DELAYS_MS
        losses = LOSSES_PCT

    rows = []
    base_time = datetime(2026, 10, 10, 14, 0, 0)
    run_idx = 0

    for bw in bandwidths:
        for delay in delays:
            for loss in losses:
                baseline_rtt = compute_baseline_rtt(delay)
                
                for rep in range(1, REPETITIONS + 1):
                    for variant in TCP_VARIANTS:
                        run_idx += 1
                        timestamp = base_time + timedelta(seconds=run_idx * 15)

                        # Compute throughput based on TCP model
                        if variant == "reno":
                            tp = compute_throughput_reno(bw, delay, loss)
                        elif variant == "cubic":
                            tp = compute_throughput_cubic(bw, delay, loss)
                        else:
                            tp = compute_throughput_bbr(bw, delay, loss)

                        retrans = compute_retransmissions(bw, delay, loss, variant)
                        # Per-variant RTT is slightly different from baseline
                        rtt_offset = np.random.normal(0, baseline_rtt * 0.03)
                        rtt = round(max(0.5, baseline_rtt + rtt_offset), 2)

                        rows.append({
                            "timestamp": timestamp.isoformat(),
                            "bandwidth_mbps": bw,
                            "delay_ms": delay,
                            "loss_pct": loss,
                            "tcp_variant": variant,
                            "throughput_mbps": tp,
                            "retransmissions": retrans,
                            "rtt_ms": rtt,
                            "run_id": rep
                        })

    df = pd.DataFrame(rows)
    print(f"[SYNTH] Generated {len(df)} raw experiment rows ({mode} mode)")
    return df


def generate_and_save(mode="full", output_dir="data"):
    """Generate synthetic data and save to CSV."""
    os.makedirs(output_dir, exist_ok=True)

    # Generate raw dataset
    df_raw = generate_raw_dataset(mode)
    raw_path = os.path.join(output_dir, "dataset_a_raw.csv")
    df_raw.to_csv(raw_path, index=False)
    print(f"[SAVED] Raw dataset → {raw_path} ({len(df_raw)} rows)")

    return df_raw


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  Synthetic Data Generator for TCP Experiments")
    print("=" * 60)
    
    df = generate_and_save(mode="full")
    
    print(f"\nDataset shape: {df.shape}")
    print(f"\nSample rows:")
    print(df.head(10).to_string(index=False))
    
    print(f"\n--- Statistics ---")
    print(f"Unique conditions: {df.groupby(['bandwidth_mbps', 'delay_ms', 'loss_pct']).ngroups}")
    print(f"Variants: {df['tcp_variant'].unique()}")
    print(f"Repetitions per condition-variant: {df.groupby(['bandwidth_mbps', 'delay_ms', 'loss_pct', 'tcp_variant']).size().median():.0f}")
    
    print(f"\nThroughput by variant:")
    print(df.groupby("tcp_variant")["throughput_mbps"].describe().round(2).to_string())
