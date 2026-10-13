#!/usr/bin/env python3
"""
Module M6: TCP Variant Predictor
==================================
Command-line interface that accepts network condition features and
returns the recommended TCP congestion control algorithm.

Usage:
    python predictor.py --bandwidth 20 --delay 50 --loss 2
    python predictor.py --interactive
    python predictor.py --batch conditions.csv

Author: Shreyans Bohra (24BCT0013) – Data & ML
"""

import argparse
import os
import sys
import json
import joblib
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_PATH   = "models/random_forest.joblib"
FEATURE_COLS = ["bandwidth_mbps", "delay_ms", "loss_pct", "baseline_rtt"]
TCP_VARIANTS = ["bbr", "cubic", "reno"]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(model_path=MODEL_PATH):
    """Load the trained Random Forest model."""
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found at {model_path}")
        print("        Run ml_pipeline.py first to train the model.")
        return None
    
    model = joblib.load(model_path)
    print(f"[OK] Model loaded from {model_path}")
    return model


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict_tcp_variant(model, bandwidth_mbps, delay_ms, loss_pct, baseline_rtt=None):
    """
    Predict the best TCP variant for given network conditions.
    
    Parameters
    ----------
    model          : trained sklearn classifier
    bandwidth_mbps : float
    delay_ms       : float
    loss_pct       : float
    baseline_rtt   : float (optional, estimated from delay if not provided)
    
    Returns
    -------
    dict with prediction and probabilities
    """
    # Estimate baseline RTT if not provided (2x one-way delay + typical overhead)
    if baseline_rtt is None:
        baseline_rtt = 2 * delay_ms * 1.02  # ~2% overhead
    
    features = np.array([[bandwidth_mbps, delay_ms, loss_pct, baseline_rtt]])
    
    prediction = model.predict(features)[0]
    
    # Get prediction probabilities
    probabilities = {}
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        classes = model.classes_
        probabilities = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}
    
    result = {
        "input": {
            "bandwidth_mbps": bandwidth_mbps,
            "delay_ms": delay_ms,
            "loss_pct": loss_pct,
            "baseline_rtt": round(baseline_rtt, 2)
        },
        "recommended_tcp": prediction,
        "confidence": probabilities
    }
    
    return result


def display_prediction(result):
    """Pretty-print a prediction result."""
    inp = result["input"]
    rec = result["recommended_tcp"]
    conf = result["confidence"]
    
    print(f"\n{'─'*50}")
    print(f"  Network Conditions:")
    print(f"    Bandwidth: {inp['bandwidth_mbps']} Mbps")
    print(f"    Delay:     {inp['delay_ms']} ms")
    print(f"    Loss:      {inp['loss_pct']}%")
    print(f"    RTT:       {inp['baseline_rtt']} ms")
    print(f"\n  ★ Recommended TCP: {rec.upper()}")
    
    if conf:
        print(f"\n  Confidence Scores:")
        for variant in sorted(conf.keys()):
            bar_len = int(conf[variant] * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            marker = " ◄" if variant == rec else ""
            print(f"    {variant:6s}: [{bar}] {conf[variant]:.1%}{marker}")
    
    print(f"{'─'*50}")


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------
def interactive_mode(model):
    """Run an interactive prediction session."""
    print("\n" + "=" * 50)
    print("  TCP Variant Predictor – Interactive Mode")
    print("  Type 'quit' to exit")
    print("=" * 50)
    
    while True:
        print()
        try:
            bw = input("  Bandwidth (Mbps) [1-100]: ").strip()
            if bw.lower() == "quit":
                break
            bw = float(bw)
            
            delay = float(input("  Delay (ms)       [5-200]: ").strip())
            loss  = float(input("  Loss (%)         [0-5]:   ").strip())
            
            rtt_input = input("  Baseline RTT (ms) [Enter to estimate]: ").strip()
            rtt = float(rtt_input) if rtt_input else None
            
            result = predict_tcp_variant(model, bw, delay, loss, rtt)
            display_prediction(result)
            
        except ValueError as e:
            print(f"  [ERROR] Invalid input: {e}")
        except KeyboardInterrupt:
            print("\n  Exiting...")
            break
    
    print("\n[DONE] Interactive session ended.")


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------
def batch_predict(model, csv_path):
    """Run predictions on a CSV file of network conditions."""
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    print(f"[BATCH] Processing {len(df)} conditions from {csv_path}")
    
    results = []
    for _, row in df.iterrows():
        bw = row.get("bandwidth_mbps", row.get("bandwidth", 10))
        delay = row.get("delay_ms", row.get("delay", 50))
        loss = row.get("loss_pct", row.get("loss", 0))
        rtt = row.get("baseline_rtt", None)
        
        result = predict_tcp_variant(model, bw, delay, loss, rtt)
        results.append(result)
        display_prediction(result)
    
    return results


# ---------------------------------------------------------------------------
# Quick demo predictions
# ---------------------------------------------------------------------------
def run_demo(model):
    """Run predictions on representative network scenarios."""
    print("\n" + "=" * 60)
    print("  TCP Variant Predictor – Demo Scenarios")
    print("=" * 60)
    
    scenarios = [
        {"desc": "Low BW, Low Delay, No Loss (LAN-like)",
         "bw": 5, "delay": 5, "loss": 0},
        {"desc": "High BW, Low Delay, No Loss (Datacenter)",
         "bw": 100, "delay": 5, "loss": 0},
        {"desc": "Medium BW, High Delay, No Loss (Satellite)",
         "bw": 20, "delay": 200, "loss": 0},
        {"desc": "High BW, Medium Delay, Moderate Loss (WAN)",
         "bw": 50, "delay": 50, "loss": 2},
        {"desc": "Low BW, High Delay, High Loss (Mobile)",
         "bw": 1, "delay": 100, "loss": 5},
        {"desc": "High BW, High Delay, Low Loss (Intercontinental)",
         "bw": 100, "delay": 200, "loss": 1},
    ]
    
    for scenario in scenarios:
        print(f"\n  Scenario: {scenario['desc']}")
        result = predict_tcp_variant(model, scenario["bw"], scenario["delay"], scenario["loss"])
        display_prediction(result)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TCP Congestion Control Variant Predictor"
    )
    parser.add_argument("--bandwidth", "-b", type=float, help="Bandwidth in Mbps")
    parser.add_argument("--delay", "-d", type=float, help="Delay in ms")
    parser.add_argument("--loss", "-l", type=float, help="Packet loss percentage")
    parser.add_argument("--rtt", type=float, default=None, help="Baseline RTT in ms")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Start interactive prediction mode")
    parser.add_argument("--batch", type=str, help="CSV file for batch predictions")
    parser.add_argument("--demo", action="store_true",
                        help="Run demo with sample scenarios")
    parser.add_argument("--model", type=str, default=MODEL_PATH,
                        help=f"Model file path (default: {MODEL_PATH})")
    
    args = parser.parse_args()
    
    model = load_model(args.model)
    if model is None:
        sys.exit(1)
    
    if args.interactive:
        interactive_mode(model)
    elif args.batch:
        batch_predict(model, args.batch)
    elif args.demo:
        run_demo(model)
    elif args.bandwidth is not None and args.delay is not None and args.loss is not None:
        result = predict_tcp_variant(model, args.bandwidth, args.delay, args.loss, args.rtt)
        display_prediction(result)
    else:
        # Default: run demo
        run_demo(model)
