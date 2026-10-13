# Adaptive TCP Congestion Control Decision Engine

**Using Mininet and Machine Learning**

A Data-Driven System for Selecting the Best-Performing TCP Congestion Control Algorithm under Varying Network Conditions

## Team Members
- **Rohan Kumar** – 24BDS0463 (Networking & Experimentation)
- **Shreyans Bohra** – 24BCT0013 (Data & ML)

**Course:** Computer Networks (BCSE308P)

---

## Project Overview

This project develops an **Adaptive TCP Congestion Control Decision Engine** that:
1. Simulates varying network conditions (bandwidth, delay, packet loss) using Mininet
2. Benchmarks TCP **Reno**, **Cubic**, and **BBR** under each condition
3. Collects performance metrics into structured datasets
4. Trains a **Random Forest classifier** to predict the best-performing TCP variant
5. Evaluates whether ML-based adaptive selection outperforms any single fixed algorithm

## System Architecture

```
Python Automation Script
         |
         v
Mininet Network (H1 -- S1 -- H2)
         |
tc/netem: Bandwidth / Delay / Loss
         |
   ─────────────────────────────────
   |              |                |
TCP Reno     TCP Cubic          TCP BBR
   |              |                |
   ─────────────────────────────────
         |
  iperf3 (throughput, retransmits)
  ping / ss (RTT)
         |
         v
  Raw CSV Dataset → Aggregated ML Dataset
         |
         v
  Random Forest Classifier
         |
         v
  Recommended TCP Variant
```

## Project Structure

```
CN_Project/
├── main.py                    # End-to-end pipeline runner
├── mininet_topology.py        # M1: H1–S1–H2 topology setup
├── network_config.py          # M2: tc/netem configuration
├── benchmark_runner.py        # M3: iperf3 automation & CSV logging
├── generate_synthetic_data.py # Synthetic data (Windows demo)
├── data_preparation.py        # M4: Dataset aggregation
├── ml_pipeline.py             # M5: Random Forest training & evaluation
├── predictor.py               # M6: TCP variant predictor CLI
├── visualizations.py          # 9 plot types for analysis
├── test_cases.py              # 10 test cases (TC-01 to TC-10)
├── requirements.txt           # Python dependencies
├── data/
│   ├── dataset_a_raw.csv      # Raw benchmark data (1,350 rows)
│   └── dataset_b_ml.csv       # Aggregated ML data (150 rows)
├── models/
│   ├── random_forest.joblib   # Trained RF model
│   └── decision_tree.joblib   # Baseline DT model
├── plots/                     # 9 visualization PNGs
└── results/
    ├── evaluation_results.json
    └── test_results.json
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete pipeline
python main.py

# Run predictor
python predictor.py --bandwidth 20 --delay 50 --loss 2
python predictor.py --demo
python predictor.py --interactive

# Run tests only
python test_cases.py
```

## Key Results

| Metric | Value |
|--------|-------|
| Dataset Size | 1,350 raw rows → 150 conditions |
| Random Forest Accuracy | 83.33% |
| Random Forest F1 Score | 0.8629 |
| Decision Tree Accuracy | 90.00% |
| Always Reno Throughput | 2.39 Mbps |
| Always Cubic Throughput | 2.64 Mbps |
| Always BBR Throughput | 20.66 Mbps |
| ML Adaptive Throughput | 20.56 Mbps |

## Running on Linux/WSL2 (Mininet)

```bash
# Setup (WSL2 Ubuntu)
sudo apt install mininet iperf3 iproute2 python3-pip
pip3 install pandas scikit-learn matplotlib seaborn

# Run experiments
sudo python3 run_experiment.py --mode pilot
sudo python3 run_experiment.py --mode full --repetitions 3

# Then run ML pipeline
python3 data_preparation.py
python3 ml_pipeline.py
python3 visualizations.py
python3 predictor.py --demo
```

## Test Results

All 10 test cases pass (TC-01 through TC-10):
- TCP variant availability ✓
- Topology structure ✓
- Network condition configuration ✓
- Benchmark per variant ✓
- Repetition & aggregation ✓
- Dataset B generation ✓
- Feature columns (no leakage) ✓
- Held-out evaluation ✓
- Predictor output ✓
- Baseline comparison ✓
