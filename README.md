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
├── dashboard.py               # Local results dashboard and advisory predictor
├── visualizations.py          # 9 plot types for analysis
├── test_cases.py              # 10 test cases (TC-01 to TC-10)
├── repair_experiment.py       # Rerun missing Mininet observations
├── fetch_pantheon_data.py     # Separate Stanford Pantheon data fetcher
├── requirements.txt           # Python dependencies
├── data/
│   ├── dataset_a_raw.csv      # Synthetic development data (1,350 rows)
│   ├── dataset_a_mininet_full.csv          # Original full Mininet run
│   ├── dataset_a_mininet_full_recovered.csv # Complete, provenance-tagged dataset
│   ├── dataset_b_mininet.csv               # Aggregated real Mininet data
│   ├── dataset_pantheon_real.csv           # Separate historical Pantheon data
│   └── dataset_b_ml.csv                    # Synthetic development aggregate
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

## Latest real Mininet results

The completed Mininet experiment contains 1,350 observations across 150
conditions. On 30 held-out conditions, Random Forest accuracy was 76.67% and
adaptive throughput was 29.96 Mbps, versus 30.02 Mbps for always-BBR (−0.21%).
This result does not demonstrate an adaptive throughput improvement. The older
synthetic development scores are not experimental evidence. See
[`REPORT_MININET.md`](REPORT_MININET.md) for methodology, recovery provenance,
limitations, and the full results.

## Running on Linux/WSL2 (Mininet)

```bash
# Setup (WSL2 Ubuntu)
sudo apt install mininet iperf3 iproute2 python3-pip
pip3 install pandas scikit-learn matplotlib seaborn

# Run experiments
sudo python3 run_experiment.py --mode pilot
sudo python3 run_experiment.py --mode full --repetitions 3

# Then run ML pipeline
# This default processes the synthetic demonstration dataset only.
python3 data_preparation.py
python3 ml_pipeline.py
python3 visualizations.py
python3 predictor.py --demo
```

## Real Mininet Workflow

The default `dataset_a_raw.csv`, models, results, and plots are synthetic
development artifacts. Keep real Mininet measurements in their own files:

```bash
# After BBR is available and a Mininet pilot succeeds
sudo python3 run_experiment.py --mode smoke --repetitions 1 \
  --output data/dataset_a_mininet_smoke.csv
sudo python3 run_experiment.py --mode pilot --repetitions 3 \
  --output data/dataset_a_mininet.csv
python3 data_preparation.py --input data/dataset_a_mininet.csv \
  --output data/dataset_b_mininet.csv
python3 ml_pipeline.py --input data/dataset_b_mininet.csv \
  --model-dir models/mininet --results-dir results/mininet
python3 visualizations.py --raw data/dataset_a_mininet.csv \
  --ml data/dataset_b_mininet.csv --results results/mininet/evaluation_results.json \
  --output-dir plots/mininet
python3 predictor.py --model models/mininet/random_forest.joblib --demo
```

The completed full-grid analysis is stored separately in
`data/dataset_a_mininet_full_recovered.csv`, `data/dataset_b_mininet.csv`,
`models/mininet_full/`, `results/mininet_full/`, and `plots/mininet_full/`.
Start its local results dashboard with:

```bash
python3 dashboard.py
# Open http://127.0.0.1:8000
```

## Two-flow fairness and queue-discipline extension

The advanced experiment compares a shared FIFO (`pfifo`) bottleneck against
FQ-CoDel while two TCP senders run concurrently. It records per-flow receiver
throughput/retransmissions, Jain fairness, and ping RTT measured during load.
It writes a separate CSV and never modifies the completed single-flow dataset:

```bash
sudo python3 fairness_experiment.py --mode smoke --repetitions 1 \
  --output data/dataset_c_fairness_smoke.csv
sudo python3 fairness_experiment.py --mode pilot --repetitions 6 \
  --output data/dataset_c_fairness_mininet.csv
python3 analyze_fairness.py data/dataset_c_fairness_mininet.csv \
  --max-loaded-rtt-increase-ms 20 \
  --output results/fairness_mininet_summary.json
python3 -m unittest -v test_fairness_experiment.py
```

Run the smoke test before the pilot. The pilot comprises 36 paired tests
(three Reno/Cubic/BBR pairings × two queue disciplines × six repetitions),
with 10-second overlapping flows and a concurrent loaded-ping sample. Review
the report's experiment notes and inspect the resulting CSV before making
claims; the new measurements are not used by the existing single-flow model.
The optional RTT ceiling is an example application objective, not a universal
threshold; choose and justify it for your use case.

To show the queue/fairness tables in the dashboard, pass each condition's CSV
as a separate option (the dashboard keeps their provenance separate):

```bash
python3 dashboard.py \
  --fairness-csv data/dataset_c_fairness_mininet.csv \
  --fairness-csv data/dataset_c_fairness_5mbps.csv \
  --fairness-csv data/dataset_c_fairness_50mbps.csv
```

The dashboard displays measured Mininet throughput and held-out comparisons,
and provides an advisory model prediction. It does not change the active TCP
algorithm or run network traffic.

Do not combine synthetic, smoke-test, Pantheon, and Mininet datasets.

If an experiment CSV contains missing receiver-side metrics, preserve it and
repair only those rows into a new file:

```bash
sudo python3 repair_experiment.py --input data/dataset_a_mininet_full.csv \
  --output data/dataset_a_mininet_full_recovered.csv
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
