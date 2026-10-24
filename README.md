# Adaptive TCP Congestion Control Decision Engine

**Using Mininet and Machine Learning**

A data-driven system for selecting the highest-throughput TCP congestion-control
variant under varying controlled network conditions.

## Team Members

- **Rohan Kumar** – 24BDS0463 (Networking & Experimentation)
- **Shreyans Bohra** – 24BCT0013 (Data & ML)

**Course:** Computer Networks (BCSE308P)

## Project Overview

This project:

1. Simulates bandwidth, delay and packet loss with Mininet and `tc/netem`.
2. Benchmarks TCP **Reno**, **Cubic** and **BBR** under identical conditions.
3. Collects receiver throughput, sender retransmissions and baseline RTT.
4. Trains a Random Forest to predict the best-throughput TCP variant.
5. Compares adaptive selection with always-Reno, always-Cubic and always-BBR.

## System Architecture

```text
Python Automation Script
         |
         v
Mininet Network (H1 -- S1 -- H2)
         |
tc/netem: Bandwidth / Delay / Loss
         |
   ---------------------------------
   |               |               |
TCP Reno       TCP Cubic         TCP BBR
   |               |               |
   ---------------------------------
         |
iperf3 (receiver throughput, sender retransmits)
ping (pre-selection baseline RTT)
         |
         v
Raw Benchmark CSV → Aggregated ML CSV
         |
         v
Decision Tree baseline + Random Forest
         |
         v
Advisory recommendation: Reno / Cubic / BBR
```

Throughput is used to label and evaluate conditions; it is never a classifier
input. The dashboard is advisory only: it does not switch the active host TCP
algorithm or run live network traffic.

## Review 1, 2 and 3 Summary

| Review | Focus | Project evidence |
| --- | --- | --- |
| Review 1 | Problem, design and planning | Defined the H1–S1–H2 topology, parameter grid, dataset design and condition-held-out evaluation. |
| Review 2 | Implementation and validation | Completed the automation, 54-run pilot, 1,350 valid full-run observations, aggregation, tests and advisory dashboard. |
| Review 3 | Results and conclusion | Evaluated the models, compared all fixed baselines, documented limitations and reported the separate fairness extension. |

Submission files: `Adaptive_TCP_Project_Presentation.pptx`,
`Adaptive_TCP_Review_2_Final.docx` and `Adaptive_TCP_Review_3_Final.docx`.

## Project Structure

```text
CN_Project/
├── main.py                    # Original synthetic-demo pipeline runner
├── mininet_topology.py        # M1: H1–S1–H2 topology setup
├── network_config.py          # M2: HTB/netem conditions and grids
├── benchmark_runner.py        # M3: iperf3 collection and CSV logging
├── run_experiment.py          # Controlled Mininet experiment runner
├── repair_experiment.py       # Recovery with measurement provenance
├── data_preparation.py        # M4: aggregation and completeness validation
├── ml_pipeline.py             # M5: model training and held-out/CV evaluation
├── predictor.py               # M6: advisory CLI predictor
├── dashboard.py               # Local results dashboard and advisory API
├── visualizations.py          # Generates nine analysis plots
├── fairness_experiment.py     # Separate paired-flow fairness experiment
├── analyze_fairness.py        # Fairness summaries and confidence intervals
├── test_cases.py              # Ten development-data checks
├── data/                      # Kept-separated input and measurement datasets
├── models/mininet_full/       # Final Mininet models
├── results/mininet_full/      # Final evaluation JSON
└── plots/mininet_full/        # Nine final Mininet plots
```

## Latest Controlled-Mininet Results

The final single-flow experiment contains **1,350 valid observations** from
150 conditions × 3 variants × 3 repetitions. Dataset B has one complete record
per condition; its labels are **117 BBR, 23 Reno and 10 Cubic**.

| Result | Value |
| --- | ---: |
| Random Forest held-out accuracy (30 conditions) | 76.67% |
| Random Forest 5-fold out-of-fold accuracy | 84.00% |
| Held-out always-BBR throughput | 30.02 Mbps |
| Held-out adaptive throughput | 29.96 Mbps |
| Adaptive vs best fixed policy | −0.21% |

The project therefore does **not** claim an adaptive throughput improvement.
Always-BBR is already near-optimal for this throughput-only controlled grid.
This negative result is valid and is reported honestly. See
[REPORT_MININET.md](REPORT_MININET.md) for the full method and findings.

## Data Integrity and Provenance

Do not combine the following data sources:

| Data | Role | Used for final ML results? |
| --- | --- | --- |
| `dataset_a_mininet_full_recovered.csv` | Complete raw controlled-Mininet data | Yes |
| `dataset_b_mininet.csv` | 150-condition aggregated ML data | Yes |
| `dataset_a_raw.csv`, `dataset_b_ml.csv` | Synthetic development fixtures | No |
| Smoke and pilot CSVs | Pipeline validation/preliminary observation | No ML evaluation |
| Pantheon CSVs | Separate historical measurements that include Vegas | No; Vegas is never relabelled as Reno |
| `dataset_c_fairness_*.csv` | Separate two-flow experiments | No; incompatible with the single-flow model |

The recovered full dataset contains 1,335 client-JSON receiver measurements
and 15 tagged server-summary recoveries. No failed run was stored as a
zero-throughput measurement.

## Setup on Linux / WSL2

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Mininet collection also requires Linux networking support:

```bash
sudo apt update
sudo apt install mininet iperf3 iproute2 openvswitch-switch python3-pip
sysctl net.ipv4.tcp_available_congestion_control
```

Confirm that `reno`, `cubic` and `bbr` are listed before any collection. Do
not run privileged or long Mininet experiments without authorization.

## Quick Start: Predictor and Dashboard

```bash
python3 predictor.py \
  --model models/mininet_full/random_forest.joblib \
  --bandwidth 20 --delay 50 --loss 2

python3 dashboard.py
# Open http://127.0.0.1:8000
```

If baseline RTT is omitted, the predictor estimates it as approximately
`2 × delay × 1.02`; a measured baseline RTT is preferable.

## Reproduce the Completed Analysis Safely

Use separate output paths so submitted artifacts are not overwritten:

```bash
mkdir -p artifacts/reproduced/models artifacts/reproduced/results artifacts/reproduced/plots

python3 data_preparation.py \
  --input data/dataset_a_mininet_full_recovered.csv \
  --output artifacts/reproduced/dataset_b_mininet.csv

python3 ml_pipeline.py \
  --input artifacts/reproduced/dataset_b_mininet.csv \
  --model-dir artifacts/reproduced/models \
  --results-dir artifacts/reproduced/results \
  --cv-folds 5

python3 visualizations.py \
  --raw data/dataset_a_mininet_full_recovered.csv \
  --ml artifacts/reproduced/dataset_b_mininet.csv \
  --results artifacts/reproduced/results/evaluation_results.json \
  --output-dir artifacts/reproduced/plots
```

## Generate the Nine Final Plots

The report and presentation use the evidence-backed plots in
`plots/mininet_full/`. Regenerate them with:

```bash
python3 visualizations.py \
  --raw data/dataset_a_mininet_full_recovered.csv \
  --ml data/dataset_b_mininet.csv \
  --results results/mininet_full/evaluation_results.json \
  --output-dir plots/mininet_full
```

| Plot | Purpose |
| --- | --- |
| `01_throughput_comparison.png` | Mean throughput by variant |
| `02_best_variant_distribution.png` | Conditions won by each variant |
| `03_throughput_vs_params.png` | Throughput versus controlled parameters |
| `04_variant_heatmap.png` | Detailed condition-wise median throughput |
| `05_feature_importance.png` | Random Forest feature importance |
| `06_confusion_matrix.png` | Held-out Random Forest predictions |
| `07_baseline_comparison.png` | Adaptive, fixed-baseline and oracle throughput |
| `08_retransmissions.png` | Retransmission comparison |
| `09_model_comparison.png` | Decision Tree versus Random Forest metrics |

Use `plots/mininet_full/` for final evidence, not plots made from synthetic
development data.

## Optional: New Mininet and Fairness Experiments

The completed results above are the submission evidence. A new pilot/full run
or repair should write to a new file and requires authorization; do not
overwrite the completed datasets.

The fairness extension compares `pfifo` and FQ-CoDel with two simultaneous
flows. It stores its 108 complete paired tests separately and is not fed to the
single-flow ML model. To display the existing fairness datasets:

```bash
python3 dashboard.py \
  --fairness-csv data/dataset_c_fairness_mininet.csv \
  --fairness-csv data/dataset_c_fairness_5mbps.csv \
  --fairness-csv data/dataset_c_fairness_50mbps.csv
```

## Tests

```bash
python3 test_cases.py
python3 -m unittest -v test_fairness_experiment.py
python3 -m unittest -v test_ml_cross_validation.py
```

The ten `test_cases.py` checks are development-data checks. The real
controlled-network evidence is the completed 54/54 pilot and 1,350/1,350
full-run observations.

## Limitations

- One controlled Mininet topology, machine/kernel and fixed grid; not live Internet validation.
- Labels optimize median throughput only, not latency or fairness.
- Cubic is a minority class: 10 labels overall and one in the held-out split.
- The dashboard and predictor are advisory, not runtime TCP switching.
