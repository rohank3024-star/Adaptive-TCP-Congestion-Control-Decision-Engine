# Adaptive TCP Decision Engine — Mininet Findings

## Executive summary

This report summarizes the controlled Mininet experiment and the held-out
evaluation completed on 17 October 2026. The experiment measured Reno, Cubic,
and BBR in 150 emulated network conditions, with three runs per variant and
condition (1,350 observations total). A Random Forest predicts the highest
measured-throughput variant from bandwidth, one-way delay, packet loss, and
pre-selection ping RTT.

The Random Forest reached 76.67% accuracy on 30 held-out conditions, but its
adaptive policy averaged 29.96 Mbps versus 30.02 Mbps for always-BBR on the
same test conditions (−0.21%). This experiment therefore does not demonstrate
that ML selection improves throughput over the best fixed policy. BBR won the
measured-throughput label in 117 of 150 conditions, so always-BBR is a strong
baseline in this particular test grid.

## Data and method

- Topology: Mininet H1–S1–H2, with configured bandwidth, one-way delay, and
  packet loss applied using `tc/netem`.
- Algorithms: Linux TCP Reno, Cubic, and BBR.
- Grid: 1, 5, 15, 50, and 100 Mbps; 5, 20, 50, 100, and 200 ms one-way delay;
  0%, 1%, 2%, 3%, and 5% loss; three repetitions per grid point.
- Raw dataset: [`data/dataset_a_mininet_full_recovered.csv`](data/dataset_a_mininet_full_recovered.csv)
  (1,350 rows, 150 complete conditions).
- Aggregated dataset: [`data/dataset_b_mininet.csv`](data/dataset_b_mininet.csv)
  (150 rows; per-variant median throughput across three runs).
- Model inputs: bandwidth, delay, loss, and baseline ping RTT. Post-selection
  iperf RTT is retained as an outcome field and is not used as a predictor.
- Split: deterministic 80/20 group split by complete condition (120 train,
  30 test; random state 42), preventing repeated runs of a condition from
  crossing the train/test boundary.

## Measurement recovery and provenance

The archived original [`data/dataset_a_mininet_full.csv`](data/dataset_a_mininet_full.csv)
contains 15 Reno rows at 1 Mbps and 0% loss with no
receiver throughput because iperf3 failed while returning its final client
result. The server log showed that the transfer itself had completed and
contained a receiver-side summary. Those 15 observations were rerun into the
separate recovered CSV and throughput was read from the server's final
receiver summary; values were 0.87–0.94 Mbps. Each row is marked
`iperf_server_log_receiver_summary` in
`throughput_source`. The 1,335 other valid rows are marked
`iperf_client_json_receiver`. The original full CSV and earlier 1,335-row
partial repair artifact were checked: every one of its rows exactly matched
the corresponding row in the recovered dataset. The redundant partial artifact
was then removed; both the original full CSV and complete recovered CSV remain.

The iperf RTT summary remained unavailable for these 15 recovered rows and is
stored as `-1`. Their pre-selection ping baseline RTT is present and valid;
that baseline is the RTT feature used for the model. All 1,350 observations
have positive throughput and baseline RTT, nonnegative retransmission counts,
and exactly three repetitions for each condition/variant group.

## Results

| Result | Value |
| --- | ---: |
| Complete raw observations | 1,350 |
| Complete network conditions | 150 |
| Best-throughput labels: BBR / Reno / Cubic | 117 / 23 / 10 |
| Decision Tree held-out accuracy | 83.33% |
| Random Forest held-out accuracy | 76.67% |
| Random Forest held-out weighted F1 | 0.8207 |
| Always Reno on held-out conditions | 8.02 Mbps |
| Always Cubic on held-out conditions | 8.39 Mbps |
| Always BBR on held-out conditions | 30.02 Mbps |
| Random Forest adaptive selection | 29.96 Mbps |
| Adaptive vs best fixed | −0.21% |

The held-out test labels were imbalanced (24 BBR, 5 Reno, 1 Cubic). The
Random Forest classified 21/24 BBR, 0/1 Cubic, and 2/5 Reno labels correctly.
The single Cubic example makes class-specific Cubic performance especially
uncertain. Decision Tree accuracy was higher than Random Forest accuracy on
this one fixed split; neither result should be generalized beyond the tested
grid without additional validation.

## Interpretation and limitations

These are real measurements from a controlled, locally emulated Mininet
network, not measurements from live Internet paths. The recovered rows are
real receiver-side summary measurements, but their iperf RTT outcome is
unavailable. The dataset covers one topology, one machine/kernel setup, a
fixed parameter grid, and three repetitions. Evaluation uses one 30-condition
holdout rather than repeated nested validation. The measured-throughput label
also reflects the tested conditions and short iperf runs; it is not a claim
about fairness, latency under load, application quality, or deployment safety.

The predictor is advisory only: the dashboard does not alter a host's active
TCP congestion control algorithm or launch a benchmark. If a user omits
baseline RTT, the dashboard estimates it as twice the configured one-way delay
plus 2%; a measured baseline RTT is preferable when available.

Pantheon historical data and synthetic development data are separate from
this Mininet dataset and were not used in these model results.

## Reproduce the analysis

```bash
python3 data_preparation.py \
  --input data/dataset_a_mininet_full_recovered.csv \
  --output data/dataset_b_mininet.csv
python3 ml_pipeline.py --input data/dataset_b_mininet.csv \
  --model-dir models/mininet_full --results-dir results/mininet_full
python3 visualizations.py \
  --raw data/dataset_a_mininet_full_recovered.csv \
  --ml data/dataset_b_mininet.csv \
  --results results/mininet_full/evaluation_results.json \
  --output-dir plots/mininet_full
python3 dashboard.py
```

The run-specific plots are in [`plots/mininet_full`](plots/mininet_full), and
the machine-readable evaluation is
[`results/mininet_full/evaluation_results.json`](results/mininet_full/evaluation_results.json).
