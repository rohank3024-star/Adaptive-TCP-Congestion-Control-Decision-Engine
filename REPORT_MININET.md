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

## Paired-flow fairness and loaded-latency extension

`fairness_experiment.py` adds a separate experiment rather than appending
incompatible measurements to the original single-flow ML dataset. Two hosts
run simultaneous TCP transfers to a third host through a shared bandwidth-
limited switch egress. The bottleneck uses either `pfifo` or `fq_codel` with
the same 1,000-packet limit; endpoint egress qdiscs apply the configured
one-way delay and loss. A ping from each sender runs during both transfers.
Each complete paired run records receiver throughput and sender retransmits
for both flows, per-sender loaded ping RTT/loss, and Jain's fairness index.
Unloaded baseline ping is captured immediately before the simultaneous flows,
so RTT increase under load is directly available. TCP algorithm roles, pair
order, and queue-discipline order alternate between repetitions to reduce
host-role and ordering bias.
The qdisc is checked after installation before data collection.

Run a short Reno-vs-BBR smoke check first, writing to a new file:

```bash
sudo python3 fairness_experiment.py --mode smoke --repetitions 1 \
  --output data/dataset_c_fairness_smoke.csv
```

The smoke run tests both queue disciplines at a 20 Mbps shared bottleneck,
50 ms endpoint delay, and no configured loss. After checking it, run all three
variant pairings with repetitions (6 balances each algorithm's sender role):

```bash
sudo python3 fairness_experiment.py --mode pilot --repetitions 6 \
  --bandwidth 20 --delay 50 --loss 0 \
  --output data/dataset_c_fairness_mininet.csv
sudo python3 fairness_experiment.py --mode pilot --repetitions 6 \
  --bandwidth 5 --delay 20 --loss 0 \
  --output data/dataset_c_fairness_5mbps.csv
sudo python3 fairness_experiment.py --mode pilot --repetitions 6 \
  --bandwidth 50 --delay 100 --loss 0 \
  --output data/dataset_c_fairness_50mbps.csv
python3 analyze_fairness.py data/dataset_c_fairness_mininet.csv \
  --max-loaded-rtt-increase-ms 20 \
  --output results/fairness_mininet_summary.json
python3 analyze_fairness.py data/dataset_c_fairness_5mbps.csv \
  --max-loaded-rtt-increase-ms 20 \
  --output results/fairness_5mbps_summary.json
python3 analyze_fairness.py data/dataset_c_fairness_50mbps.csv \
  --max-loaded-rtt-increase-ms 20 \
  --output results/fairness_50mbps_summary.json
python3 dashboard.py \
  --fairness-csv data/dataset_c_fairness_mininet.csv \
  --fairness-csv data/dataset_c_fairness_5mbps.csv \
  --fairness-csv data/dataset_c_fairness_50mbps.csv
```

### Initial smoke observation (exploratory only)

The first smoke test completed both queue cases with no failed runs and produced
four flow rows in `data/dataset_c_fairness_smoke.csv`. For one Reno-vs-BBR pair,
the measured summary was:

| Queue | Reno / BBR throughput | Mean loaded RTT | RTT increase over baseline | Ping loss | Jain fairness |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pfifo` | 12.01 / 1.78 Mbps | 1031.27 ms | 930.91 ms | 20% (1 of 5 probes) | 0.645 |
| `fq_codel` | 7.47 / 10.27 Mbps | 100.40 ms | approximately 0 ms | 0% (0 of 5 probes) | 0.976 |

Baseline RTT was about 100 ms in both cases, consistent with the configured
50 ms one-way endpoint delay. This single pair/repetition suggests severe
queueing delay and lower fairness with FIFO in this particular run, while
FQ-CoDel kept measured RTT near baseline. The 5-packet ping sample is coarse,
and one repetition cannot establish a general queue-discipline effect; use the
repeated pilot and confidence intervals before drawing conclusions.

### Six-repetition pilot result (one configured network condition)

The pilot completed all 36 paired tests (18 per queue discipline), with no
failed or incomplete pairs. The following are paired differences computed for
each algorithm pair and repetition as `FQ-CoDel minus pfifo`; intervals are
two-sided 95% Student-t intervals over the six paired repetitions:

| Competing flows | Aggregate throughput change (Mbps) | Loaded RTT increase change (ms) | Jain fairness change |
| --- | ---: | ---: | ---: |
| BBR + Cubic | -0.33 [-0.45, -0.22] | -222.78 [-234.39, -211.18] | +0.244 [+0.222, +0.266] |
| BBR + Reno | +5.72 [+4.75, +6.69] | -953.39 [-972.01, -934.76] | +0.329 [+0.304, +0.353] |
| Cubic + Reno | +3.28 [+2.44, +4.11] | -948.42 [-1046.46, -850.38] | +0.352 [+0.286, +0.418] |

In this setup (20 Mbps shared egress, 50 ms one-way endpoint delay, 0% configured
loss), FQ-CoDel reduced loaded RTT and improved fairness for all three flow
pairs. Its aggregate throughput was slightly lower for BBR+Cubic, but higher
for the pairs involving Reno. The pilot therefore illustrates a measurable
latency/fairness versus throughput tradeoff rather than a universal winner.
These intervals summarize one emulated topology and one condition with six
repetitions; they do not establish behavior across other bandwidths, delays,
loss rates, topologies, or real Internet paths. They also are not adjusted for
the three pairwise comparisons.

### Cross-condition follow-up

Two additional six-repetition pilots used the same topology, queue setup, and
zero configured loss, changing the bottleneck bandwidth and one-way delay.
Together with the 20 Mbps / 50 ms pilot above, these provide 108 complete
paired tests (36 per condition). The table reports paired `fq_codel - pfifo`
changes; intervals are 95% Student-t confidence intervals over six repetitions.

| Condition | Competing flows | Aggregate throughput Δ Mbps [95% CI] | Loaded RTT-increase Δ ms [95% CI] | Jain fairness Δ [95% CI] |
| --- | --- | ---: | ---: | ---: |
| 5 Mbps / 20 ms | BBR + Cubic | -0.09 [-0.15, -0.03] | -185.90 [-187.62, -184.19] | +0.227 [+0.222, +0.232] |
| 5 Mbps / 20 ms | BBR + Reno | +1.38 [+1.27, +1.49] | -1585.95 [-2163.29, -1008.61] | +0.411 [+0.402, +0.421] |
| 5 Mbps / 20 ms | Cubic + Reno | +1.22 [+0.99, +1.44] | -1603.02 [-1877.37, -1328.68] | +0.430 [+0.425, +0.435] |
| 20 Mbps / 50 ms | BBR + Cubic | -0.33 [-0.45, -0.22] | -222.78 [-234.39, -211.18] | +0.244 [+0.222, +0.266] |
| 20 Mbps / 50 ms | BBR + Reno | +5.72 [+4.75, +6.69] | -953.39 [-972.01, -934.76] | +0.329 [+0.304, +0.353] |
| 20 Mbps / 50 ms | Cubic + Reno | +3.28 [+2.44, +4.11] | -948.42 [-1046.46, -850.38] | +0.352 [+0.286, +0.418] |
| 50 Mbps / 100 ms | BBR + Cubic | -1.19 [-1.63, -0.75] | -190.28 [-195.46, -185.10] | +0.034 [-0.005, +0.072] |
| 50 Mbps / 100 ms | BBR + Reno | +1.94 [-0.33, +4.21] | -269.37 [-271.47, -267.27] | +0.192 [+0.165, +0.220] |
| 50 Mbps / 100 ms | Cubic + Reno | -17.34 [-18.70, -15.98] | -295.57 [-325.30, -265.85] | +0.283 [+0.131, +0.435] |

Across the nine pair/condition combinations, FQ-CoDel reduced measured loaded
RTT increase. Aggregate throughput varied: it was not uniformly higher, and
the observed reduction for Cubic+Reno at 50 Mbps / 100 ms is an important
tradeoff rather than a result to hide. The fairness-change interval for
BBR+Cubic at 50 Mbps / 100 ms includes zero, so this experiment does not show a
clear fairness improvement for that particular combination. These three
controlled points are stronger than a single-condition pilot but still do not
support broad claims about other network paths or conditions. Per-condition
CSV files and machine-readable summaries are kept separate under `data/` and
`results/` so they are not mistaken for rows in the ML training dataset.

The analyzer skips incomplete pairs and reports per-queue/per-algorithm-pair
means, paired FQ-CoDel-minus-FIFO differences, and 95% Student-t confidence
intervals over complete repetitions. The
optional RTT-increase ceiling reports throughput only for observations meeting
that explicit latency objective; choose and justify the ceiling for the target
application rather than treating 20 ms as universal. This CSV is a new
measurement artifact; it must not be fed to the current single-flow
throughput-label model. These smoke observations are not pilot results. The
queue comparison isolates the switch's server-facing egress qdisc, while the
configured endpoint loss and delay model a controlled test path, not a live
Internet route.

The run-specific plots are in [`plots/mininet_full`](plots/mininet_full), and
the machine-readable evaluation is
[`results/mininet_full/evaluation_results.json`](results/mininet_full/evaluation_results.json).
