# Adaptive TCP Congestion Control Decision Engine

## Verified state — 17 October 2026

### Dataset integrity

- `data/dataset_a_raw.csv` and `data/dataset_b_ml.csv` are synthetic
  development artifacts. Their models, plots, and reported scores are not
  experimental evidence.
- `data/dataset_a_smoke_real.csv` and `data/dataset_a_smoke_receiver.csv`
  are small Mininet smoke tests only. Their RTT values require validation.
- Pantheon CSVs are real historical measurements from one retrieved path. They
  cannot form a Reno/Cubic/BBR dataset because the retrieved records include
  Vegas, not Reno.

### Experiment pipeline

- `mininet_topology.py` now creates an unimpaired topology. Conditions are
  applied only by `network_config.py`.
- `benchmark_runner.py` writes real measurements to
  `data/dataset_a_mininet.csv` by default. It records `baseline_rtt_ms` from
  ping separately from post-selection `iperf_rtt_ms`, rejects incompatible CSV
  schemas, and does not convert failed runs into zero-throughput rows.
  `throughput_source` records whether the measurement came from client JSON or
  the iperf receiver log; missing iperf RTT remains explicitly unavailable.
- `data_preparation.py` rejects failed/incomplete conditions and uses
  `baseline_rtt_ms` for new Mininet datasets. It supports explicit
  `--input` and `--output` paths.
- `ml_pipeline.py` and `visualizations.py` accept explicit input/output paths,
  so real artifacts remain separate from synthetic artifacts.

### Validation performed

- Python compilation passed.
- An isolated full data-preparation → ML → visualization workflow passed using
  the synthetic dataset as a fixture, writing 2 models, evaluation JSON, and 9
  plots into a temporary directory.
- No existing datasets, model artifacts, plots, or results were overwritten.
- A real 18-condition Mininet pilot completed successfully on 17 October 2026:
  54 rows, one result for each Reno/Cubic/BBR combination, no failed rows, and
  baseline RTTs consistent with configured delay. It is a pipeline-validation
  dataset, not an ML evaluation dataset: it has one repetition and 10 of 18
  winning throughputs are within 5% of the runner-up.
- The full 150-condition × 3-repetition run completed (1,350 rows), but it has
  15 failed Reno rows at 1 Mbps / 0% loss, across all five delays and three
  repetitions. These rows had zero receiver throughput and missing iperf RTT.
- The former `data/dataset_a_mininet_full_repaired.csv` held 1,335 valid rows;
  all 1,335 matched the complete recovered file exactly, so that redundant
  legacy-schema intermediate was removed after verification.
- `repair_experiment.py` now resumes an existing partial output, checks every
  expected key against the source, retries each missing observation three
  times, prints iperf JSON error text and summary fields for failed runs, and
  exits nonzero if any rows remain missing. The client final-result exchange
  failed (`Bad file descriptor`), but the server log recorded a receiver result
  (`2.25 MBytes 937 Kbits/sec receiver`). The benchmark can now retain that
  actual receiver throughput with source `iperf_server_log_receiver_summary`;
  iperf RTT may be `-1` because it is not an input feature. Write the repaired
  rows to a new output file, preserving the original and legacy partial file.
- `data/dataset_a_mininet_full_recovered.csv` is now complete: 1,350 rows,
  150 conditions, 450 observations per TCP variant, three repetitions per
  condition/variant, 1,335 client-JSON receiver measurements, and 15 valid
  server-summary recoveries. No row has invalid throughput, baseline RTT, or
  retransmission values.
- Stage 6 completed using `data/dataset_b_mininet.csv`,
  `models/mininet_full/`, and `results/mininet_full/`. The deterministic
  condition-held-out evaluation used 120 training and 30 test conditions. The
  Random Forest achieved 76.67% accuracy (weighted F1 0.8207); always-BBR
  averaged 30.02 Mbps on the test set, while the adaptive policy averaged
  29.96 Mbps (−0.21%). Do not claim an adaptive gain. Label counts are imbalanced
  (BBR 117, Reno 23, Cubic 10 overall; test labels 24/5/1).
- Stage 7 dashboard is implemented in `dashboard.py`; its localhost HTTP
  summary and prediction endpoints passed smoke checks. Stage 8 findings are
  documented in `REPORT_MININET.md`. Run-specific plots are in
  `plots/mininet_full/`.

### Current status

BBR was enabled successfully on 17 October 2026; the available algorithms are
now `reno cubic bbr`.

All eight project workflow stages are complete for the controlled Mininet
dataset. To reopen the dashboard, run `.venv/bin/python dashboard.py` and visit
`http://127.0.0.1:8000`. The report and evaluation JSON are the handoff sources
for results; do not substitute the older synthetic artifacts in `results/` or
`plots/`.
