# 10.18 — Intel Lab temperature prediction and GP–GRG replay

## Objective

Train the existing 3D spatiotemporal GP on three days of recorded temperature. Test the following day. Use the trained GP for recorded-data informative sampling and compare recursive depths. Verify sampling events, time accounting, posterior updates and failure recovery.

The temperature-performance results in this report use Intel measurements. Navigation runs on the existing simulated map. Dense temperature maps are GP predictions; dense measured ground truth is unavailable.

## Dataset and training selection

Dataset: [linsea423/Intel_Lab_Data](https://github.com/linsea423/Intel_Lab_Data). Original provider and schema: [Intel Berkeley Research Lab dataset](https://db.csail.mit.edu/labdata/labdata.html). Downloaded commit: `f46a8102e0ae93c98969e919c0b68897a202e0f3`.

Original records, sensor coordinates, laboratory diagram and checksums: [Data/Intel_lab_data](../../Data/Intel_lab_data). Coordinates are in metres; temperature is in °C. Original timestamps have no specified timezone.

| Item | Setting |
|---|---|
| Historical dates | 2004-02-29, 2004-03-01, 2004-03-02. |
| Test date | 2004-03-03. |
| Continuous time | Hours from 2004-02-29 00:00; history [0,72), test [72,96). |
| Selected training records | Exactly 2,000: 667, 667 and 666 by day. |
| Selection method | Near-equal sensor quotas per day. Select distinct actual records nearest equally spaced time targets. Seed 18 assigns remainder quotas. |
| Sensor coverage | 53 training sensors and 52 test sensors. |
| Cleaning | Known coordinates, finite temperature and fixed 0–50°C plausibility screen. Preserve empty fields during parsing. |
| Available cleaned data | 280,944 historical records; 90,690 fourth-day records. |

The screen removes 151 historical and 205 test temperature records. Forty other four-day rows lack sensor/temperature values. This fixed screen does not establish that all retained sensor readings are accurate. No values are imputed or spatially interpolated. Test labels do not select training records, hyperparameters or normalization.

Selected observations retain their original record IDs, coordinates, acquisition times and temperature values:

- [train_2000.csv](../../Data/Intel_lab_data/processed/three_day_2000/train_2000.csv): selected history.
- [train_allocation.csv](../../Data/Intel_lab_data/processed/three_day_2000/train_allocation.csv): day/sensor quotas.
- [test_day4.csv](../../Data/Intel_lab_data/processed/three_day_2000/test_day4.csv): complete cleaned test day.
- [split.json](../../Data/Intel_lab_data/processed/three_day_2000/split.json): exact intervals, source and preprocessing counts.

[Record-provenance audit](10.18_result/baseline/intel_lab/record_provenance_audit.json) checks the selected history, complete test day and executed records from all three depths against their original `data.txt` line IDs. It also verifies that the saved GP's training inputs and targets match the 2,000 selected Intel observations exactly.

## Model and system logic

Input is z=(x,y,t); output is temperature. Reuse the existing constant-mean Exact GP and 3D ARD RBF:

$$
k(z,z')=\sigma_f^2\exp\left[-\frac12\left(
\frac{(x-x')^2}{\ell_x^2}+\frac{(y-y')^2}{\ell_y^2}+\frac{(t-t')^2}{\ell_t^2}\right)\right].
$$

Train using the same 300 Adam steps, initialization and CPU float64 implementation. Fit normalization and parameters on the 2,000 historical observations only. The fitted spatial lengthscales are 19.652 m and 25.751 m; the temporal lengthscale is 2.790 h. Learned likelihood-noise variance is 0.944°C². Parameters stay fixed during replay; successful observations update the conditioning dataset.

For available observations D, let C=K_XX+σ_n²I. Posterior mean and latent covariance are:

$$
\mu_Q=m_Q+K_{QX}C^{-1}(Y-m_X),\qquad
\Sigma_Q=K_{QQ}-K_{QX}C^{-1}K_{XQ}.
$$

Cholesky solves implement conditioning. Real-measurement intervals include likelihood noise; uncertainty maps show latent standard deviation.

System flow: historical records → initial GP → GRG route → execute next waypoint → acquire next recorded measurement → charge waiting and dwell → update GP → replan → return Home.

GRG retains the sampling-point joint MI objective:

$$
J(P\mid D)=\tfrac12\log\det\left(I+\Sigma_{S(P)S(P)\mid D}/\sigma_n^2\right).
$$

Every intentional event enters S(P), including repeated locations. Shared recursive junctions are represented once. Only successfully acquired observations update D. Reconstruction errors are evaluation metrics and do not steer the planner.

## Experiments and measured results

### 1. Fixed-origin fourth-day prediction

Predict all 90,690 valid fourth-day records at their actual positions and timestamps. The origin is 72 h. No fourth-day observations enter the GP in this experiment.

| Metric | Result |
|---|---:|
| RMSE | 4.93747°C |
| MAE | 3.57528°C |
| 95% observation-interval coverage | 91.05% |
| Mean 95% interval width | 17.54264°C |
| Training time | 85.03 s |
| Prediction time | 34.50 s |

The existing RBF has limited full-day forecast accuracy. Predictions approach the learned constant mean away from historical data, while uncertainty widens. The observed daily changes are not reproduced well. This experiment does not isolate the kernel as the sole source of error.

Data, per-hour/per-sensor metrics and figures: [forecast report](10.18_result/baseline/intel_lab/forecast/report.md).

### 2. Intel GP–GRG baseline replay

Replay the first 900 s of the fourth day. Keep the 30×30 m map, two synthetic obstacles, footprint, 0.5 m navigation grid, 0.35 m/s speed, 5 s dwell and depth-2 default. The user authorized Home at sensor 16, (1.5,2) m. Use 29 footprint-reachable sensor coordinates as candidates; 23 other valid test sensors are outside or unreachable. Home and repeated spatial sampling remain eligible.

Within each sensor, sort test records by time. Even-indexed records form the acquisition pool; odd-indexed records form a disjoint evaluation pool. Planning can consult acquisition timestamps, not future temperatures. At navigation arrival, wait for the next acquisition-pool record, measure at its actual timestamp, then consume 5 s dwell. A stored record cannot be acquired twice.

$$
T_\mathrm{mission}=T_\mathrm{travel}+T_\mathrm{acquisition\ waiting}+T_\mathrm{dwell}
+T_\mathrm{online\ computation},\qquad T_\mathrm{online\ computation}=0.
$$

Waiting is included in the logged sampling component. GRG schedules the same acquisition times and reserves final return. No active waiting action or revisit cooldown is added. The known record-availability schedule is an offline replay assumption.

Evaluation uses 28 fixed held-out real records in the final five-minute window before 72.25 h. Their timestamps differ. The final posterior may include online data later than some reference records, so this evaluates retrospective reconstruction. Dense maps at 72.25 h retain the 0.75 m grid but have no dense measured truth. These targets differ from the full-day prediction experiment.

| Metric | History only | After online sampling |
|---|---:|---:|
| Held-out RMSE | 0.48958°C | 0.54691°C |
| Mean latent variance | 1.11523°C² | 0.43012°C² |
| 95% observation coverage | 100% | 100% |

The robot acquires 7 distinct records, makes 8 planning calls and returns at 870.26 s. Acquisition waiting takes 368.15 s; dwell takes 35 s. All 12 saved mission checks pass. Modeled variance decreases, but held-out temperature accuracy does not improve.

Logs, measured reference records, maps and checks: [baseline replay report](10.18_result/baseline/intel_lab/replay/report.md).

### 3. Recursive-depth benchmark

Use identical history, initial checkpoint, sensor candidates, map, record availability and held-out targets. Change recursive depth only. Check one deterministic depth-2 repeat. The planning-call and mission-worker watchdogs are 90 s and 900 s wall time; neither is charged to the mission clock. Depth 5 is excluded.

| Depth | Measurements | Final RMSE (°C) | Mean latent variance (°C²) | Joint MI (nats) | Total GRG wall time (s) |
|---|---:|---:|---:|---:|---:|
| 1 | 6 | 0.56980 | 0.47823 | 1.89694 | 0.022 |
| 2 | 7 | 0.54691 | 0.43012 | 2.15273 | 2.936 |
| 3 | 12 | 0.49041 | 0.29721 | 2.90277 | 238.624 |

All depths return at about 870.26 s. Depth 3 collects more MI and reduces modeled variance further, with much higher planning latency. No run lowers RMSE below the history-only value. The depth-2 repeat reproduces observations, trajectory, metrics, routes and event times. One fixed dataset does not establish statistical significance or planner superiority.

Detailed runs and timing: [depth report](10.18_result/grg_depth_benchmark/intel_lab/report.md).

### 4. Correctness verification

Passed: 8 updated recursive categories, 52 baseline regressions, 6 historical artifact/watchdog checks and 5 Intel data/replay checks. Maximum production-versus-independent MI difference is 1.424×10⁻¹¹ nats.

The recursive and failure tests use small controlled fixtures, including a historical synthetic-trained checkpoint. Their results verify program behavior, not Intel temperature accuracy. Intel-specific tests check real-record time matching, acquisition waiting, record uniqueness, chronological splitting and executed-only posterior updates.

Revisits include A-B-A and A-A, with separate times and dwells. Temporary navigation failure excludes exactly the next top-level cycle. Execute a feasible alternative action or terminate safely; do not advance artificial empty cycles. Permanent exclusion requires static-map evidence. Failed measurements do not update the GP. Partial-motion failure retains safe return from the actual pose. The existing 200-iteration guard remains a mission bound.

Evidence and remaining limitations: [correctness report](10.18_result/grg_correcness/current/report.md).

## Software architecture and next replacement points

| Code | Responsibility |
|---|---|
| [intel_lab/data.py](testing_code/intel_lab/data.py) | Parse, screen and select real historical records; preserve provenance. |
| [intel_lab/forecast.py](testing_code/intel_lab/forecast.py) | Fit the existing GP and evaluate fourth-day prediction. |
| [intel_lab/replay.py](testing_code/intel_lab/replay.py) | Recorded measurement source, timestamp scheduling, sensor candidates and held-out evaluation. |
| [baseline/gp.py](testing_code/baseline/gp.py) | Shared GP training, posterior covariance and online conditioning. |
| [baseline/grg.py](testing_code/baseline/grg.py) | Shared recursive search, ordered sampling events and MI. |
| [baseline/mission.py](testing_code/baseline/mission.py), [failures.py](testing_code/baseline/failures.py) | Execute, validate, account for time, condition, replan and manage exclusions. |
| [baseline/navigation.py](testing_code/baseline/navigation.py) | Footprint inflation, reachable grid, Dijkstra paths and pose/time feedback. |
| [intel_lab/benchmark.py](testing_code/intel_lab/benchmark.py) | Shared-checkpoint depth runs, watchdogs and repeat checks. |

ROS/Nav2 integration must replace simulated navigation and recorded acquisition with actual pose/time/status and synchronized sensor measurements. Preserve coordinate and timestamp associations. Add computation charges consistently to the mission clock. Arbitrary-pose replanning, dynamic-obstacle recovery and real sensing remain unverified.

Acceptance requires valid record provenance, numerical MI agreement, correct event timing and dwell, successful-only GP updates, one-cycle exclusions, collision-free paths, budget compliance and safe return. Improved RMSE is evaluated separately. Kernel comparisons, target-field reward changes and reconstruction optimization were not part of this run.

## Reports and commands

- [Main Notebook](10.18_test_report.ipynb): current Intel experiments, recorded outputs, figures and conclusions.
- [Intel results copy](10.18_result/baseline/intel_lab/intel_lab_results.ipynb): the same current results under the result folder.
- [Archived synthetic Notebook](10.18_result/baseline/archived_notebooks/10.18_synthetic_report.ipynb): previous cells and outputs, preserved separately.

From this folder, use the Python environment at `../10.16_GP3D_baseline_test/10.16_testing_code/.venv/bin/python`:

The repository includes the complete raw ZIP and processed training/test CSVs. The extracted raw text stays local. Before regenerating the data split, extract it once:

```bash
unzip -n ../../Data/Intel_lab_data/raw/data.zip -d ../../Data/Intel_lab_data/raw
```

```bash
python testing_code/intel_lab/data.py
python testing_code/intel_lab/forecast.py
python testing_code/intel_lab/benchmark.py
python testing_code/grg_correctness/run_checks.py
python testing_code/intel_lab/verify.py
python testing_code/intel_lab/plots.py
python testing_code/intel_lab/report.py
```

The report builder writes the current Intel Notebook from saved real-data results. It preserves earlier synthetic cells in the separate archive. It does not repeat training or add old performance tables to the current report.
