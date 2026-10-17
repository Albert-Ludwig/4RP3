# 10.16 test — RBF GP Baseline

Purpose: establish two temperature reconstruction and prediction baselines using the existing GPyTorch Exact GP code. No informative planning is included.

_**Update**_: after the test, 2 research targets:

1. GP kernel test, use different kernel or other method to enhance the performance of the GP field prediction.
2. GRG planning test, seek any better reasoning method to sample the temperature and give a path for tracing.

**Next step**: Main object: integrate the 3D GP and GRG, to prepare for the Nav2 autonomous navigation on bunker mini.

| Task                           | Method and objective                                                                                                                                     |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Synthetic spatiotemporal field | A 3D ARD RBF GP maps (x, y, t) to temperature. Reconstruct the field at several timestamps and evaluate held-out spatial and temporal predictions.       |
| Dashboard temperature series   | A temporal RBF GP reconstructs a missing interval and predicts future observations. This single exported series does not support spatial reconstruction. |

Training-only scaling and validation-based checkpoint selection are used. Evaluation retains GP prediction errors and predictive uncertainty. Synthetic ground truth is not treated as real spatial evidence.

## Index

| Location                                                                                | Contents                                                                                                                                                                      |
| --------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [10.16_test_results.ipynb](10.16_test_results.ipynb)                                    | Executed record of both experiments: configuration, prediction/evaluation code, numerical results, all figures and links to implementation source files                       |
| [Synthetic code](<10.16_testing_code/3D_time&spatial temp field rebuild & prediction/>) | Shared GP model, synthetic field, training, evaluation, plotting, tests and configuration                                                                                     |
| [Dashboard code](<10.16_testing_code/dashboard time temp prediction/>)                  | Real-data audit, temporal splits, training, plotting, tests and configuration; reuses the shared GP implementation                                                            |
| [Synthetic results](<10.16_result/3D_time&spatial temp field rebuild & prediction/>)    | `main/`, `spatial/`, `temporal/`: split data, checkpoints, histories, hyperparameters and predictions; `robustness/`: GP sample/noise sensitivity runs; `figures/`: nine PNGs |
| [Dashboard results](<10.16_result/dashboard time temp prediction/>)                     | Original export and audit; `gap/` and `forecast/`: split data, checkpoints, histories, hyperparameters and predictions; `figures/`: four PNGs                                 |
| Each result folder                                                                      | `metrics.csv`, executed `config.json`, provenance `manifest.json` and `verification.json`; additional NPZ/horizon/sensitivity tables belong to their respective task          |
| `10.16_testing_code/.venv/`, `.cache/`                                                  | Shared runtime environment and caches                                                                                                                                         |

The notebook replays saved checkpoints without retraining. Its Implementation references section links to the source files used by the executable workflows. The original reference notebooks remain unchanged.

## Rerun

From `10.16_testing_code/`, use the shared environment and a new output directory; completed results are protected from overwriting.

```bash
.venv/bin/python '3D_time&spatial temp field rebuild & prediction/experiment.py' --output '../10.16_result/3D_time&spatial temp field rebuild & prediction/rerun'
.venv/bin/python 'dashboard time temp prediction/dashboard_experiment.py' --output '../10.16_result/dashboard time temp prediction/rerun'
```
