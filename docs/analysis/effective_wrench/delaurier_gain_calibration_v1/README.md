# Gain-calibrated DeLaurier baseline

- Dataset: `/home/zn/flap-system-identification/dataset/canonical_v0.2_training_ready_split_hq_v4_direct_airspeed_logsplit_paper_alt5_v1`
- Calibration: one scalar least-squares gain per wrench channel, fitted on train split only.
- Evaluation: the fitted train gains are frozen and applied to train/val/test predictions.
- Predictions saved: `False`

## Rows

- train: 308702
- val: 79587
- test: 60671

## Fitted gains

- fx_b: 0.350640194
- fy_b: -0.000336044682
- fz_b: 0.321617218
- mx_b: -0.00150171225
- my_b: -1.11352197e-05
- mz_b: 7.64921733e-06

## Test RMSE

- fx_b: 6.64018 -> 5.44259
- fy_b: 6.11102 -> 1.44597
- fz_b: 18.4299 -> 10.9555
- mx_b: 0.151777 -> 0.00331436
- my_b: 1.2851 -> 0.00432593
- mz_b: 0.210934 -> 0.000511986
