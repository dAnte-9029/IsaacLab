# Physically calibrated DeLaurier baseline

- Dataset: `/home/zn/flap-system-identification/dataset/canonical_v0.2_training_ready_split_hq_v4_direct_airspeed_logsplit_paper_alt5_v1`
- Calibration rows: `12000`
- Best normalized objective: `8938.3794`
- Calibration: bounded random search over eight interpretable physical parameters using train data only.
- Evaluation: final parameters are frozen and evaluated on full train/val/test splits.

## Rows

- train: 308702
- val: 79587
- test: 60671

## Parameters

- wing_normal_force_scale: 0.535448178
- wing_chordwise_force_scale: 1.44343476
- delaurier_theta_w_deg: 0.544054223
- twist_eta_max_deg: 9.61232661
- delaurier_induced_drag_efficiency: 0.678259255
- fuselage_drag_cda: 0.00962193853
- tail_lift_scale: 0.46929097
- phase_delay_s: 0.0341818693

## Test RMSE

- fx_b: 6.64018 -> 5.42031
- fy_b: 6.11102 -> 3.52358
- fz_b: 18.4299 -> 13.105
- mx_b: 0.151777 -> 0.0714185
- my_b: 1.2851 -> 0.601734
- mz_b: 0.210934 -> 0.101904
