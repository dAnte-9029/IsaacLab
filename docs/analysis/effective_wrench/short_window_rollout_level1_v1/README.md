# Level 1 short-window rollout

- Type: teacher-forced offline rigid-body integration.
- Inputs: real log initial state and real-log-aligned predicted wrench sequence.
- This is an auxiliary test; it does not close the loop through simulated-state aero inputs.
- Split: `test`
- NN bundle: `/home/zn/flap-system-identification/artifacts/20260507_temporal_backbone_final/runs/final_transformer_d64_l2_h4_hist128/causal_transformer_paper_no_accel_v2_phase_actuator_airdata/model_bundle.pt`

## Median Final Position Error

- 0.5s delaurier_gain_calibrated: 0.541314 m (IQR 0.315618)
- 0.5s delaurier_physically_calibrated: nan m (IQR nan)
- 0.5s delaurier_uncalibrated: nan m (IQR nan)
- 0.5s label_oracle: 0.138281 m (IQR 0.0824082)
- 0.5s nn_effective_wrench: 0.16917 m (IQR 0.091111)
- 1s delaurier_gain_calibrated: 2.463 m (IQR 0.911858)
- 1s delaurier_physically_calibrated: nan m (IQR nan)
- 1s delaurier_uncalibrated: nan m (IQR nan)
- 1s label_oracle: 0.574537 m (IQR 0.463427)
- 1s nn_effective_wrench: 1.10309 m (IQR 0.705054)
