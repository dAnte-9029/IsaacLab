# Controllers and Control Allocation

Controller changes may include new cascades, nonlinear control, MPC, learned control, allocation, or estimator structures when requested. Do not preserve the existing PX4-like cascade merely because it is the current implementation.

Make action/sign/units, actuator constraints, saturation, state/reset behavior, and control-rate assumptions explicit where relevant. Do not silently rewrite physical plant parameters to improve controller metrics.

For a fair comparison, keep scenarios, actuator limits, and evaluation metrics aligned; separately tune competing controllers when their architectures require it. Do not tune a final held-out case and then report it as unseen evaluation.

Run focused checks on the changed control path (for example sign, reset, saturation, or one closed-loop trajectory). Do not automatically require tests for every loop, broad gain sweeps, detailed telemetry inventories, or regression of unrelated controller levels.
