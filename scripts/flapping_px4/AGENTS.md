
Flapping-PX4 Script Guidelines

These instructions apply to command-line entry points, parameter sweeps, evaluation scripts, plotting scripts, and result-export workflows in this directory.

1. Keep scripts thin

Scripts should:

parse arguments;
construct configurations;
call reusable package APIs;
save results;
report commands and output locations.

Scripts must not contain new aerodynamic formulas, controller implementations, or duplicated environment logic.

Move reusable logic into source/flapping_bot/.

2. Reproducibility

Every experiment or sweep should record where applicable:

experiment name;
timestamp;
Git commit SHA;
working-tree dirty state;
command-line invocation;
plant configuration;
controller profile;
mission configuration;
random seed;
checkpoint path and hash;
output schema version.

Do not silently overwrite previous experiment outputs.

3. Separation of workflows

Keep these workflows distinct:

parameter calibration;
controller tuning;
held-out evaluation;
sensitivity analysis;
plotting and reporting.

A held-out evaluation script must not tune parameters during evaluation.

A plotting script must not alter or regenerate source experiment data unless explicitly designed as a processing pipeline.

4. CLI behavior

New scripts should provide:

--help;
explicit input and output paths;
sensible but documented defaults;
a minimal smoke-run option when practical;
clear failure messages;
non-zero exit status on failure.

Avoid machine-specific absolute paths.

5. Output discipline

Large videos, checkpoints, raw logs, and sweep outputs must not be committed to Git.

Commit only:

scripts;
small manifests;
small summary tables when needed;
schemas;
plotting code;
documentation describing reproduction.

Before changing an existing output format, check downstream analysis and plotting scripts.
