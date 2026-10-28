# Flapping-PX4 Scripts

Scripts can orchestrate experiments, command-line options, evaluation, plots, and exports. Keep reusable physics, controllers, and environment logic in `source/flapping_bot/` when practical, without creating an abstraction solely for a one-off experiment.

Use explicit plant/controller/mission selections and preserve existing results rather than silently overwriting them. Record the configuration, seed where applicable, and output location needed to interpret the result; do not generate automatic checkpoint hashes, verbose source receipts, or audit manifests without a concrete need.

Keep tuning data distinct from genuinely held-out evaluation. Do not alter the source experiment data while plotting. Add CLI or smoke-run features only when useful to the requested workflow; do not apply a universal checklist to every exploratory script.

Avoid committing large raw logs, videos, caches, or checkpoints by default. Do not launch unrequested grid sweeps, all-tests checks, or plotting variants.
