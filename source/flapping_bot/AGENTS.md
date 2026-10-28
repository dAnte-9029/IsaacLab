# Flapping-Bot Extension

Usual responsibilities: `physics/` for aerodynamic/plant calculations, `px4_like/` for controller logic, `direct/` for environment integration, `path_tracking/` for references/guidance, and `assets/` / `scenes/` for robot and scene definitions.

Maintain understandable module boundaries, but allow a single approved research task to change physics, controllers, environment, observation, reward, and training code together. Do not force an integrated scientific idea into multiple sequential approval stages or protect the old architecture at the expense of the goal.

Keep physically measured parameters, model assumptions, and controller tuning separate and traceable. Avoid hidden configuration changes, unintended CPU tensors in GPU paths, shape/dtype errors, and unbounded NaN/Inf propagation. Batch operations should remain efficient.

Add only diagnostics needed to attribute the changed behavior. Preserve a usable reference baseline for direct comparison when the experiment needs one; do not automatically implement every alternate mode, a full regression battery, or a redundant logging framework.

Consult specialized nested `AGENTS.md` files for the modules being changed, not every instruction in unrelated subdirectories.
