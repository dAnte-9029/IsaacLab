# FlappingBot Straight-Flight RL Known Issues

## Hydra + dataclasses gotchas

- Frozen dataclasses in env configs can crash training:
  - Symptom: `dataclasses.FrozenInstanceError` during `env_cfg.from_dict(...)`.
  - Fix: do not use `@dataclass(frozen=True)` for config objects passed through Hydra.

- Tuples of nested config objects can be converted into tuples of dicts:
  - Symptom: `AttributeError: 'dict' object has no attribute ...` when initializing models from cfg.
  - Fix: use `list[...]` for collections of nested config objects (e.g. `wings=[WingQSMCfg(...), ...]`),
    so `from_dict` can merge into existing objects instead of replacing the collection.

## Headless warnings

- `GLFW initialization failed` is expected when running with `--headless`.

## Asset USD reference warnings

- You may see warnings about unresolved prim references for `flap_robot_v50.usd` during headless runs.
  - If simulation still runs and the URDF is used for physics, this is usually not a blocker for training.
  - If you see missing visuals or missing links, re-run `./isaaclab.sh --install` and ensure asset paths exist.

