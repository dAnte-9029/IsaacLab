# ADR-2026-08-12: Project-local offline ground geometry

- Status: Accepted
- Date: 2026-08-12
- Scope: `FlappingRoomSceneCfg` ground creation and measured PureRL evaluation startup

## Context

A fresh C2b evaluation failed before environment construction in Isaac Lab's upstream ground-plane spawner. The
Kit log showed that the configured remote `default_environment.usd` could not produce an Sdf layer after an
approximately 46-second asset request. The upstream spawner then attempted `Stage.GetPrimAtPath()` with a null
child path. This made evaluation depend on remote asset availability even though the ground only supplies a flat
collision surface.

The project must not patch `source/isaaclab/` or other upstream framework packages. Saved run configuration can
also predate a scene fix, so measured PureRL authority evaluation must load the current registered task contract.

## Decision

`FlappingRoomSceneCfg` uses a project-local procedural static cuboid instead of `GroundPlaneCfg`. The cuboid is
500 by 500 by 0.1 m, centered at `z=-0.05 m`, so its top collision surface remains exactly at `z=0`. It retains
static and dynamic friction of 0.5 and zero restitution. Its simple black preview material requires no remote
asset.

Measured PureRL watcher commands use `--no_saved_cfg` so evaluation receives the current registered scene
contract. CPU-native watcher commands also override the robot source URDF and conversion directory in the same
way as training, using a writable watcher-specific portable cache rather than generating USD beside protected
source assets.

## Alternatives considered

- Retry or wait for the remote ground USD. Rejected because authority evaluation must not depend on network or
  asset-server availability.
- Patch the upstream ground-plane spawner to handle a failed child prim. Rejected because upstream Isaac Lab is
  outside the approved modification boundary and a graceful error would not remove the remote dependency.
- Commit or cache a copy of the upstream environment USD. Rejected because the task only needs flat collision
  geometry and a large copied asset would add unnecessary ownership and provenance work.
- Keep loading saved environment configuration for old runs. Rejected because it can silently reinstate the
  failed ground configuration and other obsolete evaluation contracts.

## Consequences

- Fresh `FlappingRoomSceneCfg` processes can create the ground without network access.
- The collision surface location, extent, friction, and restitution remain equivalent for the current tasks.
- The decorative ground appearance changes from the upstream environment asset to a plain black surface.
- All project tasks using `FlappingRoomSceneCfg` receive the local ground; plant dynamics, task rewards,
  observations, and termination thresholds are unchanged.
- Measured PureRL evaluation uses current registered configuration and writes generated robot assets only to the
  configured portable watcher cache.

## Assumptions

- A finite 500 by 500 m surface covers all currently approved flapping-room and curriculum trajectories.
- No accepted metric depends on the decorative upstream ground asset.
- Ground contact at `z=0` and the existing root-height termination remain the relevant physical contract.

## Validation requirements

- A static contract test must verify local cuboid geometry, placement, collision, and friction values.
- Measured PureRL watcher command tests must require `--no_saved_cfg`, the project URDF, and a watcher-specific
  writable USD cache.
- A fresh CPU-native evaluation must construct the scene and complete at least one full fixed suite.
- Protected upstream and asset source directories must remain clean after validation.

## Reconsideration triggers

Reconsider this decision if a task exceeds the 500 m footprint, requires terrain or visual ground features, or
if Isaac Lab provides a fully local primitive ground implementation whose startup and collision behavior are
demonstrably equivalent.
