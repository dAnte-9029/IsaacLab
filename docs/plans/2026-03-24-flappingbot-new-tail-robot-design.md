# New Tail Robot Integration Design

**Context**

The latest robot model at `/home/zn/flapping_wing_robot/urdf/固定翼扑翼机552.SLDASM.urdf` changes the tail architecture from the previous simplified tail pair into:

- two flapping wing joints
- one vertical-tail control joint (`rudder`)
- two horizontal-tail control joints (`left_elevon`, `right_elevon`)

The current repository assumes a four-joint robot in the straight-flight stack and a two-tail-joint robot in the older direct environment. That assumption is now incorrect for the latest hardware-aligned URDF.

**Goal**

Unify the entire IsaacLab flapping stack on the latest robot while preserving the existing four-channel control interface:

- `flap_freq`
- `rudder`
- `elevon_pitch`
- `elevon_roll`

All task families should use the same physical robot asset. The simulation should command five physical joints but expose four control channels to keep the PX4-like controller, baseline scripts, and RL training interfaces stable.

**Robot Integration Approach**

The asset layer will be updated so that `FlappingBotCfg` points to a new repository-managed URDF/USD pair derived from the latest model. Joint and link names will be normalized to the existing code conventions where possible:

- `left_wing`
- `right_wing`
- `rudder`
- `left_tail`
- `right_tail`

This keeps the straight-flight stack mostly compatible while making the new rudder explicit as a real joint. The old tail pair remains the left/right horizontal control surfaces.

The asset configuration must also be extended to:

- initialize the new `rudder` joint
- add a separate actuator group for the rudder
- preserve the wing actuator group
- preserve the elevon actuator group

**Control Mapping**

The public control interface remains four-dimensional. Internally the environment maps it to five physical joints:

- action 0 -> flapping frequency for both wings
- action 1 -> rudder joint angle
- action 2 -> symmetric elevon component
- action 3 -> differential elevon component

The resulting physical tail commands are:

- `rudder = rudder_cmd`
- `left_tail = elevon_pitch + elevon_roll`
- `right_tail = elevon_pitch - elevon_roll`

This keeps the PX4-like controller and straight-flight RL action contract unchanged while aligning the robot model with the actual mechanism.

**Tail Aerodynamic Model**

The tail model will be upgraded from the current lumped `elevator + rudder` representation into a five-surface structure:

- fixed horizontal stabilizer
- left elevon
- right elevon
- fixed vertical stabilizer
- rudder

Each surface uses the same low-order quasi-steady local-flow formulation based on:

- body-frame translational velocity
- body-frame angular velocity
- surface moment arm
- hinge axis
- chord axis
- area
- lift/sideforce slope
- drag polar
- deflection angle for movable surfaces

Fixed surfaces have zero deflection and provide static stability plus rate damping through local flow. Movable surfaces provide control authority.

**Parameter Policy**

Only parameters that can be read directly from repository-managed files will be auto-populated now. Anything that cannot be read directly will be represented explicitly as a required placeholder. No geometric or aerodynamic estimation will be added in this pass.

Directly readable inputs include:

- joint origin
- joint axis
- joint limits
- link inertial mass and inertia
- link mesh path
- link COM in link frame

Potentially derivable from mesh with deterministic geometry processing:

- projected control-surface area for links that exist as separate meshes

Not directly readable from the current files and therefore left as placeholders:

- fixed horizontal-tail area split from `base_link`
- fixed vertical-tail area split from `base_link`
- aerodynamic center location
- mean aerodynamic chord
- span and aspect ratio used for aero derivatives
- installation incidence
- lift-curve slope and drag-polar coefficients
- control-surface effectiveness

**File Strategy**

The implementation will touch three layers:

1. Asset layer
   - add a repository-local normalized URDF asset for the latest robot
   - update `FlappingBotCfg`

2. Environment/control layer
   - update the straight-flight environment to resolve and command the rudder
   - update the older direct environment so it also accepts the new robot without joint-resolution failures

3. Aerodynamics/metadata layer
   - introduce a tail-geometry loader that reads direct parameters from URDF and mesh files
   - update the tail aero config structure so explicit placeholders exist for unreadable parameters

**Verification**

Verification should cover:

- task registration still works
- the new asset resolves all expected joints
- straight-flight action mapping writes the rudder and both elevons correctly
- direct file-derived tail metadata loads without guessing
- existing pure-Python tail aero tests continue to run

**Non-Goals For This Pass**

- tuning aerodynamic coefficients
- estimating hidden fixed-tail geometry from CAD by heuristics
- re-trimming PX4-like gains
- preserving compatibility with old checkpoints
