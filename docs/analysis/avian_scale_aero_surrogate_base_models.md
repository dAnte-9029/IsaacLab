# Avian-Scale Aerodynamic Base Models for Flapping-Wing Simulation Surrogates

## 1. Executive Summary

- The strongest non-DeLaurier/non-Wang candidates are free-wake UVLM, unsteady lifting-line/nonlinear lifting-line, unsteady potential-flow panel/VLM hybrids, Volterra or state-space reduced-order models, and dynamic-stall-augmented section models.
- For a 1.6 m wingspan ornithopter in forward flight, free-wake UVLM is the best first offline teacher because it naturally represents finite-span circulation, wake memory, flapping kinematics, body motion, and force/moment outputs without requiring full CFD.
- The most directly bird-scale paper found is Yang et al. (2025), which couples a modified UVLM with multi-flexible-body dynamics for a falcon-inspired ornithopter in forward flight. It is a high-priority read because it is close to the project scale and problem setting.
- Unsteady lifting-line theory, especially Phlips, East, and Pratt (1981), is attractive as a lower-cost online physics prior because it was explicitly formulated for bird forward flight and maps naturally from span/chord/pose/body velocity to sectional circulation and total forces.
- Volterra and state-space ROMs are attractive for surrogate/teacher design because they expose aerodynamic memory states and input-output operators, but they need training data from experiments, UVLM, panel methods, or CFD.
- Dynamic stall models are important for large-amplitude bird-scale flapping at moderate/high Reynolds number, especially during high angle-of-attack stroke reversal and aggressive maneuvering, but they are not a complete 3-D wing model by themselves.
- Unsteady panel methods provide a more detailed potential-flow teacher than lifting-line and can include deforming geometry/free wake, but implementation and runtime are higher than UVLM or lifting-line for large dataset generation.
- Insect hover QSM, fruit-fly/hawkmoth-specific models, and very-low-Reynolds-number clap-and-fling models should be downgraded. Their coefficients, leading-edge-vortex assumptions, and hover-centric validation do not transfer cleanly to a 1.6 m forward-flight ornithopter.
- Recommended offline-to-online path: implement a clean aerodynamic model interface, generate offline labels with UVLM/free-wake and a lighter ULLT or dynamic-stall baseline, then train surrogates to predict full body-frame wrench, aerodynamic power, and optionally latent circulation/wake states.

## 2. Problem Setting and Scale

This report targets a bird-scale flapping-wing vehicle/ornithopter with approximate wingspan `b ~= 1.6 m`, forward-flight operation, and Reynolds number substantially above insect-scale FWMAV regimes. The intended use is not high-fidelity CFD. The model should be usable inside simulation loops, offline teacher rollouts, batch dataset generation, or as a physics prior for a learned surrogate.

The current project already considers DeLaurier-style modified strip theory and Wang et al. (2016) quasi-steady force/torque modeling. The goal here is therefore to identify fundamentally different aerodynamic bases: wake-resolving potential-flow methods, circulation-based lifting-line methods, state-space/indicial models, dynamic stall models, and data-augmented reduced-order models.

Bird-scale forward flight changes priorities:

- Forward speed and body pitch matter as much as stroke kinematics.
- Finite-span effects, wake convection, downwash, and body/wing coupling matter.
- High-amplitude unsteady effects are present, but insect-hover-specific quasi-steady assumptions are not the main line.
- Airfoil polars, stall behavior, reduced frequency, wing flexibility, and inertial/aeroelastic coupling become practical modeling parameters.

## 3. Screening Criteria

Candidate models were screened against:

- **Avian-scale relevance**: evidence for bird-scale, ornithopter, flapping-wing aircraft, or at least finite-wing forward-flight use.
- **Free-flight / forward-flight support**: ability to handle nonzero forward speed, body orientation, and body rates rather than hover only.
- **Computability**: explicit formulas, algorithms, or implementable time stepping.
- **Observable simulation inputs**: wing geometry, span/chord distribution, wing pose, generalized coordinates `q`, `qdot`, `qddot`, body velocity, body angular velocity, local angle of attack, reduced frequency.
- **Outputs**: body-frame force, body-frame moment, lift, drag, thrust, aerodynamic power, pressure/circulation/wake states where available.
- **Runtime cost**: suitability for online rollout, offline teacher rollout, or batch label generation.
- **Parameter identifiability**: geometry, airfoil polars, stall parameters, wake parameters, ROM coefficients.
- **Flexible-wing extensibility**: direct support or clear coupling path to structural modes or multi-body wing kinematics.
- **Surrogate-teacher suitability**: whether the model can generate consistent labels and expose useful latent states.

Searches emphasized original papers, review papers, and simulation/control-oriented work with formulas, algorithms, validation examples, or open implementations.

## 4. Candidate Model Families

### 4.1 Unsteady Vortex Lattice Method (UVLM) / Free-Wake UVLM

**Representative papers**

- Murua, Palacios, and Graham (2012), "Applications of the unsteady vortex-lattice method in aircraft aeroelasticity and flight dynamics", Progress in Aerospace Sciences. DOI: `10.1016/j.paerosci.2012.06.001`.
- Yang et al. (2025), "Numerical simulation framework of bird-inspired ornithopter in forward flight using modified Unsteady Vortex Lattice Method coupled with Multi-Flexible-Body Dynamics", Journal of Fluids and Structures. DOI: `10.1016/j.jfluidstructs.2024.104263`.
- Fritz and Long (2005), "Object-oriented unsteady vortex lattice method for flapping flight", Journal of Aircraft. DOI: `10.2514/1.7357`.
- Roccia et al. (2013), "Modified Unsteady Vortex-Lattice Method to Study Flapping Wings in Hover Flight", AIAA Journal. DOI: `10.2514/1.J052262`.
- Vest and Katz (1996), "Unsteady Aerodynamic Model of Flapping Wings", AIAA Journal. DOI: `10.2514/3.13250`.
- Ptera Software, an open-source Python flapping-wing simulator using an unsteady ring-vortex lattice method. URL: <https://github.com/camUrban/PteraSoftware>.

**Model idea**

The wing is discretized into vortex-lattice panels. Bound vortex strengths are solved from no-penetration boundary conditions. A wake is shed from the trailing edge, convected with the local velocity field, and contributes induced velocity back onto the wing. Forces and moments are computed from pressure differences, impulse, or unsteady Bernoulli/Kutta-Joukowski style panel loads.

**Required inputs**

- Wing planform, chord/span distribution, twist, camber approximation, panel mesh.
- Time history of body pose, body velocity, body angular velocity.
- Wing joint/shape coordinates `q`, `qdot`, sometimes `qddot`.
- Air density and optionally airfoil polar corrections or separated-flow corrections.
- Wake initialization and convection settings.

**Predicted outputs**

- Body-frame and wing-frame aerodynamic force and moment.
- Lift, drag, thrust, side force.
- Aerodynamic power from generalized aerodynamic forces and wing rates.
- Bound circulation and wake geometry/states, useful as surrogate latent labels.

**Suitability for bird-scale ornithopter**

High. UVLM is a natural finite-wing forward-flight model and has been applied to flapping flight, aeroelasticity, flight dynamics, and bird-inspired ornithopters. Yang et al. (2025) is especially relevant because it combines modified UVLM with multi-flexible-body dynamics for a falcon-inspired ornithopter.

**Strengths**

- Fundamentally different from strip/QSM models because wake memory and finite-span induced effects are solved explicitly.
- Produces rich labels: full wrench, power, circulation, wake state.
- Good offline teacher candidate for dataset generation and surrogate training.
- Supports forward flight and body motion naturally.
- Can couple to flexible/multi-body wings.

**Weaknesses**

- Potential-flow baseline cannot handle massive separation or deep stall unless augmented.
- Free-wake rollup can become numerically expensive and sensitive.
- Requires mesh/time-step choices and wake stabilization.
- Online use in RL may be too slow unless reduced or surrogate-accelerated.

**Implementation difficulty**

Medium to high. A minimal fixed-wake or relaxed-wake UVLM is feasible, but robust free-wake and flexible-wing coupling require careful engineering.

**Runtime estimate**

- Offline teacher: good, especially with moderate panel count and wake truncation.
- Online rollout: possible only with coarse meshes or surrogate acceleration.
- Batch generation: good if vectorized/parallelized and wake length is bounded.

**Suitability as surrogate teacher**

Very high. It can generate direct wrench/power labels and intermediate circulation/wake labels.

**Can replace DeLaurier/Wang as a fundamentally different base model?**

Yes. UVLM replaces local quasi-steady section forces with a circulation/wake-resolving unsteady finite-wing model.

### 4.2 Unsteady Lifting-Line / Nonlinear Lifting-Line Models

**Representative papers**

- Phlips, East, and Pratt (1981), "An unsteady lifting line theory of flapping wings with application to the forward flight of birds", Journal of Fluid Mechanics. DOI: `10.1017/S0022112081000311`.
- Ramesh et al. (2014), "An unsteady airfoil theory applied to pitching and surging motions", Theoretical and Computational Fluid Dynamics. DOI: `10.1007/s00162-014-0332-0`.
- Bird-flight and finite-wing extensions discussed in Platzer et al. (2008), Shyy et al. (2013), and bird-flight engineering reviews.

**Model idea**

The wing is represented by a spanwise bound circulation distribution. The model solves for circulation using local effective angle of attack plus induced downwash from trailing vortices. Unsteady variants include wake memory, apparent-mass terms, or indicial response. Nonlinear variants add airfoil polars, stall-limited lift curves, or iterative induced-velocity corrections.

**Required inputs**

- Spanwise stations, chord, twist, local airfoil polar data.
- Local wing translational and rotational velocity from body and wing kinematics.
- Body velocity and angular velocity.
- Wake/induced velocity model, or state variables for unsteady downwash.
- Optional stall/dynamic-stall parameters.

**Predicted outputs**

- Sectional lift/drag/moment.
- Integrated body-frame force and moment.
- Induced power, profile power, aerodynamic power.
- Spanwise circulation and induced velocity states.

**Suitability for bird-scale ornithopter**

High. Phlips et al. explicitly targeted bird forward flight. Lifting-line is less detailed than UVLM but aligns well with 1.6 m finite wings in forward flight and is more scalable for online simulation.

**Strengths**

- Much cheaper than UVLM or panel methods.
- Better finite-wing basis than pure strip theory because induced velocity and spanwise circulation are solved globally.
- Natural interface for airfoil polar and stall corrections.
- Suitable as an online physics prior or lightweight teacher.

**Weaknesses**

- Less accurate for low aspect ratio, strong sweep, strong wing-wake interaction, or complex 3-D separation.
- Wake treatment is approximate unless upgraded to free-wake lifting-line.
- Dynamic stall and leading-edge-vortex effects need extra models.

**Implementation difficulty**

Medium. A Prandtl/unsteady lifting-line solver with polar lookup and wake-state update is implementable in Python and can later be accelerated.

**Runtime estimate**

- Offline teacher: excellent.
- Online rollout: good.
- Batch generation: excellent.

**Suitability as surrogate teacher**

High. It can supply cheap labels and interpretable latent states such as circulation and induced velocity.

**Can replace DeLaurier/Wang as a fundamentally different base model?**

Yes. It uses a global circulation/induced-flow solve rather than independent quasi-steady blade elements.

### 4.3 Discrete Vortex / Vortex Particle / Reduced Wake Models

**Representative papers**

- Shukla and Eldredge (2007), "An inviscid model for vortex shedding from a deforming body", Theoretical and Computational Fluid Dynamics. DOI: `10.1007/s00162-007-0041-2`.
- Brunton and Rowley (2010), "Empirical state-space representations for Theodorsen's lift model", Journal of Fluids and Structures. DOI: `10.1016/j.jfluidstructs.2009.11.005`.
- Kumar et al. (2025), "Aerodynamic performance and flow mechanism of 3D flapping wing using discrete vortex method", Journal of Fluids and Structures. DOI: `10.1016/j.jfluidstructs.2024.104125`.
- Willis, Persson, Peraire, and Breuer (2005/2007 conference and journal work on vortex-particle/flapping-wing simulations). DOI not verified.

**Model idea**

The wake is represented by discrete vortices, vortex blobs, particles, or reduced vortex elements shed from wing edges. The method tracks vorticity convection and induced velocities more explicitly than strip theory while using far fewer degrees of freedom than CFD.

**Required inputs**

- Wing/body geometry or edge geometry for vortex shedding.
- Time-varying kinematics and body velocities.
- Shedding rules, core size, vortex strength update, wake truncation/merging rules.
- Optional viscous diffusion or empirical separation parameters.

**Predicted outputs**

- Unsteady force/moment from impulse, pressure integration, or vortex-induced loads.
- Wake topology, vortex strengths, induced velocities.
- Power estimates from aerodynamic generalized forces.

**Suitability for bird-scale ornithopter**

Medium. The method family is physically relevant for wake memory and vortex effects, but many canonical examples are 2-D foils or smaller flapping wings. 3-D bird-scale implementation is possible but more involved than UVLM/lifting-line.

**Strengths**

- Captures wake memory and vortex-induced forces more directly than quasi-steady methods.
- Can expose compact wake latent states for surrogate learning.
- Can bridge between potential-flow teacher and CFD-like wake structure.

**Weaknesses**

- Robust vortex shedding and force evaluation are nontrivial.
- 3-D vortex particle methods can become expensive.
- Parameter choices such as vortex core, merging, and diffusion affect stability.

**Implementation difficulty**

High for a robust 3-D bird-scale implementation; medium for a 2-D/sectional reduced model.

**Runtime estimate**

- Offline teacher: moderate to high cost.
- Online rollout: usually too expensive unless heavily reduced.
- Batch generation: possible with bounded wake size and parallelization.

**Suitability as surrogate teacher**

Medium to high. Strong for wake-state learning, less attractive as the first implementation because UVLM is more established for finite wings.

**Can replace DeLaurier/Wang as a fundamentally different base model?**

Yes, especially if the model carries discrete wake states and computes forces from vortex dynamics rather than local force coefficients.

### 4.4 Unsteady Panel Methods

**Representative papers**

- Smith, Wilkin, and Williams (1996), "The advantages of an unsteady panel method in modelling the aerodynamic forces on rigid flapping wings", Journal of Experimental Biology. DOI: `10.1242/jeb.199.5.1073`.
- Vest and Katz (1996), "Unsteady Aerodynamic Model of Flapping Wings", AIAA Journal. DOI: `10.2514/3.13250`.
- Katz and Plotkin (2001), "Low-Speed Aerodynamics", Cambridge University Press. DOI not verified.
- Panel/free-wake methods also appear in flapping-wing and rotorcraft literature, but bird-scale ornithopter implementations are less common than UVLM.

**Model idea**

The wing surface is discretized into source/doublet/vortex panels. Boundary conditions solve the potential flow around a moving body, and a wake is shed to satisfy the Kutta condition. Compared with UVLM, panel methods can represent thickness and pressure distribution more directly.

**Required inputs**

- Surface mesh, wing shape, and deformation.
- Body and wing kinematics.
- Wake shedding/convection rules.
- Air density; optional separation/stall correction.

**Predicted outputs**

- Surface pressure, integrated aerodynamic force and moment.
- Lift/drag/thrust/power.
- Wake geometry and potential/circulation states.

**Suitability for bird-scale ornithopter**

Medium to high. Physics are relevant to bird-scale forward flight, but many published flapping panel validations are generic or smaller-scale. The method is useful as an offline teacher if implementation effort is acceptable.

**Strengths**

- More geometric fidelity than lifting-line or standard UVLM.
- Can produce pressure-like labels useful for spatial surrogate training.
- Good for deforming wings if mesh motion is managed.

**Weaknesses**

- Still potential flow, so separated flow and stall need corrections.
- More complex and expensive than UVLM.
- Wake treatment can dominate implementation complexity.

**Implementation difficulty**

High.

**Runtime estimate**

- Offline teacher: possible but heavier than UVLM.
- Online rollout: unlikely without surrogate reduction.
- Batch generation: moderate if coarse meshes and short wakes are used.

**Suitability as surrogate teacher**

Medium to high. Strong labels, but high engineering cost.

**Can replace DeLaurier/Wang as a fundamentally different base model?**

Yes. It is a field/pressure/wake method rather than a sectional quasi-steady model.

### 4.5 Indicial / State-Space Unsteady Aerodynamic Models

**Representative papers**

- Theodorsen (1935), "General Theory of Aerodynamic Instability and the Mechanism of Flutter", NACA Report 496. URL: <https://ntrs.nasa.gov/citations/19930090935>.
- Wagner (1925), "Uber die Entstehung des dynamischen Auftriebes von Tragflugeln". DOI not verified.
- Peters, Karunamoorthy, and Cao (1995), "Finite state induced flow models. Part I: Two-dimensional thin airfoil", Journal of Aircraft. DOI not verified.
- Brunton and Rowley (2010), "Empirical state-space representations for Theodorsen's lift model", Journal of Fluids and Structures. DOI: `10.1016/j.jfluidstructs.2009.11.005`.
- Taha, Hajj, and Beran (2014), "State-space representation of the unsteady aerodynamics of flapping flight", Aerospace Science and Technology. DOI: `10.1016/j.ast.2014.01.011`.

**Model idea**

Unsteady airloads are represented by linear or weakly nonlinear state-space filters approximating Wagner, Kussner, Theodorsen, or finite-state induced-flow responses. Inputs are effective angle of attack, plunge/pitch/surge motion, gusts, and their rates. Outputs are lift, moment, and sometimes induced-flow states.

**Required inputs**

- Local section kinematics: angle of attack, pitch/plunge/surge rates, reduced frequency.
- Chord, air density, local flow speed.
- State-space coefficients or indicial approximations.
- 3-D correction or lifting-line integration for finite wings.

**Predicted outputs**

- Sectional lift and pitching moment.
- Added-mass/circulatory decomposition.
- Aerodynamic memory states.
- Integrated wing/body force and moment after spanwise assembly.

**Suitability for bird-scale ornithopter**

Medium. The underlying unsteady airfoil theory is not insect-scale and is relevant at bird-scale Reynolds numbers, but it is fundamentally 2-D/linear unless extended with nonlinear polars, dynamic stall, and finite-wing corrections.

**Strengths**

- Very cheap and differentiable.
- Provides interpretable aerodynamic memory states.
- Excellent candidate as a neural surrogate architecture prior.
- Works well as a component inside lifting-line or stripwise finite-wing models.

**Weaknesses**

- Pure Theodorsen/Wagner theory assumes small disturbances, attached flow, and harmonic/linear response.
- Needs 3-D and nonlinear corrections for real ornithopter maneuvers.
- Alone, it is not a complete bird-scale finite-wing model.

**Implementation difficulty**

Low to medium.

**Runtime estimate**

- Offline teacher: excellent.
- Online rollout: excellent.
- Batch generation: excellent.

**Suitability as surrogate teacher**

Medium as a standalone teacher; high as a physics prior or latent-state component.

**Can replace DeLaurier/Wang as a fundamentally different base model?**

Partly. It is fundamentally unsteady/memory-based, but should be embedded in a finite-wing framework to be a full replacement.

### 4.6 Dynamic Stall Models for Flapping Wings

**Representative papers**

- Leishman and Beddoes (1989), "A semi-empirical model for dynamic stall", Journal of the American Helicopter Society. DOI not verified.
- Leishman and Beddoes (1989), "State-space model for unsteady airfoil behavior and dynamic stall", AIAA paper. DOI not verified.
- Beddoes (1983), "Representation of airfoil behavior", Vertica. DOI not verified.
- Sheng, Galbraith, and Coton (2008), "A modified dynamic stall model for low Mach numbers", Journal of Solar Energy Engineering. DOI not verified.
- Ramesh et al. (2014) and other unsteady airfoil models provide alternatives for leading-edge vortex onset and nonlinear lift in large-amplitude motion.

**Model idea**

Dynamic stall models augment airfoil section aerodynamics with lag states for attached-flow circulation, trailing-edge separation, leading-edge vortex lift, vortex convection, and reattachment. They are typically driven by angle of attack, pitch rate, Mach/Reynolds settings, and empirical stall parameters.

**Required inputs**

- Sectional angle of attack and rate, local flow speed, chord, reduced frequency.
- Static airfoil polars and stall angles.
- Dynamic stall time constants and vortex/separation parameters.
- Spanwise integration model, usually strip/lifting-line/UVLM hybrid.

**Predicted outputs**

- Sectional lift, drag, and pitching moment including hysteresis.
- Integrated body-frame force/moment when combined with a finite-wing model.
- Aerodynamic power via generalized force dot velocity.

**Suitability for bird-scale ornithopter**

Medium to high as an augmentation. Bird-scale Reynolds numbers and high-amplitude strokes can enter dynamic stall, especially near stroke reversal or aggressive maneuvers. The model is not a complete 3-D wing model by itself.

**Strengths**

- Captures stall hysteresis and nonlinear airfoil behavior ignored by pure UVLM/lifting-line.
- Cheap enough for online rollout.
- Parameters can be identified from airfoil polars, wind-tunnel data, or UVLM/CFD residuals.

**Weaknesses**

- Semi-empirical and parameter-sensitive.
- Rotorcraft/helicopter heritage may not directly match deforming flapping wings.
- Requires coupling to a finite-wing induced-flow model.

**Implementation difficulty**

Medium if using a standard Leishman-Beddoes style section model; high if calibrating robustly for custom flexible wings.

**Runtime estimate**

- Offline teacher: excellent as a correction layer.
- Online rollout: excellent.
- Batch generation: excellent.

**Suitability as surrogate teacher**

Medium as standalone; high as a residual/augmentation target over lifting-line or UVLM.

**Can replace DeLaurier/Wang as a fundamentally different base model?**

Not alone. It can be part of a fundamentally different base if coupled with lifting-line, UVLM, or state-space unsteady aerodynamics.

### 4.7 Reduced-Order / Data-Augmented Physics Models

**Representative papers**

- Ruiz, Acosta, and Ollero (2022), "Aerodynamic reduced-order Volterra model of an ornithopter under high-amplitude flapping", Aerospace Science and Technology. DOI: `10.1016/j.ast.2022.107331`.
- Taha, Hajj, and Beran (2014), "State-space representation of the unsteady aerodynamics of flapping flight", Aerospace Science and Technology. DOI: `10.1016/j.ast.2014.01.011`.
- Brunton and Rowley (2010), "Empirical state-space representations for Theodorsen's lift model", Journal of Fluids and Structures. DOI: `10.1016/j.jfluidstructs.2009.11.005`.
- Stanford and Beran (2010), "Analytical Sensitivity Analysis of an Unsteady Vortex-Lattice Method for Flapping-Wing Optimization", Journal of Aircraft. DOI not verified.

**Model idea**

The aerodynamic map is represented by reduced operators: Volterra kernels, linear/nonlinear state-space systems, identified latent dynamics, Gaussian process residuals, or neural surrogates constrained by physics features. Inputs are kinematic histories and body velocities; outputs are force, moment, power, or latent circulation states.

**Required inputs**

- Time histories of wing kinematics and body velocities.
- Geometry descriptors and possibly modal/flexible coordinates.
- Training labels from experiments, UVLM, panel methods, CFD, or flight data.
- Choice of memory window or recurrent/state-space architecture.

**Predicted outputs**

- Full aerodynamic wrench.
- Lift/drag/thrust and aerodynamic power.
- Optional residual over a low-order physics prior.
- Optional latent circulation/wake states.

**Suitability for bird-scale ornithopter**

High if trained on bird-scale data. Ruiz et al. is directly relevant because it models an ornithopter under high-amplitude flapping with a Volterra ROM.

**Strengths**

- Directly aligned with surrogate/teacher goals.
- Can be fast enough for online rollout.
- Can encode unsteady memory without carrying a large wake mesh.
- Can fuse UVLM labels, wind-tunnel data, and flight logs.

**Weaknesses**

- Generalization is only as good as the training envelope.
- Needs careful excitation design and validation.
- Pure data models may violate physical trends outside distribution.

**Implementation difficulty**

Medium. A simple Volterra/NARX/RNN surrogate is straightforward; robust physics-constrained deployment requires careful dataset and validation design.

**Runtime estimate**

- Offline teacher: not a first-principles teacher unless trained from one.
- Online rollout: excellent.
- Batch generation: excellent after training.

**Suitability as surrogate teacher**

High as a deployable surrogate; medium as an original teacher unless labels come from experiments or higher-fidelity models.

**Can replace DeLaurier/Wang as a fundamentally different base model?**

Yes if the model is trained from UVLM/panel/flight data and uses memory-state dynamics rather than a quasi-steady strip law.

### 4.8 Other Bird-Scale Ornithopter Models Found in Search

**Representative papers**

- Nekoo et al. (2025), "Recent advances in dynamics and control of ornithopters: A review", International Journal of Robotics Research. DOI: `10.1177/02783649251343638`.
- Mateos et al. (2022), "A simplified model for forward-flight transitions of a bio-inspired unmanned aerial vehicle", Aerospace. DOI: `10.3390/aerospace9100617`.
- Hedenstrom and Johansson (2015), "Bat flight: aerodynamics, kinematics and flight morphology", Journal of Experimental Biology. DOI: `10.1242/jeb.031203` (bat rather than bird, but useful scaling context).
- Iosilevskii (2014), "Forward flight of birds revisited. Part 1: aerodynamics and performance", Royal Society Open Science. DOI: `10.1098/rsos.140248`.

**Model idea**

These works include control-oriented simplified models, bird-flight performance models, and reviews. They are useful for scale assumptions, power trends, trim checks, and dynamics integration, but they are often not sufficient as standalone aerodynamic teachers.

**Suitability**

Medium as context and validation references. Use them to define plausible flight speeds, power curves, trim behavior, and control regimes.

**Can replace DeLaurier/Wang?**

Usually no as standalone aerodynamic bases, but they can guide validation and envelope design.

## 5. Comparison Table

| Model family | Representative references | Bird-scale relevance | Forward-flight support | Force output | Moment output | Power output | Wake/unsteady effects | Stall/separation handling | Flexible-wing extensibility | Runtime cost | Implementation difficulty | Surrogate-teacher priority |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| Free-wake UVLM | Murua 2012; Yang 2025; Fritz & Long 2005; Roccia 2013 | High | High | Yes | Yes | Yes | High | Low unless augmented | High | Medium-high | Medium-high | Very high |
| Unsteady/nonlinear lifting-line | Phlips 1981; finite-wing ULLT literature | High | High | Yes | Yes | Yes | Medium | Medium with polar/stall augmentation | Medium | Low-medium | Medium | High |
| Discrete vortex/vortex particle | Shukla & Eldredge 2007; Kumar 2025; Willis/Peraire work | Medium | Medium-high | Yes | Yes | Yes | High | Medium, model-dependent | Medium | Medium-high | High | Medium-high |
| Unsteady panel methods | Smith 1996; Vest & Katz 1996; Katz & Plotkin | Medium-high | High | Yes | Yes | Yes | High | Low unless augmented | Medium-high | High | High | Medium-high |
| Indicial/state-space unsteady aero | Theodorsen 1935; Brunton 2010; Taha 2014 | Medium | Medium | Yes | Yes | Yes | Medium | Low unless nonlinearized | Medium | Very low | Low-medium | Medium as teacher, high as prior |
| Dynamic stall models | Leishman-Beddoes; Beddoes; Sheng 2008 | Medium-high as augmentation | High | Yes, sectional | Yes, sectional | Yes | Medium | High | Medium | Very low | Medium | Medium-high as correction |
| Volterra/data-augmented ROM | Ruiz 2022; Taha 2014; Brunton 2010 | High if trained at scale | High | Yes | Yes | Yes | Learned memory | Learned or residual | High if features include flex | Very low after training | Medium | Very high as deployable surrogate |
| Momentum/actuator/wake oscillator | Iosilevskii 2014; bird-flight performance models | Medium | High for performance | Usually net force/power | Usually weak | Yes | Low-medium | Low | Low | Very low | Low | Low-medium |

## 6. Recommended Shortlist

### Priority 1: Free-Wake UVLM with Optional Separated-Flow Correction

**Why suitable for 1.6 m bird-scale flapping wing**

It is finite-wing, forward-flight capable, and already used in aircraft aeroelasticity and bird-inspired ornithopter simulation. Yang et al. (2025) makes this the closest direct match.

**New information relative to DeLaurier/Wang**

It supplies bound circulation, induced velocity, wake memory, wake geometry, and finite-span downwash, not just instantaneous local section coefficients.

**Minimum implementation**

- Discretize each wing into ring-vortex panels.
- Solve no-penetration at each time step.
- Shed trailing-edge wake panels and convect a truncated wake.
- Compute body-frame wrench and power.
- Add optional polar/dynamic-stall residual for separated regimes.

**Needed parameters/data**

Wing mesh, kinematics, air density, wake time step, wake truncation length, optional airfoil polar/stall parameters.

**Largest risk**

Potential-flow loads can be wrong in separated high-AoA regions unless augmented.

### Priority 2: Unsteady/Nonlinear Lifting-Line with Airfoil Polars and Wake States

**Why suitable**

It is directly connected to bird forward flight through Phlips et al. (1981), cheaper than UVLM, and natural for a 1.6 m finite wing.

**New information relative to DeLaurier/Wang**

It solves spanwise circulation and induced velocity globally instead of treating wing strips independently.

**Minimum implementation**

- Spanwise stations with chord/twist/airfoil polar lookup.
- Iterative induced velocity/circulation solve.
- Unsteady wake or lag states.
- Integrate sectional force/moment/power into body frame.

**Needed parameters/data**

Span/chord/twist, static airfoil polars, optional stall/dynamic-stall parameters, wake-state coefficients.

**Largest risk**

Accuracy may degrade for strong 3-D separation, low aspect ratio, or complex wing-wake interaction.

### Priority 3: Volterra / State-Space ROM Trained from UVLM or Experiments

**Why suitable**

It is designed for high-amplitude ornithopter flapping and can run online at RL speeds once trained.

**New information relative to DeLaurier/Wang**

It explicitly learns aerodynamic memory and nonlinear input-output history, not just instantaneous force laws.

**Minimum implementation**

- Generate excitation trajectories over airspeed, frequency, amplitude, pitch, and body rates.
- Fit Volterra/NARX/state-space/RNN model to wrench and power.
- Optionally include UVLM circulation/wake latent labels.

**Needed parameters/data**

Training dataset, memory window, geometry descriptors, validation maneuvers, regularization.

**Largest risk**

Out-of-distribution extrapolation.

### Priority 4: Dynamic-Stall-Augmented Lifting-Line

**Why suitable**

Bird-scale flapping can enter dynamic stall during high-AoA segments. Dynamic stall models add the nonlinear hysteresis missing from potential-flow UVLM/ULLT.

**New information relative to DeLaurier/Wang**

It introduces lagged separation and vortex-lift states rather than static lift/drag coefficients.

**Minimum implementation**

- Implement a section-level Leishman-Beddoes or similar lag model.
- Couple it to nonlinear lifting-line induced velocity.
- Integrate forces/moments/power over the wing.

**Needed parameters/data**

Airfoil static polars, stall angles, lag constants, vortex/separation parameters, Reynolds range.

**Largest risk**

Parameter identification for deforming flapping wings.

### Priority 5: Unsteady Panel / VLM-Panel Hybrid Teacher

**Why suitable**

It can provide pressure-like labels and geometry fidelity beyond lifting-line, useful if wing thickness/deformation matters.

**New information relative to DeLaurier/Wang**

It computes a potential-flow field over a moving/deforming surface and wake.

**Minimum implementation**

Use or adapt an existing panel/free-wake solver for flapping wing meshes; start with rigid wings and add deformation later.

**Needed parameters/data**

Surface mesh, wake scheme, time step, solver tolerances, optional separation model.

**Largest risk**

High implementation cost for only moderate benefit over UVLM in early surrogate development.

## 7. Proposed Simulation/Surrogate Integration Path

This section outlines a new route rather than a coexistence patch around DeLaurier/Wang.

### Unified Model Interface

Recommended Python-side conceptual interface:

```python
class AvianAeroModel:
    def reset(self, geometry, initial_state, options): ...
    def step(self, state, action, geometry, dt) -> AeroOutput: ...
```

### Input Fields

- `body_pose_w`: position and orientation.
- `body_lin_vel_b`, `body_ang_vel_b`.
- `wing_q`, `wing_qdot`, `wing_qddot`: generalized wing/flapping/flex coordinates.
- `wing_pose_b`, `wing_twist`, `wing_surface_points`.
- `span_stations`, `chord`, `twist`, `airfoil_id`.
- `air_density`, `viscosity`, `wind_w`.
- Optional `aero_internal_state`: circulation, wake, indicial states, dynamic-stall states.

### Output Label Fields

- `force_b`: total body-frame aerodynamic force.
- `moment_b`: total body-frame aerodynamic moment about body origin or center of mass.
- `force_wing_sections_b`: optional sectional loads.
- `lift_drag_thrust`: diagnostic decomposition.
- `aero_power`: aerodynamic power.
- `circulation`: bound circulation or lifting-line gamma.
- `wake_state`: truncated wake coordinates/vortex strengths or learned latent target.
- `stall_state`: optional dynamic-stall/separation states.

### Offline Dataset Generation

1. Sample flight envelope: airspeed, body pitch, flapping frequency, amplitude, stroke-plane angle, wing twist, and phase.
2. Generate deterministic kinematic trajectories and short free-flight rollout snippets.
3. Run UVLM/free-wake teacher with bounded wake length.
4. Run a cheaper ULLT/dynamic-stall baseline for residual labels.
5. Store state/action/geometry histories and labels in chunked arrays, for example Zarr or HDF5.
6. Split by maneuvers, not by random time samples, to test generalization.

### Surrogate Targets

- Full aerodynamic wrench `(Fx, Fy, Fz, Mx, My, Mz)`.
- Residual over a simple circulation/lifting-line baseline.
- Circulation or induced-velocity latent state.
- Truncated wake latent state.
- Aerodynamic power.
- Sectional force distribution if control allocation or wing structural feedback is needed.

### Validation Tests

- **Trim consistency**: predicted mean lift balances weight at plausible airspeeds and pitch.
- **Force/power trends**: lift, thrust, and power vary monotonically/plausibly with frequency, amplitude, airspeed, and pitch.
- **Trajectory rollout**: closed-loop free-flight rollouts remain stable over multi-wingbeat horizons.
- **Sensitivity checks**: sweep frequency, amplitude, airspeed, body pitch, and stroke-plane angle.
- **Energy checks**: positive mean aerodynamic power for powered flapping except gliding/autorotation cases.
- **Wake/circulation checks**: bound circulation and downwash remain smooth under small kinematic perturbations.

### Suggested IsaacLab Module Locations

No code changes are made here, but a clean project placement would be:

- `source/flapping_bot/flapping_bot/aerodynamics/` for model implementations.
- `source/flapping_bot/flapping_bot/aerodynamics/base.py` for common interfaces and dataclasses.
- `source/flapping_bot/flapping_bot/aerodynamics/uvlm/` for teacher generation.
- `source/flapping_bot/flapping_bot/aerodynamics/lifting_line.py` for lightweight online model.
- `source/flapping_bot/flapping_bot/aerodynamics/surrogate.py` for learned models.
- `scripts/flapping_bot/generate_aero_dataset.py` for offline label generation.
- `docs/analysis/` for validation reports.

## 8. Low-Priority / Excluded Models

- **Insect-hover QSM models**: downgraded because validation often targets fruit fly, hawkmoth, or centimeter-scale hover, with Reynolds number and wing-wake regimes far from a 1.6 m forward-flight ornithopter.
- **Clap-and-fling dominated models**: low priority unless the bird-scale platform actually uses wing-wing interaction near dorsal stroke reversal.
- **Pure actuator-disk or momentum models**: useful for rough performance scaling and power sanity checks, but too low-dimensional for full body-frame moment and maneuvering labels.
- **Hover-only UVLM/QSM variants**: useful algorithmically, but must be revalidated for forward-flight body motion and wake convection.
- **Pure Theodorsen/Wagner section models without finite-wing coupling**: useful as priors, but not complete enough as a full ornithopter teacher.
- **Full CFD as primary teacher**: high value for spot validation, but too expensive for the requested low/mid-order simulation loop.
- **Pure black-box neural surrogates without physics inputs or coverage guarantees**: fast, but risky outside the training envelope.

## 9. Must-Read Papers

1. **Yang et al. (2025)**, "Numerical simulation framework of bird-inspired ornithopter in forward flight using modified Unsteady Vortex Lattice Method coupled with Multi-Flexible-Body Dynamics". DOI: `10.1016/j.jfluidstructs.2024.104263`.  
   Why it matters: closest direct match to bird-inspired ornithopter forward flight with flexible-body coupling. Family: UVLM/free-wake.

2. **Phlips, East, and Pratt (1981)**, "An unsteady lifting line theory of flapping wings with application to the forward flight of birds". DOI: `10.1017/S0022112081000311`.  
   Why it matters: explicit bird forward-flight lifting-line basis. Family: unsteady lifting-line.

3. **Murua, Palacios, and Graham (2012)**, "Applications of the unsteady vortex-lattice method in aircraft aeroelasticity and flight dynamics". DOI: `10.1016/j.paerosci.2012.06.001`.  
   Why it matters: UVLM review and implementation context for aeroelastic/flight-dynamics coupling. Family: UVLM.

4. **Ruiz, Acosta, and Ollero (2022)**, "Aerodynamic reduced-order Volterra model of an ornithopter under high-amplitude flapping". DOI: `10.1016/j.ast.2022.107331`.  
   Why it matters: directly relevant ROM approach for high-amplitude ornithopter flapping. Family: reduced-order/data-augmented.

5. **Vest and Katz (1996)**, "Unsteady Aerodynamic Model of Flapping Wings". DOI: `10.2514/3.13250`.  
   Why it matters: classic unsteady potential-flow/panel-style model for flapping finite wings. Family: UVLM/panel.

6. **Fritz and Long (2005)**, "Object-oriented unsteady vortex lattice method for flapping flight". DOI: `10.2514/1.7357`.  
   Why it matters: implementable UVLM architecture for flapping flight. Family: UVLM.

7. **Taha, Hajj, and Beran (2014)**, "State-space representation of the unsteady aerodynamics of flapping flight". DOI: `10.1016/j.ast.2014.01.011`.  
   Why it matters: aerodynamic memory/state-space formulation well suited to surrogate priors. Family: state-space unsteady aero.

8. **Brunton and Rowley (2010)**, "Empirical state-space representations for Theodorsen's lift model". DOI: `10.1016/j.jfluidstructs.2009.11.005`.  
   Why it matters: compact state-space approximation of classical unsteady lift. Family: indicial/state-space.

9. **Smith, Wilkin, and Williams (1996)**, "The advantages of an unsteady panel method in modelling the aerodynamic forces on rigid flapping wings". DOI: `10.1242/jeb.199.5.1073`.  
   Why it matters: panel-method flapping force modeling and comparison to experiments. Family: unsteady panel.

10. **Roccia et al. (2013)**, "Modified Unsteady Vortex-Lattice Method to Study Flapping Wings in Hover Flight". DOI: `10.2514/1.J052262`.  
    Why it matters: useful modified UVLM algorithm, although hover/small-flapping validation is lower priority for this project. Family: UVLM.

11. **Kumar et al. (2025)**, "Aerodynamic performance and flow mechanism of 3D flapping wing using discrete vortex method". DOI: `10.1016/j.jfluidstructs.2024.104125`.  
    Why it matters: recent 3-D DVM route for flapping wings. Family: discrete vortex.

12. **Theodorsen (1935)**, "General Theory of Aerodynamic Instability and the Mechanism of Flutter", NACA Report 496. URL: <https://ntrs.nasa.gov/citations/19930090935>.  
    Why it matters: foundation for indicial/state-space unsteady aerodynamic priors. Family: indicial/state-space.

13. **Leishman and Beddoes (1989)**, "A semi-empirical model for dynamic stall". DOI not verified.  
    Why it matters: standard dynamic stall state model to augment section aerodynamics. Family: dynamic stall.

14. **Iosilevskii (2014)**, "Forward flight of birds revisited. Part 1: aerodynamics and performance". DOI: `10.1098/rsos.140248`.  
    Why it matters: bird forward-flight performance and scaling sanity checks. Family: bird-scale validation/performance.

## 10. References

- Brunton, S. L., & Rowley, C. W. (2010). Empirical state-space representations for Theodorsen's lift model. *Journal of Fluids and Structures*, 26(3), 406-420. DOI: `10.1016/j.jfluidstructs.2009.11.005`. BibTeX key: `brunton2010empirical`.
- Fritz, T. E., & Long, L. N. (2005). Object-oriented unsteady vortex lattice method for flapping flight. *Journal of Aircraft*, 42(6), 1491-1499. DOI: `10.2514/1.7357`. BibTeX key: `fritz2005object`.
- Hedenstrom, A., & Johansson, L. C. (2015). Bat flight: aerodynamics, kinematics and flight morphology. *Journal of Experimental Biology*, 218, 653-663. DOI: `10.1242/jeb.031203`. BibTeX key: `hedenstrom2015bat`.
- Iosilevskii, G. (2014). Forward flight of birds revisited. Part 1: aerodynamics and performance. *Royal Society Open Science*, 1, 140248. DOI: `10.1098/rsos.140248`. BibTeX key: `iosilevskii2014forward`.
- Katz, J., & Plotkin, A. (2001). *Low-Speed Aerodynamics* (2nd ed.). Cambridge University Press. DOI not verified. BibTeX key: `katz2001low`.
- Kumar, A., Khaskheli, A., Akhter, M. Z., & Nizamani, Z. A. (2025). Aerodynamic performance and flow mechanism of 3D flapping wing using discrete vortex method. *Journal of Fluids and Structures*, 132, 104125. DOI: `10.1016/j.jfluidstructs.2024.104125`. BibTeX key: `kumar2025aerodynamic`.
- Leishman, J. G., & Beddoes, T. S. (1989). A semi-empirical model for dynamic stall. *Journal of the American Helicopter Society*. DOI not verified. BibTeX key: `leishman1989semi`.
- Mateos, A., Acosta, J. A., Ruiz, A., & Ollero, A. (2022). A simplified model for forward-flight transitions of a bio-inspired unmanned aerial vehicle. *Aerospace*, 9(10), 617. DOI: `10.3390/aerospace9100617`. BibTeX key: `mateos2022simplified`.
- Murua, J., Palacios, R., & Graham, J. M. R. (2012). Applications of the unsteady vortex-lattice method in aircraft aeroelasticity and flight dynamics. *Progress in Aerospace Sciences*, 55, 46-72. DOI: `10.1016/j.paerosci.2012.06.001`. BibTeX key: `murua2012applications`.
- Nekoo, S. R., et al. (2025). Recent advances in dynamics and control of ornithopters: A review. *The International Journal of Robotics Research*. DOI: `10.1177/02783649251343638`. BibTeX key: `nekoo2025recent`.
- Peters, D. A., Karunamoorthy, S., & Cao, W. M. (1995). Finite state induced flow models. Part I: Two-dimensional thin airfoil. *Journal of Aircraft*. DOI not verified. BibTeX key: `peters1995finite`.
- Phlips, P. J., East, R. A., & Pratt, N. H. (1981). An unsteady lifting line theory of flapping wings with application to the forward flight of birds. *Journal of Fluid Mechanics*, 112, 97-125. DOI: `10.1017/S0022112081000311`. BibTeX key: `phlips1981unsteady`.
- Ramesh, K., Gopalarathnam, A., Granlund, K., Ol, M. V., & Edwards, J. R. (2014). An unsteady airfoil theory applied to pitching and surging motions. *Theoretical and Computational Fluid Dynamics*, 28, 501-521. DOI: `10.1007/s00162-014-0332-0`. BibTeX key: `ramesh2014unsteady`.
- Roccia, B. A., Preidikman, S., Massa, J. C., & Mook, D. T. (2013). Modified Unsteady Vortex-Lattice Method to Study Flapping Wings in Hover Flight. *AIAA Journal*, 51(10), 2628-2642. DOI: `10.2514/1.J052262`. BibTeX key: `roccia2013modified`.
- Ruiz, C., Acosta, J. A., & Ollero, A. (2022). Aerodynamic reduced-order Volterra model of an ornithopter under high-amplitude flapping. *Aerospace Science and Technology*, 121, 107331. DOI: `10.1016/j.ast.2022.107331`. BibTeX key: `ruiz2022aerodynamic`.
- Sheng, W., Galbraith, R. A. McD., & Coton, F. N. (2008). A modified dynamic stall model for low Mach numbers. *Journal of Solar Energy Engineering*. DOI not verified. BibTeX key: `sheng2008modified`.
- Shukla, R. K., & Eldredge, J. D. (2007). An inviscid model for vortex shedding from a deforming body. *Theoretical and Computational Fluid Dynamics*, 21, 343-368. DOI: `10.1007/s00162-007-0041-2`. BibTeX key: `shukla2007inviscid`.
- Smith, M. J. C., Wilkin, P. J., & Williams, M. H. (1996). The advantages of an unsteady panel method in modelling the aerodynamic forces on rigid flapping wings. *Journal of Experimental Biology*, 199(5), 1073-1083. DOI: `10.1242/jeb.199.5.1073`. BibTeX key: `smith1996advantages`.
- Stanford, B. K., & Beran, P. S. (2010). Analytical Sensitivity Analysis of an Unsteady Vortex-Lattice Method for Flapping-Wing Optimization. *Journal of Aircraft*. DOI not verified. BibTeX key: `stanford2010analytical`.
- Taha, H. E., Hajj, M. R., & Beran, P. S. (2014). State-space representation of the unsteady aerodynamics of flapping flight. *Aerospace Science and Technology*, 34, 1-11. DOI: `10.1016/j.ast.2014.01.011`. BibTeX key: `taha2014state`.
- Theodorsen, T. (1935). General Theory of Aerodynamic Instability and the Mechanism of Flutter. *NACA Report 496*. URL: <https://ntrs.nasa.gov/citations/19930090935>. BibTeX key: `theodorsen1935general`.
- Vest, M. S., & Katz, J. (1996). Unsteady Aerodynamic Model of Flapping Wings. *AIAA Journal*, 34(7), 1435-1440. DOI: `10.2514/3.13250`. BibTeX key: `vest1996unsteady`.
- Yang, H.-H., Lee, S.-G., Lee, E.-H., & Han, J.-H. (2025). Numerical simulation framework of bird-inspired ornithopter in forward flight using modified Unsteady Vortex Lattice Method coupled with Multi-Flexible-Body Dynamics. *Journal of Fluids and Structures*, 133, 104263. DOI: `10.1016/j.jfluidstructs.2024.104263`. BibTeX key: `yang2025numerical`.
