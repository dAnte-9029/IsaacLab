# ADR: URDF wing roots and root-local CSV geometry for ULLT

- Status: Accepted
- Date: 2026-10-27
- Scope: Geometry interpretation for the opt-in ULLT backend
- Supersedes: Geometry interpretation and alleged span discrepancy in
  `ADR-2026-10-26-ullt-wing-geometry-and-deferred-twist.md`; other constraints remain

## Context

The user clarifies that the URDF is authoritative: the wing roots are at
body `y=+/-0.056 m`, and the existing CSV describes the complete shape of each
wing measured outward from its own root. The central space is empty and does
not participate in wing aerodynamic loading.

The earlier interpretation combined the user's approximate 1.4 m span and
50 mm root offsets to infer a 0.650 m root-to-tip extent. That interpretation
is superseded. In particular, the earlier conclusion that DeLaurier places
an overlong wing outside each hinge is withdrawn; it relied on the wrong
reference point for the CSV coordinates.

## Decision

1. Use the existing URDF wing-root origins and orientations, including the
   `+/-0.056 m` root offsets, as the geometric registration authority.
2. Use `outputs_DeLaurier/right_wing_te_fit_poly5_gap50.csv` as the full
   root-local span/chord distribution. Its inferred outboard extent is 0.700 m.
   Do not subtract 50 or 56 mm, truncate it to 0.650 m, or remove its inner
   50 mm of nonzero chord based on the filename.
3. Place left/right sections outward from their respective wing origins using
   the mirrored link convention and actual link transforms. Do not add any
   lifting panels or wing forces in the central gap.
4. Interpret the previously stated "central 100 mm" as an approximate
   description of that empty region; the exact root-to-root spacing follows
   the URDF and is 112 mm. Do not create an extra 100 mm exclusion inside
   the root-local wing geometry. Cross-wing induced-flow coupling remains.
5. Retain the existing aerodynamic defaults, robot assets and formal RL
   configuration. Twist and a separate leading-edge pitch degree of freedom
   remain deferred as recorded in the preceding ADR.

The root-to-tip CSV placement already used by DeLaurier agrees with this
clarification. This does not validate other aerodynamic approximations or
change the distinction between a mechanical pitching axis, quarter-chord
lifting line and three-quarter-chord kinematic sampling point.

## Alternatives considered

- Enforce a 1.4 m total span by shortening the CSV: rejected by the user's
  explicit choice of URDF roots and the complete root-local CSV.
- Treat `gap50` as a current instruction to discard the inner 50 mm: rejected;
  the filename does not override the clarified coordinate contract.
- Change URDF roots to +/-50 mm: rejected; +/-56 mm is confirmed correct.

## Consequences and assumptions

Ignoring the small neutral wing roll, the resulting nominal tip span is
`2 * (0.056 + 0.700) = 1.512 m`. Actual projected span follows the link pose.
This derived value is not an independent measurement or a reason to rescale
the user-approved geometry.

The root-local CSV and URDF replace the earlier scalar span estimate as the
implementation authority. Sectional flow assumptions and the applicability
of the paper's unswept-wing downwash correction remain separate design items.

## Validation requirements

- Verify root-local stations transformed through the left/right URDF poses.
- Verify that no central lifting panels or duplicated root offsets appear.
- Verify that the inner CSV chord samples are retained, with no gap subtraction.
- Retain force/moment reference, mirror and old-backend regression checks.
- Do not claim aerodynamic accuracy from geometric consistency alone.

## Reconsideration triggers

Revisit this contract only if the user supplies a replacement CSV, URDF or
explicitly revised coordinate definition. Future twist must preserve the same
geometric reference while introducing its own section orientations.
