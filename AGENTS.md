# AGENTS.md - Genesis AI Agent Guide

Guide for AI coding assistants working with the Genesis physics simulation codebase.

## Quick Start

```bash
# Setup
uv sync
uv pip install torch --index-url https://download.pytorch.org/whl/cu126  # or cpu/metal

# Run tests
uv run pytest tests/
uv run pytest tests/ -m required  # minimal set

# Run examples
uv run examples/tutorials/hello_genesis.py
```

## How to Run Tests

```bash
uv run pytest tests/                      # All tests
uv run pytest tests/test_file.py          # Specific file
uv run pytest tests/ --backend=gpu        # GPU backend
uv run pytest tests/ -m required          # Required tests only
uv run pytest tests/ -m "not slow"        # Skip slow tests
```

See [TESTING.md](.github/contributing/TESTING.md) for details.

## How to Contribute

### PR Title Prefixes

- `[BUG FIX]` - Non-breaking bug fixes
- `[FEATURE]` - New functionality
- `[MISC]` - Minor changes (docs, typos)
- `[CHANGING]` - Behavior changes
- `[BREAKING]` - Breaking API changes

### Before Submitting

1. Install pre-commit hooks: `pre-commit install`
2. Run required tests: `uv run pytest -m required tests/`
3. Link to related issue in PR description

See [PULL_REQUESTS.md](.github/contributing/PULL_REQUESTS.md) for details.

## Formatting & Lint

Genesis uses **ruff** for linting and formatting (via pre-commit):

```bash
# Install hooks (auto-runs on commit)
pre-commit install

# Manual run
pre-commit run --all-files
```

**Rules:**
- Line length: 120 characters
- Format: ruff-format (black-compatible)
- Lint: ruff-check

See [CODING_CONVENTIONS.md](.github/contributing/CODING_CONVENTIONS.md) for code style.

## When to Ask a Human

Ask for clarification when:

- **Ambiguous requirements** - Multiple valid interpretations exist
- **Breaking changes** - Changes that affect public APIs or behavior
- **Architecture decisions** - New solvers, major refactors, new entity types
- **Performance trade-offs** - When optimization conflicts with readability
- **Test failures** - Unclear why tests fail or how to fix them
- **Cross-solver coupling** - Changes affecting multiple physics solvers

Do NOT ask when:
- Standard bug fixes with clear reproduction steps
- Documentation updates
- Adding tests for existing functionality
- Code style fixes flagged by linters

## Reference Documentation

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](.github/contributing/ARCHITECTURE.md) | Project structure, solvers, entities |
| [TESTING.md](.github/contributing/TESTING.md) | Testing guide and fixtures |
| [CODING_CONVENTIONS.md](.github/contributing/CODING_CONVENTIONS.md) | Code style and patterns |
| [EXAMPLES.md](.github/contributing/EXAMPLES.md) | Examples reference |
| [PULL_REQUESTS.md](.github/contributing/PULL_REQUESTS.md) | PR guidelines |

## Local Cloth Grasp Notes

- The scripted shirt lift demo lives at
  `examples/IPC_Solver/ipc_dual_piperx_shirt_lift.py`.
  The current verified path uses IPC FEM cloth, dual Piper-X covered fingers as
  the contact driver, a table, three cameras, and an imported DexGarmentLab
  T-shirt mesh. The old standalone parallel-gripper boxes were removed, so the
  script now requires Piper-X to be visible and driven:

  ```bash
  .venv/bin/python examples/IPC_Solver/ipc_dual_piperx_shirt_lift.py \
    --record \
    --output-dir recordings/ipc_dual_piperx_shirt_lift_ipc_y_grip
  ```

  The tabletop is centered at `(0.0, -0.48, 0.065)` and uses size
  `(1.64, 0.72, 0.13)`, twice the previous X width while keeping the shirt and
  robots centered in the same workspace. The generated Genesis URDF rotates the
  local dual-arm baseline by yaw `90 deg` and places the root at
  `(-0.30, -0.86, 0.137)`. That root height comes from the physical setup
  measurement that the Piper-X shoulder joint center is `13 cm` above the
  tabletop: the URDF shoulder center is `0.123 m` above `base_link`, while the
  table top is `z=0.13`, so `root_z = 0.13 + 0.13 - 0.123 = 0.137`. The URDF's
  built-in right-base offset becomes a side-by-side table-front baseline with
  the shirt on the `+y` side of the arms. The real Piper finger links
  `left_link7`, `left_link8`, `right_link7`, and `right_link8` are the only
  IPC-coupled manipulation links.

- The demo uses the checked-in DexGarmentLab short-sleeve T-shirt OBJ asset at
  `genesis/assets/meshes/garments/dexgarmentlab_short_sleeve_tshirt.obj`. The
  asset is generated from
  `/home/horizon/DexGarmentLab/Assets/Garment/Tops/NoCollar_Ssleeve_FrontClose/TNSC_T_Shirt_Short_Sleeve/TNSC_T_Shirt_Short_Sleeve_obj.usd`
  with `examples/IPC_Solver/export_dexgarmentlab_tshirt_asset.py`. The local
  Genesis venv does not provide `pxr`, so run the exporter with
  `/home/horizon/isaacsim_env/bin/python`. The exporter triangulates all USD
  mesh prim faces, applies each prim's local-to-world transform, centers X/Y,
  min-Z aligns the garment for the tabletop, and applies the Isaac demo's
  short-sleeve garment scale of `0.55`. The demo defaults to the checked-in OBJ;
  use `--refresh-shirt-asset` only when intentionally regenerating it from the
  DexGarmentLab USD.
- Do not use hidden particle attachment in the Genesis IPC shirt lift demo.
  The old PBD path fixed/overrode selected particles with `fix_particles_*` or
  `set_particles_pos()`, which made lift reliable but was not the same behavior
  as Genesis' IPC teleop examples. The current demo has no attachment calls;
  the shirt is moved only by IPC contact/friction against the covered Piper-X
  finger links.
- When `--show-piper` is used, the script injects RobotTwin-style
  `left_camera` and `right_camera` fixed links under `left_link6` and
  `right_link6` using the calibrated Piper-X wrist camera origins from
  `/home/horizon/RoboOrchardLab/post_training_foldclothes/fold_clothes_ro_piperx.config.json`.
  The fixed `head_camera` is the resized D455 profile from that config:
  `392x252`, `fy=310`, `fovy=44.23872564716461`. Use the calibration position
  as a camera center in the Genesis workspace frame with the table-front `y`
  offset and the real-root-height `z` offset applied, giving roughly
  `(-0.007, -1.177, 0.741)`. Do not reuse
  the older transformed `z=1.35` placement; it frames the shirt too high and
  hides more of the arms than the real RobotTwin view. The D455 look direction
  is derived from the config quaternion by taking `-R[0, :]` as optical forward
  and `R[1, :]` as image-left, which points through the shirt center at about
  tabletop height. When recording, the script saves individual camera MP4s plus
  `left_mid_right.mp4`, a horizontal stack of `left_camera`,
  `head_camera`, and `right_camera`. Wrist cameras use a `1 cm` near plane;
  the default Genesis near plane clipped nearby finger geometry in wrist views
  while the farther head camera still showed the full fingers.
- The generated Piper-X URDF mirrors the ClothesFoldingEnv covered-finger setup:
  it keeps the original grey J7/J8 finger visuals and adds black cover visuals
  from `/home/horizon/gripper_finger_cover.stl` at scale `0.001`. For Genesis
  IPC, do not leave the covers as visual-only and do not use one large collision
  box for the whole finger. Each finger link now gets two active collision
  boxes, similar in spirit to Panda's separate fingertip pad boxes: a slim
  finger-body box and a separate cover-surface box. The cover collision box is
  the IK/manipulation reference point. Exact STL cover mesh collision is avoided
  for now because it is slower and more fragile for IPC than primitive pad
  boxes. The local STL cover mesh bounds are about
  `0.0437 x 0.0700 x 0.0224 m`; in the Piper finger joint frame, link-local
  `Z` rotates into the gripper closing direction. A full visual-thickness cover
  box intersects at Genesis IPC build time, so the primitive cover box is a
  slim fingertip pad aligned to the STL visual inner face rather than a full
  cover-volume collider.
- For contact-only Piper-X shirt lift tuning, the covered-finger IPC box sizes
  are finger body `(0.020, 0.065, 0.010)` and cover surface
  `(0.043709, 0.070000, 0.014000)` in link-local XYZ, with cover collision
  centers matched to the STL visual transform and used for IK local points. A
  full visual-thickness cover box `(0.043709, 0.070000, 0.022420)` fails IPC
  initialization because opposing covers intersect in the neutral closed
  gripper geometry. The latest local clamp tuning uses gripper
  IPC `coup_friction=12.0`, gripper gains `kp=5000`, `kv=500`, and force limit
  `3000`; with the previous oversized cover proxy `(0.044, 0.105, 0.014)` this
  produced about `0.113 m` max cloth lift over the initial centroid. A thicker
  oversized cover box `(0.048, 0.115, 0.018)` failed Genesis IPC initialization
  because the closed finger collision geometry self-intersected. Revalidate IPC
  init and lift quality after changing the cover box dimensions/centers.
  Use `--init-only --closed-opening <gap>` for fast IPC sanity probes, and use
  `--focus-grasp --record` while tuning: the full approach still simulates, but
  recording renders only lower/contact/close/hold/push/lift/release phases for
  faster review.
- The real-Piper Genesis script initializes normal episodes at open gripper qpos
  before `scene.build()` and starts the first recorded frame with the approach
  action; do not add a settle/open/hold-open lead-in just for video cleanliness.
  For IPC init/gap checks, use `--init-only`; that path still initializes at the
  configured closed opening before build. Post-build `set_qpos()` jumps can make
  the first `ipc_world.advance()` very slow because the soft-constraint bodies
  have to correct a large frame-1 transform jump.
- In the real-Piper Genesis recording, the standalone proxy grippers are no
  longer present. If the shirt still moves, it is from IPC contact against the
  Piper finger links. The current scripted IK targets one real IPC collision
  center per arm, `left_link7` and `right_link8`, using Genesis'
  `local_points` argument; this avoids the previous gripper-base target
  mismatch where low gripper-base IK error still missed the shirt. A four-finger
  collision-center IK target was over-constrained for these arms at the current
  poses and produced about `8 cm` position error, while the one-finger-per-arm
  target solves to roughly `1e-4 m` or better. The resulting IPC-only sequence
  visibly deforms/moves the shirt, but it is still not a reliable lift grasp
  without an attachment/sticking mechanism or further contact/trajectory tuning.
- The Genesis IPC Piper shirt script uses `SimOptions(dt=0.02, substeps=4)`.
  Each `scene.step()` advances `0.02 s` of simulated time, so external commands
  run at `50 Hz`; internal physics substeps are `0.005 s` each, effectively
  `200 Hz`. The camera recorder saves every rendered simulation step at
  `50 FPS`, so MP4 playback matches simulated real time. Phase-step minima are
  scaled for the `0.02 s` timestep to preserve roughly the same simulated
  durations as the previous `0.01 s` setup. The first local 50 Hz full run wrote
  `/home/horizon/genesis-world/recordings/ipc_dual_piperx_50hz_full_video/left_mid_right.mp4`;
  it did not qualitatively change the current gripper issue, with `grip_gap`
  still around `0.035 m` during squeeze/lift.
  The script logs `finger_z` from the actual IPC collision-box centers rather
  than from the misleading link origins.
- Genesis can warn about falling back to the legacy URDF parser for Piper DAE
  meshes and filtering neutral self-collision geometry pairs; the local short
  runtime check completed despite those warnings.
- Genesis IPC cloth uses `gs.materials.FEM.Cloth` with OBJ shell meshes and
  `IPCCouplerOptions`; it does not expose the same vertex constraint API used by
  non-IPC FEM examples. `FEMEntity.set_vertex_constraints()` rejects IPCCoupler.
- The imported DexGarmentLab T-shirt mesh is not as clean as Genesis'
  `IPC/grid20x20.obj` cloth. With teleop-style `thickness=0.001`, IPC rejected
  it during world initialization because nearby shirt surface triangles were
  closer than the effective shell thickness. Use `thickness=0.0002` for this
  imported USD mesh unless the mesh is flattened/cleaned.
- The contact-only IPC lift needs higher finger/cloth friction and a slower
  lift than the baseline teleop grid scene. The verified run used
  `friction_mu=2.0` on the FEM cloth, table `coup_friction=0.4`, gripper
  `coup_type="two_way_soft_constraint"`, gripper `coup_friction=4.0`, and
  front/back-closing gripper pairs with Piper-like `0.026 x 0.012 x 0.080 m`
  box geometry. A side-closing setup and a `10 mm` closed gap only wrinkled the
  shirt and slipped during lift.
- The current scripted IPC sequence follows the ClothesFoldingEnv EE phase
  structure instead of the earlier Genesis-only shake and second-grasp routine:
  open-start approach, lower to fingertip contact, hold contact open, close,
  hold closed, push `0.20 m` along `+y`, lift to `0.46 m`, hold lift, release,
  and retreat. Do not add a second squeeze/push/lift phase after the push; it
  can look like a second manipulation attempt in the recording. Keep the
  gripper target closed throughout the lift and high-hold phases; at short
  horizon scales the articulation can still be rising when a release phase
  starts, so the closed lift/high-hold windows need explicit minimum durations
  before any open-gripper release command. A local ablation that slowed lift
  from 50 to 100 steps at 50 Hz made the final cloth lift worse
  (`0.0586 m` versus `0.0827 m` max lift over initial), so visible one-sided
  slipping should be debugged first through pre-lift clamp quality, gripper
  authority, and left/right gripper-gap diagnostics rather than lift speed
  alone. For post-lift shaking, keep the grippers closed and use per-arm cloth
  proximity diagnostics when investigating slip: compare left/right gripper
  gap, nearby particle counts, nearest cloth distance, and max height of nearby
  cloth. A later `0.040 m` world-X shake recording kept both grippers closed
  and retained similar left/right nearby-particle counts through shake, with
  the major drop occurring after the explicit release command. The current
  stress-test shake repeats the centered left/right/left/center pattern four
  times with one-quarter segment duration, so it has four times the cycles and
  four times the lateral speed while keeping the phase duration about the same.
  Do not infer visual gripper opening from the commanded Piper-X gripper qpos
  alone. In a robot-only FK/settling probe the cover-center gap is monotonic
  (`0.035 m` command gave about `0.088 m`, `0.049 m` gave about `0.111 m`),
  but in the full IPC-coupled shirt scene the actual lowered/contact
  cover-center gap is nonmonotonic: `0.035` gave about `0.0608 m`, `0.030`
  about `0.0643 m`, `0.025` about `0.0673 m`, `0.020` about `0.0575 m`, and
  near-limit `0.049` only about `0.0506 m`. The current calibrated wider visual
  opening is therefore `0.025 m`, even though it is numerically smaller than
  the older `0.035 m` qpos command. Log physical cover-center gap/vector
  diagnostics when tuning gripper opening; larger qpos is not necessarily
  larger visual opening in the IPC-coupled episode. Increasing the target by
  50% to `0.0525 m` also caused a Genesis joint-limit warning and reduced
  actual clamping, so do not use that wider qpos command as the default unless
  the imported gripper limits/drive mapping are changed. Use
  `--show-finger-collision-boxes` to render colored body/cover collision boxes
  in the generated Piper URDF; it also makes the cover STL half-transparent for
  inspection. Raising the gripper drive to `kp=30000`, `kv=3000`, and force
  limit `50000` reduced but did not eliminate contact back-driving: during
  close/hold the commanded zero target still settled around `0.025-0.026`
  actual qpos with `~0.018-0.020 m` physical cover-center gap. Treat zero-close
  with the current thick cover boxes as geometry/contact-limited rather than
  just a weak actuator issue. The later collision-box tuning keeps the blue
  body boxes small and behind the fingertip
  (`0.014 x 0.030 x 0.006`, centers at local `y=+/-0.050`) and makes the red
  cover boxes the fingertip pads. The current red cover pad size is
  `0.012 x 0.040 x 0.003`: a thin inspection/contact slab near the visible
  fingertip. Keep the cover STL visual at its natural center. A truly
  flush-to-visual red pad, even with this smaller footprint and zero proud
  margin, fails Genesis IPC build because the two visible cover inner faces are
  already interpenetrating in neutral geometry. The current working compromise
  backs the red contact face off `3 mm` from the visible cover inner face while
  leaving the cover visual unshifted. Use
  `--gripper-close-diagnostic --show-finger-collision-boxes --record` for a
  short no-shirt, fixed-arm close/hold video. In that diagnostic with table
  contact, the grippers still settled at about `0.0251` actual qpos and
  `0.0182 m` cover-center gap, confirming that table/contact constraints can
  still keep the fingers from reaching commanded zero even without cloth. With
  the thinner/longer red pads, a full shirt run settled around `0.0258` qpos
  with only about `0.0076-0.0080 m` cover-center gap during closed hold, and
  `0.0065-0.0068 m` during lift/shake. That run lifted mostly on the right
  gripper; diagnostics showed the left side lost nearby cloth during lift
  (`left_near=0`, `left_d~0.2 m`) while the right side stayed near the cloth
  (`right_d~0.006-0.007 m`). The
  close phase alone does not produce a strong shirt buckle: in the local
  default-opening fast-shake run, close changed cloth `span_y` only from about
  `0.379` to `0.375` and kept `z_std` around `0.007`. The larger wrinkle/fold
  signal appeared during push/lift, where `span_y` contracted and `z_std`
  increased. Lowering the Genesis contact target from `TABLE_TOP_Z + 0.018` to
  `TABLE_TOP_Z + 0.012` improved immediate proximity only slightly, still did
  not create close-phase buckling, and made fast-shake grasp stability worse
  with one side losing proximity during shake; keep the default `+0.018` unless
  another trajectory/contact strategy is being tested. The
  scripted approach/lift heights are
  `0.250 m` and `0.460 m`. Do not reuse the Isaac
  fingertip-contact height `0.220 m` directly in Genesis: with this table and
  shirt mesh, that leaves the IPC finger collision centers hovering above the
  settled cloth. Use the Genesis-local contact target `TABLE_TOP_Z + 0.018` so
  the fingers lower near tabletop level before closing. Keep the
  ClothesFoldingEnv phase ratios but use a Genesis-local
  `physics_steps_per_action` scale; the articulation needs more than the Isaac
  minimum frame counts to settle onto the IK targets.
- For visual room context around the Genesis IPC table, avoid adding far walls,
  ceiling panels, or light panels as regular `scene.add_entity()` boxes. Even
  with `collision=False`, they still become extra `RigidEntity`s and locally
  caused IPC/CUDA kernel compilation failure. Draw noninteractive room geometry
  after `scene.build()` with persistent visualizer debug boxes instead; IPC
  then sees only the table, shirt, and robot while the cameras still render the
  three light-gray walls, 3 m ceiling, and under-ceiling light panels.
