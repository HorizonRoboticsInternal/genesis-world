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
  `(-0.30, -0.86, 0.0)`, so the URDF's built-in right-base offset becomes a
  side-by-side table-front baseline with the shirt on the `+y` side of the
  arms. The real Piper finger links
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
  `/home/horizon/robo_orchard_lab` branch
  `test/deploy_ckpt_in_RoboOrchardLab`. Genesis cameras attach to those links
  with the frame conversion from RobotTwin/SAPIEN camera axes (`+X` optical
  forward, `+Y` image-left, `+Z` image-up) to Genesis/OpenGL camera axes. The
  static `head_camera` uses the RobotTwin Piper-X world pose
  `[0.01715773707478663, -0.4573830598833294, 1.353635842513242]`, forward
  `[0.03060834543810837, 0.5532082633105504, -0.8324804782062258]`, and D455
  resized profile `392x252`, `fovy=44.23872564716461`, translated by the
  Genesis table's `y=-0.48` workspace offset. This replaces the older
  front/center debug view because the ClothesFoldingEnv middle camera was noted
  as slightly off. When recording, the script saves individual camera MP4s plus
  `left_mid_right.mp4`, a 60 FPS horizontal stack of `left_camera`,
  `head_camera`, and `right_camera`. Wrist cameras use a `1 cm` near plane;
  the default Genesis near plane clipped nearby finger geometry in wrist views
  while the farther head camera still showed the full fingers.
- The generated Piper-X URDF mirrors the ClothesFoldingEnv covered-finger setup:
  it keeps the original grey J7/J8 finger visuals, adds black cover visuals
  from `/home/horizon/gripper_finger_cover.stl` at scale `0.001`, and replaces
  each finger collision with a simple IPC box centered at `(0, -0.046, -0.026)`
  for `link7` and `(0, 0.046, -0.026)` for `link8`, with size
  `(0.026, 0.080, 0.012)`. The visual cover placement follows
  ClothesFoldingEnv, but the active Genesis IPC collision uses a slimmer,
  non-overlapping box: in the Piper finger joint frame, link-local `Z` rotates
  into the gripper closing direction, so using the old `0.080 m` local-Z
  thickness made the two finger collisions self-intersect at IPC initialization.
  Exact cover-length collision boxes should be revalidated against Genesis IPC
  before use.
- The real-Piper Genesis script initializes the robot at all-zero qpos before
  `scene.build()` and starts the recording with a short zero-pose settle phase.
  It then interpolates from zero to the first IK target instead of teleporting
  the articulation after IPC build. Post-build `set_qpos()` jumps can make the
  first `ipc_world.advance()` very slow because the soft-constraint bodies have
  to correct a large frame-1 transform jump.
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
- The current scripted IPC sequence moves from all-zero qpos to two
  shirt-centered contact targets at `x=-0.10` and `x=0.10`, approaches, lowers,
  closes, pushes, lifts, shakes laterally, releases, then attempts a lower
  second grasp. The center camera should stay high/wide enough to frame
  `PIPER_BASE_LIFT_Z = TABLE_TOP_Z + 0.45`.
