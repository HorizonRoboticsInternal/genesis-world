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
  The current verified path uses IPC FEM cloth, standalone IPC-coupled rigid
  parallel-gripper boxes as the actual contact driver, a table, three cameras,
  and an imported DexGarmentLab T-shirt mesh. The Piper-X arms are opt-in
  visual context and are held stationary while the standalone grippers test
  grasping:

  ```bash
  .venv/bin/python examples/IPC_Solver/ipc_dual_piperx_shirt_lift.py \
    --record \
    --output-dir recordings/ipc_dual_piperx_shirt_lift_ipc_y_grip
  ```

  Add `--show-piper` when the stationary dual Piper-X arms should be visible.
  The generated Genesis URDF rotates the local dual-arm baseline by yaw `90 deg`
  and places the root at `(-0.30, -0.86, 0.0)`, so the URDF's built-in
  right-base offset becomes a side-by-side table-front baseline with the shirt
  on the `+y` side of the arms.

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
  the shirt is moved only by IPC contact/friction against the rigid gripper
  boxes.
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
  `head_camera`, and `right_camera`.
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
- The current scripted IPC sequence uses two gripper pairs near the shirt
  middle (`x=-0.08` and `x=0.08`), lifts high, holds, shakes laterally, releases,
  then descends open to a near-table clearance of `0.2 mm`, closes again, lifts,
  and releases. The center camera should stay high/wide enough to frame
  `FINGER_LIFT_Z = TABLE_TOP_Z + 0.36`.
