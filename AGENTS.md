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

- The dual Piper-X shirt lift demo lives at
  `examples/IPC_Solver/ipc_dual_piperx_shirt_lift.py`.
  The verified fast path is:

  ```bash
  .venv/bin/python examples/IPC_Solver/ipc_dual_piperx_shirt_lift.py \
    --hide-piper --horizon-scale 0.5 --record
  ```

- Genesis IPC cloth uses `gs.materials.FEM.Cloth` with OBJ shell meshes and
  `IPCCouplerOptions`; it does not expose the same vertex constraint API used by
  non-IPC FEM examples. `FEMEntity.set_vertex_constraints()` rejects IPCCoupler.
- The local Piper-X URDF imports through Genesis' legacy URDF parser because
  some DAE meshes are not decoded by the primary parser. Keep generated URDF
  copies local to output directories and rewrite relative mesh paths to absolute
  paths when copying the URDF outside `/home/horizon/newton_cloth`.
- Raw Piper gripper collision meshes are not IPC-friendly at qpos0: closed
  finger pairs can make the IPC world invalid. For scripted cloth lift, the demo
  uses simple rigid box finger proxies and keeps the full Piper import as a
  visual/controller reference.
- PBD cloth attachment is the reliable verified lift path in this repo state.
  Use `gs.materials.PBD.Cloth`, choose nearby particles, then call
  `cloth.fix_particles_to_link(proxy.link_start, particles_idx_local=...)`.
  After attachment, `get_particles_pos()` can contain sentinel positions near
  `100` for inactive/attached particles; filter those when logging centroid or
  height diagnostics.
- PBD cloth should run on the GPU backend by default here. CPU PBD stepping was
  too slow/hung before phase diagnostics in the local verification run.
