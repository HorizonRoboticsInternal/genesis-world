## IPC examples

This repo shows how to use IPC coupler. (Incremental Potential Contact) algorithm provides a unified framework for robust contact handling across different material types including cloth, deformable FEM objects, and rigid bodies.

**Prerequisites:**
1. Install LibUIPC optional dependency: `pip install pyuipc`
2. Ensure Genesis is in development mode: `pip install -e .`

Beware only Linux/Windows x86 CPU & Nvidia GPU is supported for now.

**Test Cases:**

1. **Basic cloth simulation:**
   ```bash
   python examples/IPC_Solver/ipc_objects_falling.py -v
   ```
  Expected: A cloth falls under gravity and collides with the ground plane clustered with objects

1. **Robotic grasping of deformables:**
   ```bash
   python examples/IPC_Solver/ipc_robot_grasp_cube.py -v
   ```
 Expected: Franka Panda robot grasps and manipulates a deformable cube with IPC contact resolution

1. **Interactive cloth manipulation:**
   ```bash
   python examples/IPC_Solver/ipc_robot_cloth_teleop.py
   ```
    Expected: Interactive Franka Panda robot manipulation and manipulates two pieces of cloths

1. **Scripted T-shirt grasp/lift with IPC contact:**
   ```bash
   python examples/IPC_Solver/ipc_dual_piperx_shirt_lift.py --record
   ```
   Expected: Standalone IPC-coupled parallel grippers grasp the middle of a
   DexGarmentLab short-sleeve T-shirt mesh, lift, shake, release, regrasp close
   to the table, and lift/release again. The default mesh is the repo-local
   asset:
   `genesis/assets/meshes/garments/dexgarmentlab_short_sleeve_tshirt.obj`.
   Add `--show-piper` to include stationary dual Piper-X arm visuals installed
   along the table front edge. The contact driver remains the standalone IPC
   gripper boxes.

   The demo uses a RobotTwin-style three-camera layout: a static D455-like
   `head_camera` at `392x252`, plus `left_camera` and `right_camera` wrist
   views. With `--show-piper`, the generated Piper-X URDF injects fixed
   RobotTwin wrist camera links under `left_link6` and `right_link6`, and the
   Genesis wrist cameras attach to those links. Recording writes the three
   individual camera videos and a concatenated `left_mid_right.mp4` with
   `left_camera`, `head_camera`, and `right_camera` arranged horizontally.

   To regenerate that OBJ from the local DexGarmentLab USD asset, use Isaac
   Sim's Python because the Genesis venv does not provide `pxr`:

   ```bash
   /home/horizon/isaacsim_env/bin/python \
     examples/IPC_Solver/export_dexgarmentlab_tshirt_asset.py
   ```

   The exporter reads
   `/home/horizon/DexGarmentLab/Assets/Garment/Tops/NoCollar_Ssleeve_FrontClose/TNSC_T_Shirt_Short_Sleeve/TNSC_T_Shirt_Short_Sleeve_obj.usd`,
   triangulates all USD mesh prim faces, applies each prim's local-to-world
   transform, centers X/Y, min-Z aligns the garment, and applies uniform scale
   `0.55`.

**Verification:**
- No interpenetration between objects during contact
- Cloth renders correctly (visible mesh with proper shading)
- Two-way coupling: Genesis rigid body motion affects IPC objects and vice versa
