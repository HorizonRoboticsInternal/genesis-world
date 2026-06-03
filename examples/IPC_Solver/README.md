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
   Expected: IPC-coupled Piper-X covered fingers execute a scripted
   approach/close/lift/release sequence over the DexGarmentLab short-sleeve
   T-shirt mesh. The default mesh is the repo-local asset:
   `genesis/assets/meshes/garments/dexgarmentlab_short_sleeve_tshirt.obj`.
   The dual Piper-X arms are installed along the table front edge and are the
   IPC contact driver; the previous standalone gripper boxes have been removed.
   The table is twice the previous X width while remaining centered around the
   shirt and robots.
   The generated Piper-X URDF adds the same black finger-cover STL visuals from
   `/home/horizon/gripper_finger_cover.stl`. Genesis IPC uses simple box
   collisions on `left_link7`, `left_link8`, `right_link7`, and `right_link8`;
   the long box axis is link-local Y because the Piper finger joint frame
   rotates link-local Z into the closing direction.

   The demo uses a RobotTwin-style three-camera layout: a static D455-like
   `head_camera` at `392x252`, plus `left_camera` and `right_camera` wrist
   views. The generated Piper-X URDF injects fixed RobotTwin wrist camera links
   under `left_link6` and `right_link6`, and the Genesis wrist cameras attach
   to those links. Recording writes the three individual camera videos and a
   concatenated `left_mid_right.mp4` with `left_camera`, `head_camera`, and
   `right_camera` arranged horizontally.

   The current real-Piper path starts from all-zero qpos, then interpolates to
   IK targets at the shirt. It does not use hidden particle attachment or
   standalone proxy boxes. The scripted IK targets one real IPC collision center
   per arm (`left_link7` and `right_link8`) so the shirt visibly deforms under
   contact; reliable lifting remains a contact/trajectory tuning problem.

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
