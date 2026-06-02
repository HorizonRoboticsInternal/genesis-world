import argparse
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

import genesis as gs


PIPER_DUAL_URDF = Path("/home/horizon/newton_cloth/piper_x_description_dualarm.urdf")

LEFT_ARM_DOFS = [0, 2, 4, 6, 8, 10]
RIGHT_ARM_DOFS = [1, 3, 5, 7, 9, 11]
ALL_ARM_DOFS = LEFT_ARM_DOFS + RIGHT_ARM_DOFS
LEFT_GRIPPER_DOFS = [12, 13]
RIGHT_GRIPPER_DOFS = [14, 15]
ALL_GRIPPER_DOFS = LEFT_GRIPPER_DOFS + RIGHT_GRIPPER_DOFS
ALL_CONTROL_DOFS = ALL_ARM_DOFS + ALL_GRIPPER_DOFS

OPEN_GRIPPER = np.array([0.035, -0.035, 0.035, -0.035], dtype=np.float32)

TABLE_CENTER = np.array([0.0, -0.48, 0.065], dtype=np.float32)
TABLE_SIZE = np.array([0.82, 0.72, 0.13], dtype=np.float32)
TABLE_TOP_Z = TABLE_CENTER[2] + TABLE_SIZE[2] * 0.5
SHIRT_CENTER = np.array([0.0, -0.48, TABLE_TOP_Z + 0.010], dtype=np.float32)
ROBOT_ROOT_POS = (0.0, -0.86, 0.0)
GRIPPER_PAIR_XS = (-0.08, 0.08)
GRIPPER_START_Y = SHIRT_CENTER[1] - 0.26
GRIPPER_CONTACT_Y = SHIRT_CENTER[1] - 0.02
GRIPPER_PUSH_Y = SHIRT_CENTER[1] + 0.07
FINGER_SIZE = np.array([0.026, 0.012, 0.080], dtype=np.float32)
FINGER_HIGH_Z = TABLE_TOP_Z + 0.22
FINGER_CONTACT_Z = TABLE_TOP_Z + FINGER_SIZE[2] * 0.5 + 0.001
FINGER_NEAR_TABLE_Z = TABLE_TOP_Z + FINGER_SIZE[2] * 0.5 + 0.0002
FINGER_LIFT_Z = TABLE_TOP_Z + 0.36
SHAKE_X_OFFSET = 0.025
OPEN_FINGER_GAP = 0.075
CLOSED_FINGER_GAP = 0.0005
DEX_TSHIRT_USD = Path(
    "/home/horizon/DexGarmentLab/Assets/Garment/Tops/"
    "NoCollar_Ssleeve_FrontClose/TNSC_T_Shirt_Short_Sleeve/"
    "TNSC_T_Shirt_Short_Sleeve_obj.usd"
)
DEX_TSHIRT_OBJ = (
    Path(__file__).resolve().parents[2]
    / "genesis/assets/meshes/garments/dexgarmentlab_short_sleeve_tshirt.obj"
)
DEX_TSHIRT_EXPORTER = Path(__file__).resolve().parent / "export_dexgarmentlab_tshirt_asset.py"
ISAACSIM_PYTHON = Path("/home/horizon/isaacsim_env/bin/python")
SHIRT_COLOR = (1.0, 0.86, 0.08, 1.0)
GENESIS_IPC_CLOTH_KWARGS = {
    "E": 6e4,
    "nu": 0.49,
    "rho": 200,
    "thickness": 0.0002,
    "bending_stiffness": 10.0,
    "friction_mu": 2.0,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Scripted Genesis shirt grasp/lift demo.")
    parser.add_argument("--vis", action="store_true", default=False, help="Show the Genesis viewer.")
    parser.add_argument("--record", action="store_true", default=False, help="Record the center camera MP4.")
    parser.add_argument("--output-dir", default="recordings/ipc_dual_piperx_shirt_lift")
    parser.add_argument("--video-name", default="ipc_dual_piperx_shirt_lift.mp4")
    parser.add_argument("--shirt-obj", type=Path, default=DEX_TSHIRT_OBJ)
    parser.add_argument("--shirt-usd", type=Path, default=DEX_TSHIRT_USD)
    parser.add_argument(
        "--refresh-shirt-asset",
        action="store_true",
        default=False,
        help="Regenerate --shirt-obj from --shirt-usd before running.",
    )
    parser.add_argument("--hide-piper", action="store_true", default=False, help="Hide the Piper-X arm visuals.")
    parser.add_argument(
        "--shirt-scale",
        type=float,
        default=0.55,
        help="Uniform scale applied to the DexGarmentLab T-shirt USD mesh.",
    )
    parser.add_argument(
        "--horizon-scale",
        type=float,
        default=1.0,
        help="Scale scripted phase lengths for smoke tests.",
    )
    parser.add_argument("--no-ipc", action="store_true", default=False, help="Disable IPC contacts for ablation.")
    parser.add_argument("--cpu", action="store_true", default=False, help="Force Genesis CPU backend.")
    return parser.parse_args()


def make_piper_urdf(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not PIPER_DUAL_URDF.is_file():
        raise FileNotFoundError(f"Piper-X dual-arm URDF does not exist: {PIPER_DUAL_URDF}")

    source_root = PIPER_DUAL_URDF.parent
    tree = ET.parse(PIPER_DUAL_URDF)
    root = tree.getroot()
    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib.get("filename")
        if filename and not Path(filename).is_absolute():
            mesh.set("filename", str(source_root / filename))

    finger_collision_specs = {
        "left_link7": ("0.000 -0.046 -0.026", "0 0 0"),
        "left_link8": ("0.000 0.046 -0.026", "0 0 0"),
        "right_link7": ("0.000 -0.046 -0.026", "0 0 0"),
        "right_link8": ("0.000 0.046 -0.026", "0 0 0"),
    }
    for link in root.findall("link"):
        link_name = link.attrib.get("name")
        if link_name not in finger_collision_specs:
            continue
        for collision in list(link.findall("collision")):
            link.remove(collision)
        collision = ET.SubElement(link, "collision")
        ET.SubElement(
            collision,
            "origin",
            xyz=finger_collision_specs[link_name][0],
            rpy=finger_collision_specs[link_name][1],
        )
        geometry = ET.SubElement(collision, "geometry")
        ET.SubElement(geometry, "box", size="0.026 0.012 0.080")

    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path


def set_robot_init_qpos(robot, qpos):
    robot._variant_init_qpos = [np.array(qpos, dtype=np.float32, copy=True)]
    for joint in robot.joints:
        if joint.n_qs == 0:
            continue
        start = joint.qs_idx_local[0]
        stop = start + joint.n_qs
        joint._init_qpos = np.array(qpos[start:stop], dtype=np.float32, copy=True)


def export_tshirt_usd_to_obj(usd_path, obj_path, scale):
    usd_path = Path(usd_path).expanduser().resolve()
    obj_path = Path(obj_path).expanduser().resolve()
    if not usd_path.is_file():
        raise FileNotFoundError(f"DexGarmentLab T-shirt USD does not exist: {usd_path}")
    if not ISAACSIM_PYTHON.is_file():
        raise FileNotFoundError(f"Isaac Sim Python with pxr is missing: {ISAACSIM_PYTHON}")
    if not DEX_TSHIRT_EXPORTER.is_file():
        raise FileNotFoundError(f"DexGarmentLab T-shirt exporter is missing: {DEX_TSHIRT_EXPORTER}")

    command = [
        str(ISAACSIM_PYTHON),
        str(DEX_TSHIRT_EXPORTER),
        "--source-usd",
        str(usd_path),
        "--output-obj",
        str(obj_path),
        "--scale",
        str(scale),
    ]
    subprocess.run(command, check=True)
    return obj_path


def resolve_tshirt_obj(args):
    shirt_obj = Path(args.shirt_obj).expanduser().resolve()
    if args.refresh_shirt_asset:
        return export_tshirt_usd_to_obj(args.shirt_usd, shirt_obj, args.shirt_scale)
    if not shirt_obj.is_file():
        raise FileNotFoundError(
            f"T-shirt OBJ asset does not exist: {shirt_obj}. "
            "Run with --refresh-shirt-asset to regenerate it from --shirt-usd."
        )
    return shirt_obj


def make_scene(args, shirt_mesh_path, robot_urdf_path):
    coupler_options = (
        None
        if args.no_ipc
        else gs.options.IPCCouplerOptions(
            constraint_strength_translation=100.0,
            constraint_strength_rotation=100.0,
            n_linesearch_iterations=8,
            newton_tolerance=1e-1,
            newton_translation_tolerance=1,
            newton_semi_implicit_enable=False,
            linear_system_tolerance=1e-3,
            contact_enable=True,
            enable_rigid_rigid_contact=True,
            contact_d_hat=0.001,
            contact_resistance=1e7,
        )
    )

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, substeps=2, gravity=(0.0, 0.0, -9.81)),
        coupler_options=coupler_options,
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.16, -1.30, 0.72),
            camera_lookat=(0.0, -0.48, 0.13),
            camera_fov=42,
            max_FPS=60,
        ),
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        show_viewer=args.vis,
        renderer=gs.renderers.Rasterizer(),
    )

    table_material = (
        gs.materials.Rigid(coup_type="ipc_only", coup_friction=0.4)
        if not args.no_ipc
        else gs.materials.Rigid(friction=0.7)
    )
    shirt_material = gs.materials.FEM.Cloth(**GENESIS_IPC_CLOTH_KWARGS)
    table = scene.add_entity(
        morph=gs.morphs.Box(pos=tuple(TABLE_CENTER), size=tuple(TABLE_SIZE), fixed=True),
        material=table_material,
        surface=gs.surfaces.Default(color=(1.0, 1.0, 1.0, 1.0)),
    )
    shirt = scene.add_entity(
        morph=gs.morphs.Mesh(file=str(shirt_mesh_path), pos=tuple(SHIRT_CENTER), euler=(0.0, 0.0, 0.0)),
        material=shirt_material,
        surface=gs.surfaces.Default(color=SHIRT_COLOR),
    )
    robot = None
    if not args.hide_piper:
        robot = scene.add_entity(
            morph=gs.morphs.URDF(
                file=str(robot_urdf_path),
                pos=ROBOT_ROOT_POS,
                fixed=True,
                merge_fixed_links=False,
                collision=True,
                visualization=True,
            ),
            material=gs.materials.Rigid(needs_coup=False),
        )

    gripper_material = (
        gs.materials.Rigid(
            rho=1500.0,
            friction=1.5,
            coup_type="two_way_soft_constraint",
            coup_friction=4.0,
            sdf_cell_size=0.003,
        )
        if not args.no_ipc
        else gs.materials.Rigid(rho=1500.0, friction=1.5)
    )
    finger_proxies = []
    for finger_name, finger_pos in proxy_state(GRIPPER_START_Y, FINGER_HIGH_Z, OPEN_FINGER_GAP).items():
        finger = scene.add_entity(
            morph=gs.morphs.Box(
                pos=tuple(finger_pos),
                size=tuple(FINGER_SIZE),
                fixed=False,
            ),
            material=gripper_material,
            surface=gs.surfaces.Default(color=(0.05, 0.08, 0.12, 1.0)),
        )
        finger_proxies.append((finger_name, finger))

    center_cam = scene.add_camera(
        res=(960, 540),
        pos=(0.15, -1.34, 0.82),
        lookat=(0.0, -0.44, 0.30),
        fov=52,
        GUI=False,
    )
    left_cam = scene.add_camera(
        res=(640, 480),
        pos=(-0.46, -1.02, 0.50),
        lookat=(-0.12, -0.48, 0.16),
        fov=50,
        GUI=False,
    )
    right_cam = scene.add_camera(
        res=(640, 480),
        pos=(0.46, -1.02, 0.50),
        lookat=(0.12, -0.48, 0.16),
        fov=50,
        GUI=False,
    )
    return scene, table, shirt, robot, finger_proxies, (center_cam, left_cam, right_cam)


def proxy_state(y, z, gap):
    state = {}
    center_offset = 0.5 * float(gap) + 0.5 * float(FINGER_SIZE[1])
    for pair_index, pair_x in enumerate(GRIPPER_PAIR_XS):
        state[f"pair{pair_index}_front"] = np.array([pair_x, y - center_offset, z], dtype=np.float32)
        state[f"pair{pair_index}_back"] = np.array([pair_x, y + center_offset, z], dtype=np.float32)
    return state


def interpolate_proxy_state(start_state, end_state, steps):
    for alpha in np.linspace(0.0, 1.0, steps, endpoint=True):
        yield {
            name: (1.0 - alpha) * start_state[name] + alpha * end_state[name]
            for name in start_state
        }


def offset_proxy_state(state, offset):
    offset_array = np.array(offset, dtype=np.float32)
    return {name: pos + offset_array for name, pos in state.items()}


def shake_proxy_states(base_state, amplitude, cycles, steps_per_half_cycle):
    previous_state = base_state
    for cycle_index in range(cycles * 2):
        direction = -1.0 if cycle_index % 2 else 1.0
        target_state = offset_proxy_state(base_state, (direction * amplitude, 0.0, 0.0))
        yield from interpolate_proxy_state(previous_state, target_state, steps_per_half_cycle)
        previous_state = target_state
    yield from interpolate_proxy_state(previous_state, base_state, steps_per_half_cycle)


def initialize_finger_proxies(finger_proxies, initial_state):
    for finger_name, finger in finger_proxies:
        target_pos = initial_state[finger_name]
        finger.set_qpos(np.array([*target_pos, 1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        finger.set_dofs_kp(np.array([8000.0, 8000.0, 8000.0, 900.0, 900.0, 900.0], dtype=np.float32))
        finger.set_dofs_kv(np.array([220.0, 220.0, 220.0, 80.0, 80.0, 80.0], dtype=np.float32))
        finger.set_dofs_force_range(
            np.array([-450.0, -450.0, -450.0, -60.0, -60.0, -60.0], dtype=np.float32),
            np.array([450.0, 450.0, 450.0, 60.0, 60.0, 60.0], dtype=np.float32),
        )
        finger.control_dofs_position(target_pos, dofs_idx_local=slice(0, 3))
        finger.control_dofs_position(np.zeros(3, dtype=np.float32), dofs_idx_local=slice(3, 6))


def drive_finger_proxies(finger_proxies, target_state):
    for finger_name, finger in finger_proxies:
        finger.control_dofs_position(target_state[finger_name], dofs_idx_local=slice(0, 3))
        finger.control_dofs_position(np.zeros(3, dtype=np.float32), dofs_idx_local=slice(3, 6))


def cloth_stats(shirt):
    if hasattr(shirt, "get_particles_pos"):
        poss = shirt.get_particles_pos().detach().cpu().numpy()
    else:
        state = shirt.get_state()
        poss = state.pos.detach().cpu().numpy()
    if poss.ndim == 3:
        poss = poss[0]
    sentinel_mask = poss[:, 2] < 10.0
    poss = poss[sentinel_mask]
    z_values = poss[:, 2]
    centroid = poss.mean(axis=0)
    return {
        "centroid": centroid,
        "min_z": float(z_values.min()),
        "max_z": float(z_values.max()),
        "sentinel_count": int((~sentinel_mask).sum()),
    }


def record_frame(cameras, record):
    if record:
        for camera in cameras:
            camera.render()


def step_phase(
    scene,
    robot,
    shirt,
    finger_proxies,
    cameras,
    phase,
    proxy_targets,
    record,
    neutral_qpos=None,
    log_interval=20,
):
    for i, proxy_target in enumerate(proxy_targets):
        if robot is not None and neutral_qpos is not None:
            robot.control_dofs_position(neutral_qpos[ALL_CONTROL_DOFS], ALL_CONTROL_DOFS)
        drive_finger_proxies(finger_proxies, proxy_target)
        scene.step()
        record_frame(cameras, record)
        if i == 0 or (i + 1) % log_interval == 0:
            stats = cloth_stats(shirt)
            proxy_z = float(np.mean([target[2] for target in proxy_target.values()]))
            print(
                f"[{phase:>8s}] step={i + 1:04d} "
                f"centroid=({stats['centroid'][0]:+.3f}, {stats['centroid'][1]:+.3f}, {stats['centroid'][2]:+.3f}) "
                f"z_min={stats['min_z']:.3f} z_max={stats['max_z']:.3f} "
                f"finger_z={proxy_z:.3f} sentinel={stats['sentinel_count']}"
            )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shirt_mesh_path = resolve_tshirt_obj(args)
    robot_urdf_path = make_piper_urdf(output_dir / "piper_x_dualarm_genesis.urdf")

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, logging_level="info")
    scene, _, shirt, robot, finger_proxies, cameras = make_scene(args, shirt_mesh_path, robot_urdf_path)

    neutral_qpos = np.zeros(16, dtype=np.float32)
    neutral_qpos[[2, 3]] = 0.70
    neutral_qpos[[4, 5]] = -0.90
    neutral_qpos[ALL_GRIPPER_DOFS] = OPEN_GRIPPER
    if robot is not None:
        set_robot_init_qpos(robot, neutral_qpos)
    scene.build()

    if robot is not None:
        robot.set_dofs_kp(np.array([4500.0] * 12 + [800.0] * 4, dtype=np.float32), ALL_CONTROL_DOFS)
        robot.set_dofs_kv(np.array([450.0] * 12 + [80.0] * 4, dtype=np.float32), ALL_CONTROL_DOFS)
        robot.set_dofs_force_range(
            np.array([-87.0] * 12 + [-300.0] * 4, dtype=np.float32),
            np.array([87.0] * 12 + [300.0] * 4, dtype=np.float32),
            ALL_CONTROL_DOFS,
        )

    high_open_proxy = proxy_state(GRIPPER_START_Y, FINGER_HIGH_Z, OPEN_FINGER_GAP)
    approach_open_proxy = proxy_state(GRIPPER_CONTACT_Y, FINGER_HIGH_Z, OPEN_FINGER_GAP)
    low_open_proxy = proxy_state(GRIPPER_CONTACT_Y, FINGER_CONTACT_Z, OPEN_FINGER_GAP)
    low_closed_proxy = proxy_state(GRIPPER_CONTACT_Y, FINGER_CONTACT_Z, CLOSED_FINGER_GAP)
    pushed_closed_proxy = proxy_state(GRIPPER_PUSH_Y, FINGER_CONTACT_Z, CLOSED_FINGER_GAP)
    lift_proxy = proxy_state(GRIPPER_PUSH_Y, FINGER_LIFT_Z, CLOSED_FINGER_GAP)
    release_proxy = proxy_state(GRIPPER_PUSH_Y, FINGER_LIFT_Z, OPEN_FINGER_GAP)
    second_low_open_proxy = proxy_state(GRIPPER_PUSH_Y, FINGER_NEAR_TABLE_Z, OPEN_FINGER_GAP)
    second_low_closed_proxy = proxy_state(GRIPPER_PUSH_Y, FINGER_NEAR_TABLE_Z, CLOSED_FINGER_GAP)
    second_lift_proxy = proxy_state(GRIPPER_PUSH_Y, FINGER_LIFT_Z, CLOSED_FINGER_GAP)
    second_release_proxy = proxy_state(GRIPPER_PUSH_Y, FINGER_LIFT_Z, OPEN_FINGER_GAP)
    retreat_proxy = proxy_state(GRIPPER_START_Y, FINGER_LIFT_Z, OPEN_FINGER_GAP)
    initialize_finger_proxies(finger_proxies, high_open_proxy)

    phase_steps = {
        "settle": max(1, int(40 * args.horizon_scale)),
        "approach": max(2, int(60 * args.horizon_scale)),
        "lower": max(2, int(70 * args.horizon_scale)),
        "close": max(2, int(55 * args.horizon_scale)),
        "hold": max(1, int(35 * args.horizon_scale)),
        "push": max(2, int(45 * args.horizon_scale)),
        "lift": max(2, int(150 * args.horizon_scale)),
        "high_hold": max(1, int(45 * args.horizon_scale)),
        "shake": max(2, int(18 * args.horizon_scale)),
        "release": max(2, int(35 * args.horizon_scale)),
        "second_lower": max(2, int(85 * args.horizon_scale)),
        "second_close": max(2, int(55 * args.horizon_scale)),
        "second_hold": max(1, int(35 * args.horizon_scale)),
        "second_lift": max(2, int(140 * args.horizon_scale)),
        "second_release": max(2, int(35 * args.horizon_scale)),
        "retreat": max(2, int(45 * args.horizon_scale)),
    }

    if args.record:
        for camera in cameras:
            camera.start_recording()

    video_path = output_dir / args.video_name
    try:
        if robot is not None:
            robot.set_qpos(neutral_qpos)
        step_phase(
            scene,
            robot,
            shirt,
            finger_proxies,
            cameras,
            "settle",
            [high_open_proxy] * phase_steps["settle"],
            args.record,
            neutral_qpos,
        )
        step_phase(
            scene,
            robot,
            shirt,
            finger_proxies,
            cameras,
            "approach",
            interpolate_proxy_state(high_open_proxy, approach_open_proxy, phase_steps["approach"]),
            args.record,
            neutral_qpos,
        )
        step_phase(
            scene,
            robot,
            shirt,
            finger_proxies,
            cameras,
            "lower",
            interpolate_proxy_state(approach_open_proxy, low_open_proxy, phase_steps["lower"]),
            args.record,
            neutral_qpos,
        )
        step_phase(
            scene,
            robot,
            shirt,
            finger_proxies,
            cameras,
            "close",
            interpolate_proxy_state(low_open_proxy, low_closed_proxy, phase_steps["close"]),
            args.record,
            neutral_qpos,
        )
        step_phase(
            scene,
            robot,
            shirt,
            finger_proxies,
            cameras,
            "hold",
            [low_closed_proxy] * phase_steps["hold"],
            args.record,
            neutral_qpos,
        )
        step_phase(
            scene,
            robot,
            shirt,
            finger_proxies,
            cameras,
            "push",
            interpolate_proxy_state(low_closed_proxy, pushed_closed_proxy, phase_steps["push"]),
            args.record,
            neutral_qpos,
        )
        step_phase(
            scene,
            robot,
            shirt,
            finger_proxies,
            cameras,
            "lift",
            interpolate_proxy_state(pushed_closed_proxy, lift_proxy, phase_steps["lift"]),
            args.record,
            neutral_qpos,
        )
        step_phase(
            scene,
            robot,
            shirt,
            finger_proxies,
            cameras,
            "hi_hold",
            [lift_proxy] * phase_steps["high_hold"],
            args.record,
            neutral_qpos,
        )
        step_phase(
            scene,
            robot,
            shirt,
            finger_proxies,
            cameras,
            "shake",
            shake_proxy_states(lift_proxy, SHAKE_X_OFFSET, 2, phase_steps["shake"]),
            args.record,
            neutral_qpos,
        )
        step_phase(
            scene,
            robot,
            shirt,
            finger_proxies,
            cameras,
            "release",
            interpolate_proxy_state(lift_proxy, release_proxy, phase_steps["release"]),
            args.record,
            neutral_qpos,
        )
        step_phase(
            scene,
            robot,
            shirt,
            finger_proxies,
            cameras,
            "re_lower",
            interpolate_proxy_state(release_proxy, second_low_open_proxy, phase_steps["second_lower"]),
            args.record,
            neutral_qpos,
        )
        step_phase(
            scene,
            robot,
            shirt,
            finger_proxies,
            cameras,
            "re_close",
            interpolate_proxy_state(second_low_open_proxy, second_low_closed_proxy, phase_steps["second_close"]),
            args.record,
            neutral_qpos,
        )
        step_phase(
            scene,
            robot,
            shirt,
            finger_proxies,
            cameras,
            "re_hold",
            [second_low_closed_proxy] * phase_steps["second_hold"],
            args.record,
            neutral_qpos,
        )
        step_phase(
            scene,
            robot,
            shirt,
            finger_proxies,
            cameras,
            "re_lift",
            interpolate_proxy_state(second_low_closed_proxy, second_lift_proxy, phase_steps["second_lift"]),
            args.record,
            neutral_qpos,
        )
        step_phase(
            scene,
            robot,
            shirt,
            finger_proxies,
            cameras,
            "re_rel",
            interpolate_proxy_state(second_lift_proxy, second_release_proxy, phase_steps["second_release"]),
            args.record,
            neutral_qpos,
        )
        step_phase(
            scene,
            robot,
            shirt,
            finger_proxies,
            cameras,
            "retreat",
            interpolate_proxy_state(second_release_proxy, retreat_proxy, phase_steps["retreat"]),
            args.record,
            neutral_qpos,
        )
    finally:
        if args.record:
            cameras[0].stop_recording(save_to_filename=str(video_path), fps=60)
            for index, camera in enumerate(cameras[1:], start=1):
                camera.stop_recording(save_to_filename=str(output_dir / f"camera_{index}.mp4"), fps=60)
            print(f"Saved center-camera recording to {video_path}")

    final_stats = cloth_stats(shirt)
    initial_z = SHIRT_CENTER[2]
    print(
        "final cloth stats: "
        f"centroid_z={final_stats['centroid'][2]:.4f}, "
        f"max_z={final_stats['max_z']:.4f}, "
        f"sentinel_count={final_stats['sentinel_count']}, "
        f"max_lift_over_initial={final_stats['max_z'] - initial_z:.4f}"
    )


if __name__ == "__main__":
    main()
