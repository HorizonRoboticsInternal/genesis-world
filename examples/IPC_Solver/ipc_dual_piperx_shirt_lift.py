import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

import genesis as gs


PIPER_DUAL_URDF = "/home/horizon/newton_cloth/piper_x_description_dualarm.urdf"

LEFT_ARM_DOFS = [0, 2, 4, 6, 8, 10]
RIGHT_ARM_DOFS = [1, 3, 5, 7, 9, 11]
LEFT_GRIPPER_DOFS = [12, 13]
RIGHT_GRIPPER_DOFS = [14, 15]
ALL_ARM_DOFS = LEFT_ARM_DOFS + RIGHT_ARM_DOFS
ALL_GRIPPER_DOFS = LEFT_GRIPPER_DOFS + RIGHT_GRIPPER_DOFS
ALL_CONTROL_DOFS = ALL_ARM_DOFS + ALL_GRIPPER_DOFS

OPEN_GRIPPER = np.array([0.045, -0.045, 0.045, -0.045], dtype=np.float32)
CLOSED_GRIPPER = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

TABLE_CENTER = np.array([0.0, -0.48, 0.065], dtype=np.float32)
TABLE_SIZE = np.array([0.82, 0.72, 0.13], dtype=np.float32)
TABLE_TOP_Z = TABLE_CENTER[2] + TABLE_SIZE[2] * 0.5
SHIRT_CENTER = np.array([0.0, -0.48, TABLE_TOP_Z + 0.010], dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="Dual Piper-X scripted IPC shirt grasp/lift demo.")
    parser.add_argument("--vis", action="store_true", default=False, help="Show the Genesis viewer.")
    parser.add_argument("--record", action="store_true", default=False, help="Record the center camera MP4.")
    parser.add_argument("--output-dir", default="recordings/ipc_dual_piperx_shirt_lift")
    parser.add_argument("--video-name", default="ipc_dual_piperx_shirt_lift.mp4")
    parser.add_argument("--mesh-name", default="genesis_short_sleeve_shirt.obj")
    parser.add_argument("--horizon-scale", type=float, default=1.0, help="Scale scripted phase lengths for smoke tests.")
    parser.add_argument(
        "--cloth-mode",
        default="pbd",
        choices=["pbd", "ipc"],
        help="Use PBD particle attachment for verified lift, or IPC contact-only for ablations.",
    )
    parser.add_argument(
        "--coup-type",
        default="two_way_soft_constraint",
        choices=["two_way_soft_constraint", "external_articulation"],
        help="IPC rigid/FEM coupling mode for the Piper articulation.",
    )
    parser.add_argument("--no-ipc", action="store_true", default=False, help="Disable IPC contacts for ablation.")
    parser.add_argument("--hide-piper", action="store_true", default=False, help="Skip the Piper URDF import for fast cloth verification.")
    parser.add_argument("--cpu", action="store_true", default=False, help="Force Genesis CPU backend.")
    return parser.parse_args()


def make_short_sleeve_shirt_obj(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    xs = np.linspace(-0.34, 0.34, 35)
    ys = np.linspace(-0.25, 0.22, 25)
    vertices = []
    vertex_by_grid = {}
    faces = []

    def inside_shirt(x, y):
        in_torso = -0.17 <= x <= 0.17 and -0.25 <= y <= 0.17
        in_left_sleeve = -0.34 <= x <= -0.17 and 0.00 <= y <= 0.16
        in_right_sleeve = 0.17 <= x <= 0.34 and 0.00 <= y <= 0.16
        in_neck_notch = (x / 0.070) ** 2 + ((y - 0.18) / 0.055) ** 2 < 1.0
        return (in_torso or in_left_sleeve or in_right_sleeve) and not in_neck_notch

    def vertex_index(i, j):
        key = (i, j)
        if key not in vertex_by_grid:
            vertex_by_grid[key] = len(vertices) + 1
            vertices.append((xs[i], ys[j], 0.0))
        return vertex_by_grid[key]

    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            cell_x = 0.5 * (xs[i] + xs[i + 1])
            cell_y = 0.5 * (ys[j] + ys[j + 1])
            if inside_shirt(cell_x, cell_y):
                v00 = vertex_index(i, j)
                v10 = vertex_index(i + 1, j)
                v11 = vertex_index(i + 1, j + 1)
                v01 = vertex_index(i, j + 1)
                faces.append((v00, v10, v11))
                faces.append((v00, v11, v01))

    with path.open("w", encoding="utf-8") as f:
        f.write("# Generated IPC shell mesh for the dual Piper-X Genesis shirt lift demo.\n")
        for vx, vy, vz in vertices:
            f.write(f"v {vx:.6f} {vy:.6f} {vz:.6f}\n")
        for a, b, c in faces:
            f.write(f"f {a} {b} {c}\n")

    return path


def make_ipc_piper_urdf(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    source_root = Path(PIPER_DUAL_URDF).parent
    tree = ET.parse(PIPER_DUAL_URDF)
    root = tree.getroot()

    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib.get("filename")
        if filename and not filename.startswith("/"):
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
        ET.SubElement(collision, "origin", xyz=finger_collision_specs[link_name][0], rpy=finger_collision_specs[link_name][1])
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


def make_scene(args, shirt_mesh_path, robot_urdf_path):
    coupler_options = None
    use_ipc = args.cloth_mode == "ipc" and not args.no_ipc
    if use_ipc:
        coupler_options = gs.options.IPCCouplerOptions(
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

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, substeps=1, gravity=(0, 0, -9.81)),
        coupler_options=coupler_options,
        pbd_options=gs.options.PBDOptions(particle_size=0.01),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.15, -1.55, 0.78),
            camera_lookat=(0.0, -0.48, 0.13),
            camera_fov=42,
            max_FPS=60,
        ),
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        show_viewer=args.vis,
        renderer=gs.renderers.Rasterizer(),
    )

    table_material = gs.materials.Rigid(coup_type="ipc_only") if use_ipc else gs.materials.Rigid()
    shirt_material = (
        gs.materials.FEM.Cloth(
            rho=200.0,
            thickness=0.0015,
            E=1.0e4,
            nu=0.30,
            bending_stiffness=10.0,
            friction_mu=1.0,
        )
        if use_ipc
        else gs.materials.PBD.Cloth()
    )
    table = scene.add_entity(
        morph=gs.morphs.Box(pos=tuple(TABLE_CENTER), size=tuple(TABLE_SIZE), fixed=True),
        material=table_material,
        surface=gs.surfaces.Default(color=(1.0, 1.0, 1.0, 1.0)),
    )
    shirt = scene.add_entity(
        morph=gs.morphs.Mesh(file=str(shirt_mesh_path), pos=tuple(SHIRT_CENTER), euler=(0.0, 0.0, 0.0)),
        material=shirt_material,
        surface=gs.surfaces.Default(color=(1.0, 0.86, 0.08, 1.0)),
    )
    robot = None
    if not args.hide_piper:
        robot = scene.add_entity(
            morph=gs.morphs.URDF(
                file=str(robot_urdf_path),
                fixed=True,
                merge_fixed_links=False,
                collision=True,
                visualization=True,
            ),
            material=gs.materials.Rigid(needs_coup=False),
        )
    finger_material = gs.materials.Rigid(coup_type="two_way_soft_constraint", coup_friction=1.0) if use_ipc else gs.materials.Rigid()
    finger_surface = gs.surfaces.Default(color=(0.08, 0.08, 0.08, 1.0))
    finger_size = (0.012, 0.105, 0.060)
    finger_proxies = [
        scene.add_entity(gs.morphs.Box(pos=(-0.205, -0.48, 0.20), size=finger_size, fixed=not use_ipc), finger_material, finger_surface),
        scene.add_entity(gs.morphs.Box(pos=(-0.115, -0.48, 0.20), size=finger_size, fixed=not use_ipc), finger_material, finger_surface),
        scene.add_entity(gs.morphs.Box(pos=(0.115, -0.48, 0.20), size=finger_size, fixed=not use_ipc), finger_material, finger_surface),
        scene.add_entity(gs.morphs.Box(pos=(0.205, -0.48, 0.20), size=finger_size, fixed=not use_ipc), finger_material, finger_surface),
    ]
    center_cam = scene.add_camera(
        res=(960, 540),
        pos=(0.12, -1.24, 0.62),
        lookat=(0.0, -0.48, 0.15),
        fov=42,
        GUI=False,
    )
    left_cam = scene.add_camera(
        res=(640, 480),
        pos=(-0.46, -0.92, 0.47),
        lookat=(-0.12, -0.48, 0.16),
        fov=50,
        GUI=False,
    )
    right_cam = scene.add_camera(
        res=(640, 480),
        pos=(0.46, -0.92, 0.47),
        lookat=(0.12, -0.48, 0.16),
        fov=50,
        GUI=False,
    )
    return scene, table, shirt, robot, finger_proxies, (center_cam, left_cam, right_cam)


def qpos_with_gripper(qpos, gripper):
    qpos_with_target = np.asarray(qpos, dtype=np.float32).copy()
    qpos_with_target[ALL_GRIPPER_DOFS] = gripper
    return qpos_with_target


def solve_dual_ik(robot, left_pos, right_pos, init_qpos):
    left_link6 = robot.get_link("left_link6")
    right_link6 = robot.get_link("right_link6")
    qpos = robot.inverse_kinematics_multilink(
        links=[left_link6, right_link6],
        poss=[np.array(left_pos, gs.np_float), np.array(right_pos, gs.np_float)],
        dofs_idx_local=ALL_ARM_DOFS,
        init_qpos=init_qpos,
        respect_joint_limit=True,
        max_samples=12,
        max_solver_iters=80,
        damping=0.02,
        pos_mask=[True, True, True],
        rot_mask=[False, False, False],
    )
    return qpos_with_gripper(qpos, init_qpos[ALL_GRIPPER_DOFS])


def interpolate_qpos(start_qpos, end_qpos, steps):
    for alpha in np.linspace(0.0, 1.0, steps, endpoint=True):
        yield (1.0 - alpha) * start_qpos + alpha * end_qpos


def proxy_state(left_x, right_x, z, gap):
    return np.array([left_x, right_x, z, gap], dtype=np.float32)


def interpolate_proxy_state(start_state, end_state, steps):
    for alpha in np.linspace(0.0, 1.0, steps, endpoint=True):
        yield (1.0 - alpha) * start_state + alpha * end_state


def set_finger_proxies(finger_proxies, state):
    left_x, right_x, z, gap = state
    y = SHIRT_CENTER[1]
    x_positions = (left_x - gap * 0.5, left_x + gap * 0.5, right_x - gap * 0.5, right_x + gap * 0.5)
    for finger, x in zip(finger_proxies, x_positions, strict=True):
        finger.set_pos((float(x), float(y), float(z)))


def proxy_inner_anchors(state):
    left_x, right_x, z, gap = state
    y = SHIRT_CENTER[1]
    return {
        "left": np.array([left_x + gap * 0.5, y, z], dtype=np.float32),
        "right": np.array([right_x - gap * 0.5, y, z], dtype=np.float32),
    }


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


def select_particles_near(shirt, xy, count):
    if hasattr(shirt, "_particles"):
        positions = np.asarray(shirt._particles, dtype=np.float32)
    else:
        positions = shirt.init_positions.detach().cpu().numpy()
    distances = np.linalg.norm(positions[:, :2] - np.asarray(xy, dtype=np.float32), axis=1)
    return np.argsort(distances)[:count].astype(np.int32).tolist()


def attach_shirt_to_finger_proxies(shirt, finger_proxies):
    left_particles = select_particles_near(shirt, (-0.16, SHIRT_CENTER[1]), 20)
    right_particles = select_particles_near(shirt, (0.16, SHIRT_CENTER[1]), 20)
    current_pos = shirt.get_particles_pos().detach().cpu().numpy()
    anchors = proxy_inner_anchors(proxy_state(-0.16, 0.16, TABLE_TOP_Z + 0.024, 0.018))
    attached_patches = {
        "left": {
            "indices": left_particles,
            "offsets": current_pos[left_particles] - anchors["left"],
        },
        "right": {
            "indices": right_particles,
            "offsets": current_pos[right_particles] - anchors["right"],
        },
    }
    print(
        "attached shirt particles: "
        f"left_count={len(left_particles)} left_link={finger_proxies[1].link_start} "
        f"right_count={len(right_particles)} right_link={finger_proxies[2].link_start}"
    )
    return attached_patches


def drive_attached_particles(shirt, proxy_target, attached_patches):
    if not attached_patches:
        return
    anchors = proxy_inner_anchors(proxy_target)
    for side, patch in attached_patches.items():
        target_positions = anchors[side] + patch["offsets"]
        shirt.set_particles_pos(target_positions, particles_idx_local=patch["indices"])


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
    qpos_targets,
    proxy_targets,
    record,
    attached_patches=None,
    log_interval=20,
):
    for i, (qpos, proxy_target) in enumerate(zip(qpos_targets, proxy_targets, strict=True)):
        set_finger_proxies(finger_proxies, proxy_target)
        drive_attached_particles(shirt, proxy_target, attached_patches)
        if robot is not None:
            robot.control_dofs_position(qpos[ALL_CONTROL_DOFS], ALL_CONTROL_DOFS)
        scene.step()
        record_frame(cameras, record)
        if i == 0 or (i + 1) % log_interval == 0:
            stats = cloth_stats(shirt)
            print(
                f"[{phase:>8s}] step={i + 1:04d} "
                f"centroid=({stats['centroid'][0]:+.3f}, {stats['centroid'][1]:+.3f}, {stats['centroid'][2]:+.3f}) "
                f"z_min={stats['min_z']:.3f} z_max={stats['max_z']:.3f} sentinel={stats['sentinel_count']}"
            )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shirt_mesh_path = make_short_sleeve_shirt_obj(output_dir / args.mesh_name)
    robot_urdf_path = make_ipc_piper_urdf(output_dir / "piper_x_dualarm_ipc.urdf")

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

    approach_qpos = np.array(
        [
            -1.8495,
            0.6854,
            1.9616,
            0.9100,
            -1.3592,
            -0.4323,
            -0.2634,
            0.1435,
            0.8933,
            -0.2047,
            0.0,
            0.0,
            0.045,
            -0.045,
            0.045,
            -0.045,
        ],
        dtype=np.float32,
    )
    lower_qpos = approach_qpos.copy()
    closed_qpos = qpos_with_gripper(lower_qpos, CLOSED_GRIPPER)
    lift_qpos = closed_qpos.copy()
    lift_qpos[[2, 3]] = np.array([1.35, 1.35], dtype=np.float32)
    lift_qpos[[4, 5]] = np.array([-1.10, -1.10], dtype=np.float32)
    retreat_qpos = lift_qpos.copy()
    retreat_qpos[[0, 1]] = np.array([-1.30, 1.30], dtype=np.float32)

    phase_steps = {
        "settle": max(1, int(40 * args.horizon_scale)),
        "approach": max(2, int(70 * args.horizon_scale)),
        "lower": max(2, int(70 * args.horizon_scale)),
        "close": max(2, int(50 * args.horizon_scale)),
        "hold": max(1, int(35 * args.horizon_scale)),
        "lift": max(2, int(90 * args.horizon_scale)),
        "retreat": max(2, int(45 * args.horizon_scale)),
    }

    if args.record:
        for camera in cameras:
            camera.start_recording()

    high_proxy = proxy_state(-0.16, 0.16, 0.32, 0.090)
    low_open_proxy = proxy_state(-0.16, 0.16, TABLE_TOP_Z + 0.024, 0.090)
    low_closed_proxy = proxy_state(-0.16, 0.16, TABLE_TOP_Z + 0.024, 0.018)
    lift_proxy = proxy_state(-0.16, 0.16, 0.31, 0.018)
    retreat_proxy = proxy_state(-0.22, 0.22, 0.31, 0.018)

    video_path = output_dir / args.video_name
    attached_patches = None
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
            [neutral_qpos] * phase_steps["settle"],
            [high_proxy] * phase_steps["settle"],
            args.record,
        )
        step_phase(
            scene,
            robot,
            shirt,
            finger_proxies,
            cameras,
            "approach",
            interpolate_qpos(neutral_qpos, approach_qpos, phase_steps["approach"]),
            interpolate_proxy_state(high_proxy, high_proxy, phase_steps["approach"]),
            args.record,
        )
        step_phase(
            scene,
            robot,
            shirt,
            finger_proxies,
            cameras,
            "lower",
            interpolate_qpos(approach_qpos, lower_qpos, phase_steps["lower"]),
            interpolate_proxy_state(high_proxy, low_open_proxy, phase_steps["lower"]),
            args.record,
        )
        step_phase(
            scene,
            robot,
            shirt,
            finger_proxies,
            cameras,
            "close",
            interpolate_qpos(lower_qpos, closed_qpos, phase_steps["close"]),
            interpolate_proxy_state(low_open_proxy, low_closed_proxy, phase_steps["close"]),
            args.record,
        )
        if args.cloth_mode == "pbd":
            attached_patches = attach_shirt_to_finger_proxies(shirt, finger_proxies)
        step_phase(
            scene,
            robot,
            shirt,
            finger_proxies,
            cameras,
            "hold",
            [closed_qpos] * phase_steps["hold"],
            [low_closed_proxy] * phase_steps["hold"],
            args.record,
            attached_patches,
        )
        step_phase(
            scene,
            robot,
            shirt,
            finger_proxies,
            cameras,
            "lift",
            interpolate_qpos(closed_qpos, lift_qpos, phase_steps["lift"]),
            interpolate_proxy_state(low_closed_proxy, lift_proxy, phase_steps["lift"]),
            args.record,
            attached_patches,
        )
        step_phase(
            scene,
            robot,
            shirt,
            finger_proxies,
            cameras,
            "retreat",
            interpolate_qpos(lift_qpos, retreat_qpos, phase_steps["retreat"]),
            interpolate_proxy_state(lift_proxy, retreat_proxy, phase_steps["retreat"]),
            args.record,
            attached_patches,
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
