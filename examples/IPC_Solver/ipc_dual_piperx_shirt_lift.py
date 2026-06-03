import argparse
import math
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

SIM_DT = 0.02
SIM_SUBSTEPS = 4
RECORDING_FPS = int(round(1.0 / SIM_DT))
CFE_OPENING = 0.035
CLOSED_OPENING = 0.0


def gripper_qpos(opening):
    return np.array([opening, -opening, opening, -opening], dtype=np.float32)


def zero_initial_qpos(closed_opening):
    qpos = np.zeros(16, dtype=np.float32)
    qpos[ALL_GRIPPER_DOFS] = gripper_qpos(closed_opening)
    return qpos


OPEN_GRIPPER = gripper_qpos(CFE_OPENING)
CLOSED_GRIPPER = gripper_qpos(CLOSED_OPENING)
ZERO_INITIAL_QPOS = zero_initial_qpos(CLOSED_OPENING)

TABLE_CENTER = np.array([0.0, -0.48, 0.065], dtype=np.float32)
TABLE_SIZE = np.array([1.64, 0.72, 0.13], dtype=np.float32)
TABLE_TOP_Z = TABLE_CENTER[2] + TABLE_SIZE[2] * 0.5
ROOM_WALL_DISTANCE_FROM_TABLE = 1.0
ROOM_WALL_THICKNESS = 0.04
ROOM_WALL_CORNER_GAP = 0.01
ROOM_CEILING_Z = 3.0
ROOM_WALL_COLOR = (0.72, 0.74, 0.74, 1.0)
ROOM_LIGHT_PANEL_COLOR = (0.92, 0.92, 0.86, 1.0)
SHIRT_CENTER = np.array([0.0, -0.48, TABLE_TOP_Z + 0.010], dtype=np.float32)
PIPER_SHOULDER_CENTER_Z_IN_ROOT = 0.123
REAL_SHOULDER_CENTER_ABOVE_TABLE = 0.130
ROBOT_ROOT_Z = TABLE_TOP_Z + REAL_SHOULDER_CENTER_ABOVE_TABLE - PIPER_SHOULDER_CENTER_Z_IN_ROOT
ROBOT_ROOT_POS = (-0.30, float(TABLE_CENTER[1] - TABLE_SIZE[1] * 0.5 - 0.02), float(ROBOT_ROOT_Z))
ROBOT_ROOT_EULER = (0.0, 0.0, 90.0)
CFE_APPROACH_HEIGHT = 0.250
CFE_FINGERTIP_CONTACT_HEIGHT = 0.220
CFE_PUSH_DISTANCE = 0.200
CFE_LIFT_HEIGHT = 0.460
GENESIS_FINGERTIP_CONTACT_HEIGHT = TABLE_TOP_Z + 0.018
PIPER_BASE_START_Y = TABLE_CENTER[1] - TABLE_SIZE[1] * 0.5 - 0.06
PIPER_BASE_CONTACT_Y = SHIRT_CENTER[1]
PIPER_BASE_PUSH_Y = SHIRT_CENTER[1] + CFE_PUSH_DISTANCE
PIPER_BASE_HIGH_Z = CFE_APPROACH_HEIGHT
PIPER_BASE_CONTACT_Z = GENESIS_FINGERTIP_CONTACT_HEIGHT
PIPER_BASE_LIFT_Z = CFE_LIFT_HEIGHT
PIPER_GRASP_BASE_XS = (-0.10, 0.10)
PIPER_IK_LINKS = ("left_link7", "right_link8")
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
GRIPPER_FINGER_COVER_STL = Path("/home/horizon/gripper_finger_cover.stl")
GRIPPER_FINGER_COVER_SCALE = (0.001, 0.001, 0.001)
GRIPPER_FINGER_COVER_MESH_MIN = (-0.02976555443, -0.09951779938, 0.02910915947)
GRIPPER_FINGER_COVER_MESH_MAX = (0.01394346905, -0.02951779938, 0.05152915955)
GRIPPER_FINGER_COVER_EXTENSION = 0.020
GRIPPER_FINGER_BASE_CENTER_Y = 0.033
GRIPPER_FINGER_COVER_CENTER_SHIFT = 0.5 * GRIPPER_FINGER_COVER_EXTENSION
GRIPPER_FINGER_COVER_BASE_FORWARD_OFFSET = 0.040
GRIPPER_FINGER_COVER_CLOSING_SHIFT = 0.003
GRIPPER_FINGER_LINK7_BASE_Y = 0.004968106
GRIPPER_FINGER_LINK8_BASE_Y = -0.004968106
GRIPPER_FINGER_LINK7_COVER_BASE_Y = GRIPPER_FINGER_LINK7_BASE_Y - GRIPPER_FINGER_COVER_BASE_FORWARD_OFFSET
GRIPPER_FINGER_LINK8_COVER_BASE_Y = GRIPPER_FINGER_LINK8_BASE_Y + GRIPPER_FINGER_COVER_BASE_FORWARD_OFFSET
GRIPPER_FINGER_LINK7_COLLISION_CENTER = (
    0.0,
    -(GRIPPER_FINGER_BASE_CENTER_Y + GRIPPER_FINGER_COVER_CENTER_SHIFT),
    -0.012 + GRIPPER_FINGER_COVER_CLOSING_SHIFT,
)
GRIPPER_FINGER_LINK8_COLLISION_CENTER = (
    0.0,
    GRIPPER_FINGER_BASE_CENTER_Y + GRIPPER_FINGER_COVER_CENTER_SHIFT,
    -0.012 + GRIPPER_FINGER_COVER_CLOSING_SHIFT,
)
# Piper-X finger links are rotated relative to the gripper base; link-local Y is
# the useful finger length in Genesis, while link-local Z is the
# closing-direction thickness. Keep the boxes thin enough to avoid closed-finger
# self-intersections, similar to Panda's non-overlap fingertip pad boxes.
GENESIS_FINGER_BODY_COLLISION_BOX_SIZE = (0.020, 0.065, 0.010)
GENESIS_LINK7_BODY_COLLISION_CENTER = (0.0, -0.027, -0.018)
GENESIS_LINK8_BODY_COLLISION_CENTER = (0.0, 0.027, -0.018)
GENESIS_FINGER_COVER_COLLISION_BOX_SIZE = (0.044, 0.105, 0.014)
GENESIS_LINK7_COVER_COLLISION_CENTER = GRIPPER_FINGER_LINK7_COLLISION_CENTER
GENESIS_LINK8_COVER_COLLISION_CENTER = GRIPPER_FINGER_LINK8_COLLISION_CENTER
PIPER_IK_LOCAL_POINTS = (GENESIS_LINK7_COVER_COLLISION_CENTER, GENESIS_LINK8_COVER_COLLISION_CENTER)
GRIPPER_FINGER_LINK7_COVER_RPY = (0.0, math.pi, math.pi)
GRIPPER_FINGER_LINK8_COVER_RPY = (0.0, math.pi, 0.0)
PIPER_COUPLED_FINGER_LINKS = ("left_link7", "left_link8", "right_link7", "right_link8")
PIPER_IPC_COUP_FRICTION = 12.0
PIPER_GRIPPER_KP = 2600.0
PIPER_GRIPPER_KV = 260.0
PIPER_GRIPPER_FORCE_LIMIT = 1200.0
GENESIS_IPC_CLOTH_KWARGS = {
    "E": 6e4,
    "nu": 0.49,
    "rho": 200,
    "thickness": 0.0002,
    "bending_stiffness": 10.0,
    "friction_mu": 2.0,
}
PIPERX_WRIST_CAMERA_ORIGINS = {
    "left": {
        "xyz": "-0.0096489170911759 -0.08009372951791657 0.04279548930003773",
        "rpy": "-0.017085131075935567 -1.225237413769258 1.5698794493564652",
    },
    "right": {
        "xyz": "-0.0076363572782560665 -0.07947460457157493 0.043216980311924016",
        "rpy": "-0.002445405606400719 -1.216058912088991 1.5665889686029582",
    },
}
ROBOTWIN_HEAD_CAMERA_LOCAL_POSITION = np.array(
    [-0.007383059883329407, -0.31715773707478656, 0.6036358425132415],
    dtype=np.float32,
)
ROBOTWIN_HEAD_CAMERA_POSITION = ROBOTWIN_HEAD_CAMERA_LOCAL_POSITION + np.array(
    [0.0, ROBOT_ROOT_POS[1], ROBOT_ROOT_Z],
    dtype=np.float32,
)
ROBOTWIN_HEAD_CAMERA_QUAT_XYZW = np.array(
    [-0.6620275905116694, 0.6913463014325211, -0.2114074909007769, 0.19765281097906265],
    dtype=np.float32,
)
ROBOTWIN_HEAD_CAMERA_FOVY = 44.23872564716461
ROBOTWIN_CAMERA_RES = (392, 252)
WRIST_CAMERA_RES = (640, 480)
WRIST_CAMERA_FOVY = 50.0
WRIST_CAMERA_LINK_OFFSET = np.array(
    [
        [0.0, 0.0, -1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Scripted Genesis shirt grasp/lift demo.")
    parser.add_argument("--vis", action="store_true", default=False, help="Show the Genesis viewer.")
    parser.add_argument("--record", action="store_true", default=False, help="Record the three camera MP4s.")
    parser.add_argument("--output-dir", default="recordings/ipc_dual_piperx_shirt_lift")
    parser.add_argument("--video-name", default="ipc_dual_piperx_shirt_lift.mp4")
    parser.add_argument("--combined-video-name", default="left_mid_right.mp4")
    parser.add_argument("--shirt-obj", type=Path, default=DEX_TSHIRT_OBJ)
    parser.add_argument("--shirt-usd", type=Path, default=DEX_TSHIRT_USD)
    parser.add_argument(
        "--refresh-shirt-asset",
        action="store_true",
        default=False,
        help="Regenerate --shirt-obj from --shirt-usd before running.",
    )
    parser.add_argument(
        "--show-piper",
        action="store_true",
        default=True,
        help="Show and drive dual Piper-X arms along the table front edge.",
    )
    parser.add_argument("--hide-piper", action="store_true", default=False, help=argparse.SUPPRESS)
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
    parser.add_argument(
        "--focus-grasp",
        action="store_true",
        default=False,
        help="When recording, render only lower/contact/close/push/lift/release phases for faster grasp tuning.",
    )
    parser.add_argument(
        "--closed-opening",
        type=float,
        default=CLOSED_OPENING,
        help="Residual per-finger closed gripper opening, in meters.",
    )
    parser.add_argument(
        "--init-only",
        action="store_true",
        default=False,
        help="Build the scene and exit; useful for fast IPC self-intersection probes.",
    )
    parser.add_argument("--no-ipc", action="store_true", default=False, help="Disable IPC contacts for ablation.")
    parser.add_argument("--cpu", action="store_true", default=False, help="Force Genesis CPU backend.")
    return parser.parse_args()


def should_show_piper(args):
    return bool(args.show_piper and not args.hide_piper)


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

    add_piperx_wrist_camera_links(root)
    apply_gripper_finger_covers(root)
    apply_simple_gripper_collisions(root)

    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path


def format_urdf_values(values):
    return " ".join(f"{float(value):.12g}" for value in values)


def rotation_matrix_from_rpy(roll, pitch, yaw):
    sin_r, cos_r = math.sin(roll), math.cos(roll)
    sin_p, cos_p = math.sin(pitch), math.cos(pitch)
    sin_y, cos_y = math.sin(yaw), math.cos(yaw)
    return np.asarray(
        (
            (cos_y * cos_p, cos_y * sin_p * sin_r - sin_y * cos_r, cos_y * sin_p * cos_r + sin_y * sin_r),
            (sin_y * cos_p, sin_y * sin_p * sin_r + cos_y * cos_r, sin_y * sin_p * cos_r - cos_y * sin_r),
            (-sin_p, cos_p * sin_r, cos_p * cos_r),
        ),
        dtype=float,
    )


def rotated_finger_cover_bounds(rpy):
    mesh_min = np.asarray(GRIPPER_FINGER_COVER_MESH_MIN, dtype=float)
    mesh_max = np.asarray(GRIPPER_FINGER_COVER_MESH_MAX, dtype=float)
    corners = np.asarray(
        [
            (x, y, z)
            for x in (mesh_min[0], mesh_max[0])
            for y in (mesh_min[1], mesh_max[1])
            for z in (mesh_min[2], mesh_max[2])
        ],
        dtype=float,
    )
    rotated_corners = corners @ rotation_matrix_from_rpy(*rpy).T
    return np.min(rotated_corners, axis=0), np.max(rotated_corners, axis=0)


def finger_cover_visual_origin(target_center, rpy, cover_base_y):
    rotated_min, rotated_max = rotated_finger_cover_bounds(rpy)
    rotated_center = 0.5 * (rotated_min + rotated_max)
    origin = np.asarray(target_center, dtype=float) - rotated_center
    base_center = np.asarray(
        (
            0.5 * (GRIPPER_FINGER_COVER_MESH_MIN[0] + GRIPPER_FINGER_COVER_MESH_MAX[0]),
            GRIPPER_FINGER_COVER_MESH_MAX[1],
            0.5 * (GRIPPER_FINGER_COVER_MESH_MIN[2] + GRIPPER_FINGER_COVER_MESH_MAX[2]),
        ),
        dtype=float,
    )
    rotated_base_center = base_center @ rotation_matrix_from_rpy(*rpy).T
    origin[1] = cover_base_y - rotated_base_center[1]
    return tuple(float(value) for value in origin)


def add_finger_cover_visual(root, link_name, target_center, rpy, cover_base_y):
    link_element = root.find(f"./link[@name='{link_name}']")
    if link_element is None:
        raise RuntimeError(f"Missing URDF link for finger-cover visual: {link_name}")

    visual_element = ET.SubElement(link_element, "visual")
    ET.SubElement(
        visual_element,
        "origin",
        {
            "xyz": format_urdf_values(finger_cover_visual_origin(target_center, rpy, cover_base_y)),
            "rpy": format_urdf_values(rpy),
        },
    )
    geometry_element = ET.SubElement(visual_element, "geometry")
    ET.SubElement(
        geometry_element,
        "mesh",
        {
            "filename": str(GRIPPER_FINGER_COVER_STL),
            "scale": format_urdf_values(GRIPPER_FINGER_COVER_SCALE),
        },
    )
    material_element = ET.SubElement(visual_element, "material", {"name": "Black"})
    ET.SubElement(material_element, "color", {"rgba": "0.01 0.01 0.01 1"})


def apply_gripper_finger_covers(root):
    if not GRIPPER_FINGER_COVER_STL.is_file():
        raise FileNotFoundError(f"Missing gripper finger cover STL: {GRIPPER_FINGER_COVER_STL}")
    for side in ("left", "right"):
        add_finger_cover_visual(
            root,
            f"{side}_link7",
            GRIPPER_FINGER_LINK7_COLLISION_CENTER,
            GRIPPER_FINGER_LINK7_COVER_RPY,
            GRIPPER_FINGER_LINK7_COVER_BASE_Y,
        )
        add_finger_cover_visual(
            root,
            f"{side}_link8",
            GRIPPER_FINGER_LINK8_COLLISION_CENTER,
            GRIPPER_FINGER_LINK8_COVER_RPY,
            GRIPPER_FINGER_LINK8_COVER_BASE_Y,
        )


def add_box_collision(link_element, name, center, size, rpy=(0.0, 0.0, 0.0)):
    collision_element = ET.SubElement(link_element, "collision", {"name": name})
    ET.SubElement(
        collision_element,
        "origin",
        xyz=format_urdf_values(center),
        rpy=format_urdf_values(rpy),
    )
    geometry_element = ET.SubElement(collision_element, "geometry")
    ET.SubElement(geometry_element, "box", size=format_urdf_values(size))


def replace_link_collisions_with_boxes(root, link_name, collisions):
    link_element = root.find(f"./link[@name='{link_name}']")
    if link_element is None:
        raise RuntimeError(f"Missing URDF link for finger collision: {link_name}")
    for collision in list(link_element.findall("collision")):
        link_element.remove(collision)
    for name, center, size, rpy in collisions:
        add_box_collision(link_element, name, center, size, rpy)


def apply_simple_gripper_collisions(root):
    for side in ("left", "right"):
        replace_link_collisions_with_boxes(
            root,
            f"{side}_link7",
            (
                (
                    "finger_body_collision",
                    GENESIS_LINK7_BODY_COLLISION_CENTER,
                    GENESIS_FINGER_BODY_COLLISION_BOX_SIZE,
                    (0.0, 0.0, 0.0),
                ),
                (
                    "finger_cover_collision",
                    GENESIS_LINK7_COVER_COLLISION_CENTER,
                    GENESIS_FINGER_COVER_COLLISION_BOX_SIZE,
                    GRIPPER_FINGER_LINK7_COVER_RPY,
                ),
            ),
        )
        replace_link_collisions_with_boxes(
            root,
            f"{side}_link8",
            (
                (
                    "finger_body_collision",
                    GENESIS_LINK8_BODY_COLLISION_CENTER,
                    GENESIS_FINGER_BODY_COLLISION_BOX_SIZE,
                    (0.0, 0.0, 0.0),
                ),
                (
                    "finger_cover_collision",
                    GENESIS_LINK8_COVER_COLLISION_CENTER,
                    GENESIS_FINGER_COVER_COLLISION_BOX_SIZE,
                    GRIPPER_FINGER_LINK8_COVER_RPY,
                ),
            ),
        )


def add_piperx_wrist_camera_links(root):
    link_names = {link.attrib["name"] for link in root.findall("link")}
    for side in ("left", "right"):
        parent_link = f"{side}_link6"
        camera_link = f"{side}_camera"
        if parent_link not in link_names or camera_link in link_names:
            continue

        ET.SubElement(root, "link", {"name": camera_link})
        joint = ET.SubElement(root, "joint", {"name": f"{side}_link6_to_{camera_link}", "type": "fixed"})
        ET.SubElement(joint, "origin", PIPERX_WRIST_CAMERA_ORIGINS[side])
        ET.SubElement(joint, "parent", {"link": parent_link})
        ET.SubElement(joint, "child", {"link": camera_link})
        link_names.add(camera_link)


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


def table_room_specs():
    table_min = TABLE_CENTER - 0.5 * TABLE_SIZE
    table_max = TABLE_CENTER + 0.5 * TABLE_SIZE
    x_min = float(table_min[0] - ROOM_WALL_DISTANCE_FROM_TABLE)
    x_max = float(table_max[0] + ROOM_WALL_DISTANCE_FROM_TABLE)
    y_min = float(table_min[1] - ROOM_WALL_DISTANCE_FROM_TABLE)
    y_max = float(table_max[1] + ROOM_WALL_DISTANCE_FROM_TABLE)
    wall_center_z = 0.5 * ROOM_CEILING_Z
    wall_height = ROOM_CEILING_Z
    side_wall_y = 0.5 * (y_min + y_max)
    side_wall_length = y_max - y_min - ROOM_WALL_CORNER_GAP
    side_wall_size = (ROOM_WALL_THICKNESS, side_wall_length, wall_height)
    back_wall_size = (x_max - x_min - 2.0 * ROOM_WALL_CORNER_GAP, ROOM_WALL_THICKNESS, wall_height)
    wall_specs = (
        ((x_min - 0.5 * ROOM_WALL_THICKNESS, side_wall_y, wall_center_z), side_wall_size),
        ((x_max + 0.5 * ROOM_WALL_THICKNESS, side_wall_y, wall_center_z), side_wall_size),
        ((0.5 * (x_min + x_max), y_max + 0.5 * ROOM_WALL_THICKNESS, wall_center_z), back_wall_size),
    )
    ceiling_size = (x_max - x_min + 2.0 * ROOM_WALL_THICKNESS, y_max - y_min + 2.0 * ROOM_WALL_THICKNESS, ROOM_WALL_THICKNESS)
    ceiling_spec = (
        (0.5 * (x_min + x_max), 0.5 * (y_min + y_max), ROOM_CEILING_Z),
        ceiling_size,
    )
    light_specs = tuple(
        ((light_x, TABLE_CENTER[1] + 0.05, ROOM_CEILING_Z - 0.035), (0.55, 0.35, 0.02))
        for light_x in (-0.45, 0.45)
    )
    return wall_specs, ceiling_spec, light_specs


def draw_solid_debug_box(scene, pos, size, color):
    half_size = 0.5 * np.array(size, dtype=np.float32)
    center = np.array(pos, dtype=np.float32)
    bounds = np.stack((center - half_size, center + half_size), axis=0)
    scene.draw_debug_box(bounds, color=color, wireframe=False)


def draw_table_room(scene):
    wall_specs, ceiling_spec, light_specs = table_room_specs()
    for pos, size in wall_specs:
        draw_solid_debug_box(scene, pos, size, ROOM_WALL_COLOR)
    draw_solid_debug_box(scene, ceiling_spec[0], ceiling_spec[1], ROOM_WALL_COLOR)
    for pos, size in light_specs:
        draw_solid_debug_box(scene, pos, size, ROOM_LIGHT_PANEL_COLOR)


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
        sim_options=gs.options.SimOptions(dt=SIM_DT, substeps=SIM_SUBSTEPS, gravity=(0.0, 0.0, -9.81)),
        coupler_options=coupler_options,
        vis_options=gs.options.VisOptions(
            ambient_light=(0.34, 0.34, 0.34),
            background_color=(0.78, 0.82, 0.84),
            lights=(
                gs.options.vis.PointLight(pos=(-0.45, TABLE_CENTER[1] + 0.05, ROOM_CEILING_Z - 0.12), color=(1.0, 0.96, 0.88), intensity=18.0),
                gs.options.vis.PointLight(pos=(0.45, TABLE_CENTER[1] + 0.05, ROOM_CEILING_Z - 0.12), color=(1.0, 0.96, 0.88), intensity=18.0),
                gs.options.vis.DirectionalLight(dir=(0.2, -0.3, -1.0), color=(1.0, 1.0, 1.0), intensity=1.2),
            ),
        ),
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
    if should_show_piper(args):
        robot_material = (
            gs.materials.Rigid(
                coup_type="two_way_soft_constraint",
                coup_links=PIPER_COUPLED_FINGER_LINKS,
                coup_friction=PIPER_IPC_COUP_FRICTION,
                sdf_cell_size=0.003,
            )
            if not args.no_ipc
            else gs.materials.Rigid(friction=1.5)
        )
        robot = scene.add_entity(
            morph=gs.morphs.URDF(
                file=str(robot_urdf_path),
                pos=ROBOT_ROOT_POS,
                euler=ROBOT_ROOT_EULER,
                fixed=True,
                merge_fixed_links=False,
                collision=True,
                visualization=True,
            ),
            material=robot_material,
        )

    camera_items = add_robotwin_cameras(scene)
    return scene, table, shirt, robot, camera_items


def robotwin_camera_up(forward, left):
    up = np.cross(forward, left)
    return up / np.linalg.norm(up)


def rotation_matrix_from_quat_xyzw(quat):
    x, y, z, w = quat
    return np.asarray(
        (
            (1.0 - 2.0 * y * y - 2.0 * z * z, 2.0 * x * y - 2.0 * z * w, 2.0 * x * z + 2.0 * y * w),
            (2.0 * x * y + 2.0 * z * w, 1.0 - 2.0 * x * x - 2.0 * z * z, 2.0 * y * z - 2.0 * x * w),
            (2.0 * x * z - 2.0 * y * w, 2.0 * y * z + 2.0 * x * w, 1.0 - 2.0 * x * x - 2.0 * y * y),
        ),
        dtype=float,
    )


def robotwin_head_camera_axes():
    rotation = rotation_matrix_from_quat_xyzw(ROBOTWIN_HEAD_CAMERA_QUAT_XYZW)
    forward = -rotation[0, :]
    left = rotation[1, :]
    return forward / np.linalg.norm(forward), left / np.linalg.norm(left)


def add_robotwin_cameras(scene):
    head_pos = ROBOTWIN_HEAD_CAMERA_POSITION
    head_forward, head_left = robotwin_head_camera_axes()
    head_cam = scene.add_camera(
        res=ROBOTWIN_CAMERA_RES,
        pos=tuple(head_pos),
        lookat=tuple(head_pos + head_forward),
        up=tuple(robotwin_camera_up(head_forward, head_left)),
        fov=ROBOTWIN_HEAD_CAMERA_FOVY,
        GUI=False,
    )
    left_cam = scene.add_camera(
        res=WRIST_CAMERA_RES,
        pos=(-0.46, -1.02, 0.50),
        lookat=(-0.12, -0.48, 0.16),
        fov=WRIST_CAMERA_FOVY,
        near=0.01,
        GUI=False,
    )
    right_cam = scene.add_camera(
        res=WRIST_CAMERA_RES,
        pos=(0.46, -1.02, 0.50),
        lookat=(0.12, -0.48, 0.16),
        fov=WRIST_CAMERA_FOVY,
        near=0.01,
        GUI=False,
    )
    return (
        ("head_camera", head_cam),
        ("left_camera", left_cam),
        ("right_camera", right_cam),
    )


def attach_robotwin_wrist_cameras(robot, camera_items):
    if robot is None:
        return

    cameras_by_name = dict(camera_items)
    for side in ("left", "right"):
        camera_name = f"{side}_camera"
        camera = cameras_by_name[camera_name]
        camera.attach(robot.get_link(camera_name), WRIST_CAMERA_LINK_OFFSET)
        camera.move_to_attach()


def rotation_matrix_from_quat_wxyz(quat):
    w, x, y, z = quat
    return np.asarray(
        (
            (1.0 - 2.0 * y * y - 2.0 * z * z, 2.0 * x * y - 2.0 * z * w, 2.0 * x * z + 2.0 * y * w),
            (2.0 * x * y + 2.0 * z * w, 1.0 - 2.0 * x * x - 2.0 * z * z, 2.0 * y * z - 2.0 * x * w),
            (2.0 * x * z - 2.0 * y * w, 2.0 * y * z + 2.0 * x * w, 1.0 - 2.0 * x * x - 2.0 * y * y),
        ),
        dtype=float,
    )


def piper_finger_collision_centers(robot):
    centers = []
    for side in ("left", "right"):
        for link_short, local_center in (
            ("link7", GENESIS_LINK7_COVER_COLLISION_CENTER),
            ("link8", GENESIS_LINK8_COVER_COLLISION_CENTER),
        ):
            link = robot.get_link(f"{side}_{link_short}")
            pos = link.get_pos().detach().cpu().numpy().astype(float)
            quat = link.get_quat().detach().cpu().numpy().astype(float)
            centers.append(pos + rotation_matrix_from_quat_wxyz(quat) @ np.asarray(local_center, dtype=float))
    return np.asarray(centers, dtype=float)


def piper_finger_target_positions(y, z, gripper_qpos, x_offset=0.0):
    left_x, right_x = PIPER_GRASP_BASE_XS
    return np.asarray(
        (
            (left_x + x_offset, y, z),
            (right_x + x_offset, y, z),
        ),
        dtype=np.float32,
    )


def solve_piper_qpos(robot, seed_qpos, target_positions, gripper_qpos):
    qpos_seed = np.array(seed_qpos, dtype=np.float32, copy=True)
    qpos_seed[ALL_GRIPPER_DOFS] = gripper_qpos
    solved_qpos, error = robot.inverse_kinematics_multilink(
        links=[robot.get_link(link_name) for link_name in PIPER_IK_LINKS],
        poss=target_positions,
        local_points=PIPER_IK_LOCAL_POINTS,
        init_qpos=qpos_seed,
        dofs_idx_local=ALL_ARM_DOFS,
        rot_mask=[False, False, False],
        return_error=True,
        max_samples=40,
        max_solver_iters=120,
        damping=0.02,
    )
    solved_array = solved_qpos.detach().cpu().numpy() if hasattr(solved_qpos, "detach") else np.asarray(solved_qpos)
    target_qpos = qpos_seed.copy()
    target_qpos[ALL_ARM_DOFS] = solved_array[ALL_ARM_DOFS]
    target_qpos[ALL_GRIPPER_DOFS] = gripper_qpos
    return target_qpos, float(np.linalg.norm(error.detach().cpu().numpy()[..., :3]))


def zero_open_qpos():
    qpos = zero_initial_qpos(CLOSED_OPENING)
    qpos[ALL_GRIPPER_DOFS] = OPEN_GRIPPER
    return qpos


def build_piper_motion_targets(robot, closed_gripper):
    target_specs = (
        ("approach_open", PIPER_BASE_CONTACT_Y, PIPER_BASE_HIGH_Z, OPEN_GRIPPER, 0.0),
        ("low_open", PIPER_BASE_CONTACT_Y, PIPER_BASE_CONTACT_Z, OPEN_GRIPPER, 0.0),
        ("low_closed", PIPER_BASE_CONTACT_Y, PIPER_BASE_CONTACT_Z, closed_gripper, 0.0),
        ("pushed_closed", PIPER_BASE_PUSH_Y, PIPER_BASE_CONTACT_Z, closed_gripper, 0.0),
        ("lift", PIPER_BASE_PUSH_Y, PIPER_BASE_LIFT_Z, closed_gripper, 0.0),
        ("release", PIPER_BASE_PUSH_Y, PIPER_BASE_LIFT_Z, OPEN_GRIPPER, 0.0),
        ("retreat", PIPER_BASE_START_Y, PIPER_BASE_LIFT_Z, OPEN_GRIPPER, 0.0),
    )
    targets = {}
    seed_qpos = zero_open_qpos()
    for name, y, z, gripper_qpos, x_offset in target_specs:
        target_qpos, error_norm = solve_piper_qpos(
            robot,
            seed_qpos,
            piper_finger_target_positions(y, z, gripper_qpos, x_offset),
            gripper_qpos,
        )
        targets[name] = target_qpos
        seed_qpos = target_qpos
        print(f"[ik] {name:>17s} position_error_norm={error_norm:.5f}")
    return targets


def interpolate_qpos(start_qpos, end_qpos, steps):
    for alpha in np.linspace(0.0, 1.0, steps, endpoint=True):
        yield (1.0 - alpha) * start_qpos + alpha * end_qpos


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


def update_attached_cameras(cameras):
    for camera in cameras:
        if getattr(camera, "_attached_link", None) is not None:
            camera.move_to_attach()


def record_frame(cameras, record):
    update_attached_cameras(cameras)
    if record:
        for camera in cameras:
            camera.render()


def camera_recording_path(output_dir, video_path, camera_name):
    if camera_name == "head_camera":
        return video_path
    return output_dir / f"{camera_name}.mp4"


def should_record_phase(args, phase):
    if not args.record:
        return False
    if not args.focus_grasp:
        return True
    return phase in {"lower", "contact", "close", "hold", "push", "lift", "hi_hold", "release", "retreat"}


def save_camera_recordings(camera_items, output_dir, video_path):
    recording_paths = {}
    for camera_name, camera in camera_items:
        camera_path = camera_recording_path(output_dir, video_path, camera_name)
        camera.stop_recording(save_to_filename=str(camera_path), fps=RECORDING_FPS)
        recording_paths[camera_name] = camera_path
        print(f"Saved {camera_name} recording to {camera_path}")
    return recording_paths


def compose_left_mid_right_video(recording_paths, output_dir, combined_video_name):
    combined_path = output_dir / combined_video_name
    scale_pad = "scale=640:480:force_original_aspect_ratio=decrease,pad=640:480:(ow-iw)/2:(oh-ih)/2"
    filter_complex = (
        f"[0:v]{scale_pad}[left];"
        f"[1:v]{scale_pad}[mid];"
        f"[2:v]{scale_pad}[right];"
        "[left][mid][right]hstack=inputs=3[v]"
    )
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(recording_paths["left_camera"]),
        "-i",
        str(recording_paths["head_camera"]),
        "-i",
        str(recording_paths["right_camera"]),
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        "-r",
        str(RECORDING_FPS),
        str(combined_path),
    ]
    subprocess.run(command, check=True)
    print(f"Saved left/mid/right recording to {combined_path}")
    return combined_path


def step_phase(
    scene,
    robot,
    shirt,
    cameras,
    phase,
    qpos_targets,
    record,
    log_interval=20,
):
    phase_targets = tuple(qpos_targets)
    for i, qpos_target in enumerate(phase_targets):
        if robot is not None:
            robot.control_dofs_position(qpos_target[ALL_CONTROL_DOFS], ALL_CONTROL_DOFS)
        scene.step()
        record_frame(cameras, record)
        is_final_step = i + 1 == len(phase_targets)
        if i == 0 or is_final_step or (i + 1) % log_interval == 0:
            stats = cloth_stats(shirt)
            if robot is None:
                finger_z = float("nan")
                grip_gap = float("nan")
            else:
                finger_z = float(np.mean(piper_finger_collision_centers(robot)[:, 2]))
                gripper_qpos = robot.get_qpos(qs_idx_local=ALL_GRIPPER_DOFS).detach().cpu().numpy()
                grip_gap = float(np.mean(np.abs(gripper_qpos)))
            print(
                f"[{phase:>8s}] step={i + 1:04d} "
                f"centroid=({stats['centroid'][0]:+.3f}, {stats['centroid'][1]:+.3f}, {stats['centroid'][2]:+.3f}) "
                f"z_min={stats['min_z']:.3f} z_max={stats['max_z']:.3f} "
                f"finger_z={finger_z:.3f} grip_gap={grip_gap:.4f} sentinel={stats['sentinel_count']}"
            )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shirt_mesh_path = resolve_tshirt_obj(args)
    robot_urdf_path = (
        make_piper_urdf(output_dir / "piper_x_dualarm_genesis.urdf")
        if should_show_piper(args)
        else None
    )

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, logging_level="info")
    scene, _, shirt, robot, camera_items = make_scene(args, shirt_mesh_path, robot_urdf_path)
    cameras = [camera for _, camera in camera_items]

    closed_gripper = gripper_qpos(args.closed_opening)
    closed_init_qpos = zero_initial_qpos(args.closed_opening)
    open_start_qpos = zero_open_qpos()
    initial_qpos = closed_init_qpos if args.init_only else open_start_qpos
    if robot is not None:
        set_robot_init_qpos(robot, initial_qpos)
    scene.build()
    draw_table_room(scene)
    if args.init_only:
        print(f"IPC init probe succeeded: closed_opening={args.closed_opening:.6f}")
        return
    if robot is None:
        raise RuntimeError("The standalone grippers were removed; run without --hide-piper for robot manipulation.")
    attach_robotwin_wrist_cameras(robot, camera_items)

    if robot is not None:
        robot.set_dofs_kp(np.array([4500.0] * 12 + [PIPER_GRIPPER_KP] * 4, dtype=np.float32), ALL_CONTROL_DOFS)
        robot.set_dofs_kv(np.array([450.0] * 12 + [PIPER_GRIPPER_KV] * 4, dtype=np.float32), ALL_CONTROL_DOFS)
        robot.set_dofs_force_range(
            np.array([-87.0] * 12 + [-PIPER_GRIPPER_FORCE_LIMIT] * 4, dtype=np.float32),
            np.array([87.0] * 12 + [PIPER_GRIPPER_FORCE_LIMIT] * 4, dtype=np.float32),
            ALL_CONTROL_DOFS,
        )
        piper_targets = build_piper_motion_targets(robot, closed_gripper)
    else:
        piper_targets = {}

    physics_steps_per_action = max(1, int(round(0.35 * args.horizon_scale / SIM_DT)))
    phase_steps = {
        "approach": max(physics_steps_per_action * 2, 6),
        "lower": max(physics_steps_per_action, 18),
        "hold_contact_open": max(physics_steps_per_action, 18),
        "close": max(physics_steps_per_action * 2, 9),
        "hold_closed": max(physics_steps_per_action * 4, 18),
        "push": max(physics_steps_per_action * 3, 9),
        "lift": max(physics_steps_per_action * 10, 50),
        "hold_lift": max(physics_steps_per_action * 8, 40),
        "release": max(physics_steps_per_action * 2, 6),
        "retreat": max(physics_steps_per_action * 5, 18),
    }

    if args.record:
        for camera in cameras:
            camera.start_recording()

    video_path = output_dir / args.video_name
    try:
        step_phase(
            scene,
            robot,
            shirt,
            cameras,
            "approach",
            interpolate_qpos(open_start_qpos, piper_targets["approach_open"], phase_steps["approach"]),
            should_record_phase(args, "approach"),
        )
        step_phase(
            scene,
            robot,
            shirt,
            cameras,
            "lower",
            interpolate_qpos(piper_targets["approach_open"], piper_targets["low_open"], phase_steps["lower"]),
            should_record_phase(args, "lower"),
        )
        step_phase(
            scene,
            robot,
            shirt,
            cameras,
            "contact",
            [piper_targets["low_open"]] * phase_steps["hold_contact_open"],
            should_record_phase(args, "contact"),
        )
        step_phase(
            scene,
            robot,
            shirt,
            cameras,
            "close",
            interpolate_qpos(piper_targets["low_open"], piper_targets["low_closed"], phase_steps["close"]),
            should_record_phase(args, "close"),
        )
        step_phase(
            scene,
            robot,
            shirt,
            cameras,
            "hold",
            [piper_targets["low_closed"]] * phase_steps["hold_closed"],
            should_record_phase(args, "hold"),
        )
        step_phase(
            scene,
            robot,
            shirt,
            cameras,
            "push",
            interpolate_qpos(piper_targets["low_closed"], piper_targets["pushed_closed"], phase_steps["push"]),
            should_record_phase(args, "push"),
        )
        step_phase(
            scene,
            robot,
            shirt,
            cameras,
            "lift",
            interpolate_qpos(piper_targets["pushed_closed"], piper_targets["lift"], phase_steps["lift"]),
            should_record_phase(args, "lift"),
        )
        step_phase(
            scene,
            robot,
            shirt,
            cameras,
            "hi_hold",
            [piper_targets["lift"]] * phase_steps["hold_lift"],
            should_record_phase(args, "hi_hold"),
        )
        step_phase(
            scene,
            robot,
            shirt,
            cameras,
            "release",
            interpolate_qpos(piper_targets["lift"], piper_targets["release"], phase_steps["release"]),
            should_record_phase(args, "release"),
        )
        step_phase(
            scene,
            robot,
            shirt,
            cameras,
            "retreat",
            interpolate_qpos(
                piper_targets["release"],
                piper_targets["retreat"],
                phase_steps["retreat"],
            ),
            should_record_phase(args, "retreat"),
        )
    finally:
        if args.record:
            recording_paths = save_camera_recordings(camera_items, output_dir, video_path)
            compose_left_mid_right_video(recording_paths, output_dir, args.combined_video_name)

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
