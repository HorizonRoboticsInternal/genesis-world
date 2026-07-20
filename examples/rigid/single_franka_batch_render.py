import argparse
import math
import numpy as np
import torch

import genesis as gs
from genesis.utils.geom import trans_to_T
from genesis.utils.image_exporter import FrameImageExporter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("--show_viewer", action="store_true", default=False)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    parser.add_argument("-b", "--n_envs", type=int, default=3)
    parser.add_argument("-s", "--n_steps", type=int, default=2)
    parser.add_argument("-r", "--render_all_cameras", action="store_true", default=False)
    parser.add_argument("-o", "--output_dir", type=str, default="data/test")
    parser.add_argument("-u", "--use_rasterizer", action="store_true", default=False)
    parser.add_argument("--non_batch_render", action="store_true", default=False)
    parser.add_argument("-f", "--use_fisheye", action="store_true", default=False)
    parser.add_argument("-d", "--debug", action="store_true", default=False)
    parser.add_argument("-l", "--seg_level", type=str, default="link")
    parser.add_argument("--include_seg", action="store_true", default=False)
    args = parser.parse_args()

    # The batch renderer only runs on CUDA, so there is no CPU backend to offer here.
    gs.init(backend=gs.gpu)

    ########################## create a scene ##########################
    renderer = (
        gs.renderers.Rasterizer()
        if args.non_batch_render
        else gs.options.renderers.BatchRenderer(
            use_rasterizer=args.use_rasterizer,
        )
    )
    scene = gs.Scene(
        vis_options=gs.options.VisOptions(
            segmentation_level=args.seg_level,
            rendered_envs_idx=list(range(args.n_envs)),
            env_separate_rigid=True
        ),
        renderer=renderer,
        show_viewer=args.show_viewer,
    )

    plane = scene.add_entity(
        gs.morphs.Plane(),
        surface=gs.surfaces.Default(
            diffuse_texture=gs.textures.BatchTexture.from_images(image_folder="textures"),
        ),
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        visualize_contact=True,
    )

    debug_cam = scene.add_camera(
        res=(720, 1280),
        pos=(1.5, -0.5, 1.0),
        lookat=(0.0, 0.0, 0.5),
        fov=60,
        GUI=args.vis,
        debug=True,
    )
    cam_0 = scene.add_camera(
        res=(512, 512),
        pos=(1.5, 0.5, 1.5),
        lookat=(0.0, 0.0, 0.5),
        fov=45,
        GUI=args.vis,
    )
    cam_0.attach(franka.links[6], trans_to_T(np.array([0.0, 0.5, 0.0])))
    cam_1 = scene.add_camera(
        res=(512, 512),
        pos=(1.5, -0.5, 1.5),
        lookat=(0.0, 0.0, 0.5),
        fov=45,
        GUI=args.vis,
    )
    cam_2 = scene.add_camera(
        res=(512, 512),
        pos=(0.0, 0.0, 5.0),
        lookat=(0.0, 0.0, 0.0),
        fov=70,
        model="fisheye" if args.use_fisheye else "pinhole",
        GUI=args.vis,
    )

    if not args.non_batch_render:
        scene.add_light(
            pos=(0.0, 0.0, 1.5),
            dir=(1.0, 1.0, -2.0),
            color=(1.0, 0.0, 0.0),
            directional=True,
            castshadow=True,
            cutoff=45.0,
            intensity=0.5,
        )
        scene.add_light(
            pos=(4, -4, 4),
            dir=(0, 0, -1),
            directional=False,
            castshadow=True,
            cutoff=80.0,
            intensity=1.0,
            attenuation=0.1,
        )

    ########################## build ##########################
    scene.build(n_envs=args.n_envs, env_spacing=(1.0, 1.0))

    joints_name = (
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
        "finger_joint1",
        "finger_joint2",
    )
    motors_dof_idx = [franka.get_joint(name).dofs_idx_local[0] for name in joints_name]
    franka.set_dofs_kp(
        kp=np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
        dofs_idx_local=motors_dof_idx,
    )
    franka.set_dofs_kv(
        kv=np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
        dofs_idx_local=motors_dof_idx,
    )
    franka.set_dofs_force_range(
        lower=np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        upper=np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
        dofs_idx_local=motors_dof_idx,
    )
    base_qpos = torch.tensor(
        [0.0, -0.3, 0.0, -1.8, 0.0, 1.6, 0.8, 0.04, 0.04],
        dtype=gs.tc_float,
        device=gs.device,
    )
    trajectory_amplitude = torch.tensor(
        [0.35, 0.25, 0.30, 0.25, 0.30, 0.20, 0.25, 0.0, 0.0],
        dtype=gs.tc_float,
        device=gs.device,
    )
    joint_phase = torch.tensor(
        [0.0, 0.8, 1.6, 2.4, 3.2, 4.0, 4.8, 0.0, 0.0],
        dtype=gs.tc_float,
        device=gs.device,
    )
    env_phase = torch.linspace(
        0.0,
        2.0 * math.pi,
        args.n_envs + 1,
        dtype=gs.tc_float,
        device=gs.device,
    )[:-1, None]
    franka.set_dofs_position(
        base_qpos[None, :].repeat(args.n_envs, 1),
        motors_dof_idx,
    )

    # # Create an image exporter
    # exporter = FrameImageExporter(args.output_dir)

    if args.debug:
        debug_cam.start_recording()
    for step_idx in range(args.n_steps):
        trajectory_time = step_idx * scene.dt
        target_qpos = base_qpos + trajectory_amplitude * torch.sin(
            2.0 * math.pi * 0.25 * trajectory_time + env_phase + joint_phase
        )
        franka.control_dofs_position(target_qpos, motors_dof_idx)
        scene.step()
        render_segmentation = args.include_seg and step_idx % 2 == 1
        if args.debug:
            debug_cam.render()
        if args.render_all_cameras:
            if args.non_batch_render:
                for cam in (cam_0, cam_1, cam_2):
                    color, depth, seg, normal = cam.render(
                        rgb=True,
                        depth=step_idx % 2 == 1,
                        segmentation=render_segmentation,
                        normal=True,
                    )
            else:
                color, depth, seg, normal = scene.render_all_cameras(
                    rgb=True, depth=step_idx % 2 == 1, segmentation=render_segmentation, normal=True
                )
            # exporter.export_frame_all_cameras(i, rgb=color, depth=depth, segmentation=seg, normal=normal)
        else:
            color, depth, seg, normal = cam_1.render(
                rgb=False,
                depth=True,
                segmentation=args.include_seg,
                colorize_seg=True,
                normal=False,
            )
            # exporter.export_frame_single_camera(i, cam_1.idx, rgb=seg, depth=depth, segmentation=None, normal=normal)
    if args.debug:
        debug_cam.stop_recording()


if __name__ == "__main__":
    main()
