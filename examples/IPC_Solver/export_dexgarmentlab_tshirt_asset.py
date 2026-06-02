import argparse
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom


DEFAULT_SOURCE_USD = Path(
    "/home/horizon/DexGarmentLab/Assets/Garment/Tops/"
    "NoCollar_Ssleeve_FrontClose/TNSC_T_Shirt_Short_Sleeve/"
    "TNSC_T_Shirt_Short_Sleeve_obj.usd"
)
DEFAULT_OUTPUT_OBJ = (
    Path(__file__).resolve().parents[2]
    / "genesis/assets/meshes/garments/dexgarmentlab_short_sleeve_tshirt.obj"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export the DexGarmentLab short-sleeve T-shirt USD mesh to a Genesis OBJ asset."
    )
    parser.add_argument("--source-usd", type=Path, default=DEFAULT_SOURCE_USD)
    parser.add_argument("--output-obj", type=Path, default=DEFAULT_OUTPUT_OBJ)
    parser.add_argument(
        "--scale",
        type=float,
        default=0.55,
        help="Uniform scale applied after centering X/Y and min-Z alignment.",
    )
    return parser.parse_args()


def collect_mesh_triangles(stage):
    vertices_by_mesh = []
    faces = []
    vertex_offset = 0

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue

        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        if not points or not counts or not indices:
            continue

        xform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        mesh_vertices = np.array(
            [
                tuple(
                    xform.Transform(
                        Gf.Vec3d(float(point[0]), float(point[1]), float(point[2]))
                    )
                )
                for point in points
            ],
            dtype=np.float64,
        )

        cursor = 0
        for count in counts:
            face_indices = list(indices[cursor : cursor + count])
            cursor += count
            for i in range(1, count - 1):
                faces.append(
                    (
                        vertex_offset + face_indices[0] + 1,
                        vertex_offset + face_indices[i] + 1,
                        vertex_offset + face_indices[i + 1] + 1,
                    )
                )

        vertices_by_mesh.append(mesh_vertices)
        vertex_offset += len(mesh_vertices)

    if not vertices_by_mesh:
        raise RuntimeError("No mesh prims found in USD stage.")

    return np.concatenate(vertices_by_mesh, axis=0), faces


def center_align_and_scale(vertices, scale):
    processed_vertices = np.array(vertices, dtype=np.float64, copy=True)
    local_min = processed_vertices.min(axis=0)
    local_max = processed_vertices.max(axis=0)
    local_center = 0.5 * (local_min + local_max)

    processed_vertices[:, 0] -= local_center[0]
    processed_vertices[:, 1] -= local_center[1]
    processed_vertices[:, 2] -= local_min[2]
    processed_vertices *= scale
    return processed_vertices


def write_obj(obj_path, vertices, faces, source_usd, scale):
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    with obj_path.open("w", encoding="utf-8") as obj_file:
        obj_file.write(f"# Exported from {source_usd}\n")
        obj_file.write("# X/Y centered and min-Z aligned for Genesis table cloth.\n")
        obj_file.write(f"# Uniform scale: {scale}\n")
        for vx, vy, vz in vertices:
            obj_file.write(f"v {vx:.9f} {vy:.9f} {vz:.9f}\n")
        for a, b, c in faces:
            obj_file.write(f"f {a} {b} {c}\n")


def main():
    args = parse_args()
    source_usd = args.source_usd.expanduser().resolve()
    output_obj = args.output_obj.expanduser().resolve()
    if not source_usd.is_file():
        raise FileNotFoundError(f"DexGarmentLab T-shirt USD does not exist: {source_usd}")

    stage = Usd.Stage.Open(str(source_usd))
    if stage is None:
        raise RuntimeError(f"Could not open USD stage: {source_usd}")

    raw_vertices, faces = collect_mesh_triangles(stage)
    processed_vertices = center_align_and_scale(raw_vertices, args.scale)
    write_obj(output_obj, processed_vertices, faces, source_usd, args.scale)

    print(
        "exported_dexgarmentlab_tshirt "
        f"output={output_obj} vertices={len(processed_vertices)} faces={len(faces)} "
        f"span={np.ptp(processed_vertices, axis=0).tolist()}"
    )


if __name__ == "__main__":
    main()
