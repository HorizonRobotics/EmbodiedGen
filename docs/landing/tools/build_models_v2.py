#!/usr/bin/env python3
"""Rebuild the landing asset viewer models from a project_example_v2 export.

Source layout (per asset, dir name may contain spaces):
  <asset>/<asset>.urdf
  <asset>/mesh/<mesh>.glb            visual mesh (mesh file may be renamed/suffixed)
  <asset>/mesh/<mesh>_collision.obj  collision proxy
  <asset>/affordance/mesh_part_seg.glb + affordance_annot.json

For each complete asset this produces, under docs/landing/assets/models/:
  <key>.glb            visual mesh (copied as-is)
  <key>_collision.glb  per-part-colorized collision proxy
  <key>_afford.glb     part-seg mesh + baked parallel-jaw grippers (top-k grasps)
plus assets.json (gallery + physics meta) and affordance.json (part legend).

<key> = asset dir name, spaces -> underscores, lowercased. Grasp confidence is
only used to pick the top-k grippers; it is never displayed. Idempotent.
"""
import glob
import json
import os
import re
import sys

import numpy as np
import trimesh
from trimesh.transformations import quaternion_matrix

# Ordered source roots. Each: (dir, names) — names=None means every subdir,
# otherwise only that curated set. The original headphones showcase asset (from
# the v1 export) leads the gallery, followed by the full project_example_v3 batch.
_LUCAS = "/horizon-bucket/robot_lab/users/lucas.ding/output"
SRC_ROOTS = [
    (f"{_LUCAS}/project_example", ["ear_hear"]),
    (f"{_LUCAS}/project_example_v3", None),
]
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(__file__), "..", "assets", "models")
OUT = os.path.abspath(OUT)
TOPK = 2  # grippers baked per part (top-k by confidence)
# Assets intentionally dropped from the gallery (by key). screwdriver = the
# green-handled one (yellow-handled repair_tools is kept).
EXCLUDE = {"screwdriver", "tableware", "thermos"}

# mask_color name -> hex (must match repo PALETTE in vis_utils.py).
COLOR = {
    "red": "#E6194B", "green": "#3CB44B", "blue": "#0082C8", "yellow": "#FFE119",
    "orange": "#F58231", "purple": "#911EB4", "cyan": "#46F0F0", "magenta": "#F032E6",
    "pink": "#FABED4", "lime green": "#D2F53C", "teal": "#008080", "brown": "#AA6E28",
    "maroon": "#800000", "navy": "#000080", "olive": "#6B8E23", "gray": "#808080",
    "crimson": "#DC143C", "white": "#FFFFFF", "burnt orange": "#CC5500", "jade": "#00998F",
}
# collision per-connected-component palette (RGBA)
PAL = [(99, 179, 237), (129, 199, 132), (255, 183, 77), (186, 104, 200),
       (240, 98, 146), (77, 208, 225), (255, 138, 101), (174, 213, 129)]


def key_of(name):
    return re.sub(r"\s+", "_", name.strip()).lower()


def tag(text, name, default=""):
    m = re.search(rf"<{name}>(.*?)</{name}>", text, re.S)
    return m.group(1).strip() if m else default


def hex_rgba(h, a=255):
    h = h.lstrip("#")
    return [int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), a]


def box(extents, transform, rgba):
    m = trimesh.creation.box(extents=extents, transform=transform)
    m.visual = trimesh.visual.ColorVisuals(
        mesh=m, face_colors=np.tile(rgba, (len(m.faces), 1)))
    return m


def gripper(rgba):
    """Franka-panda style parallel-jaw marker in the grasp (hand) frame.
    Hand origin at z=0, approach along +Z, fingers separated along X, contact
    (TCP) at z=TCP. Matches the convention where grasp xyz is the hand origin.
    """
    TCP = 0.1034      # hand -> fingertip along +z
    base = 0.0584     # hand -> finger base (crossbar) along +z
    fw = 0.041        # finger half-separation (x)
    T = 0.006         # tube thickness
    tr = trimesh.transformations.translation_matrix
    parts = [
        box([T, T, base], tr([0, 0, base / 2]), rgba),                   # stem
        box([2 * fw + T, T, T], tr([0, 0, base]), rgba),                 # crossbar
        box([T, T, TCP - base], tr([fw, 0, (base + TCP) / 2]), rgba),    # finger +x
        box([T, T, TCP - base], tr([-fw, 0, (base + TCP) / 2]), rgba),   # finger -x
    ]
    return trimesh.util.concatenate(parts)


def load_mesh(path):
    m = trimesh.load(path, force="mesh", process=False)
    if isinstance(m, trimesh.Scene):
        m = m.dump(concatenate=True)
    return m


def build_collision(obj_path, dst):
    m = load_mesh(obj_path)
    parts = m.split(only_watertight=False) or [m]
    if len(parts) == 0:
        parts = [m]
    sc = trimesh.Scene()
    for i, p in enumerate(parts):
        c = PAL[i % len(PAL)]
        p.visual = trimesh.visual.ColorVisuals(
            mesh=p, face_colors=np.tile([*c, 235], (len(p.faces), 1)))
        sc.add_geometry(p)
    sc.export(dst)


def build_affordance(seg_glb, annot_path, dst):
    """Bake top-k grippers onto the part-seg mesh. The grasps live in the OBJECT
    frame, which differs from the raw part-seg glb by the URDF visual_seg origin
    (rpy = 1.5708,0,0 = Rx(90)). Bring the mesh into the object frame, place each
    gripper directly (translation @ quaternion), then rotate the whole scene back
    so it displays in the same raw-glb orientation as the visual/collision tabs.
    """
    annot = json.load(open(annot_path))
    mesh = load_mesh(seg_glb)
    Morigin = trimesh.transformations.euler_matrix(1.5708, 0, 0, axes="sxyz")
    mesh.apply_transform(Morigin)
    scene = trimesh.Scene(mesh)

    parts_out = []
    gi = 0
    for part in annot.get("affordances", []):
        grasps = part.get("grasp_group", {}) or {}
        top = sorted(grasps.values(), key=lambda g: -g.get("confidence", 0))[:TOPK]
        cname = (part.get("mask_color") or "").strip().lower()
        phex = COLOR.get(cname, "#8A9099")
        for g in top:
            pos = np.array(g["position"], float)
            q = g["orientation"]
            Robj = quaternion_matrix(
                [q["w"], q["xyz"][0], q["xyz"][1], q["xyz"][2]])[:3, :3]
            pose = np.eye(4)
            pose[:3, :3] = Robj
            pose[:3, 3] = pos
            grip = gripper(hex_rgba(phex))
            grip.apply_transform(pose)
            scene.add_geometry(grip, geom_name=f"grasp_{gi}")
            gi += 1
        scen = sorted((part.get("grasp_scenarios") or []),
                      key=lambda s: -s.get("confidence", 0))[:2]
        parts_out.append({
            "part_name": part.get("part_name", ""),
            "color": phex,
            "color_name": part.get("mask_color", ""),
            "graspable": bool(part.get("graspable", False)),
            "labels": part.get("functional_labels", [])[:5],
            "description": part.get("semantic_description", ""),
            "scenarios": [s.get("scenario", "") for s in scen],
        })
    scene.apply_transform(np.linalg.inv(Morigin))
    scene.export(dst)
    return parts_out, gi


def iter_assets():
    """Yield (name, asset_dir) across all source roots, in gallery order."""
    for root, names in SRC_ROOTS:
        if not os.path.isdir(root):
            print(f"  (source missing: {root})")
            continue
        subdirs = names if names is not None else sorted(
            d for d in os.listdir(root) if os.path.isdir(f"{root}/{d}"))
        for name in subdirs:
            yield name, f"{root}/{name}"


def main():
    os.makedirs(OUT, exist_ok=True)
    assets = []
    afford = {}
    seen = set()
    for name, d in iter_assets():
        key = key_of(name)
        if key in seen or key in EXCLUDE:
            continue
        # visual: the textured mesh glb (not the collision proxy, not the GS glb)
        glbs = [x for x in glob.glob(f"{d}/mesh/*.glb")
                if not x.endswith(("_gs.glb", "_collision.glb"))]
        # collision proxy: .obj (v2) or .glb (v1)
        colls = (glob.glob(f"{d}/mesh/*_collision.obj")
                 or glob.glob(f"{d}/mesh/*_collision.glb"))
        seg = f"{d}/affordance/mesh_part_seg.glb"
        annot = f"{d}/affordance/affordance_annot.json"
        urdf = glob.glob(f"{d}/*.urdf")
        if not (glbs and colls and os.path.isfile(seg)
                and os.path.isfile(annot) and urdf):
            print(f"  SKIP {name} (incomplete)")
            continue
        seen.add(key)

        os.system(f'cp -f "{glbs[0]}" "{OUT}/{key}.glb"')
        try:
            build_collision(colls[0], f"{OUT}/{key}_collision.glb")
            coll_name = f"{key}_collision.glb"
        except Exception as e:
            print(f"  collision fail {key}: {e}")
            coll_name = ""
        parts_out, gi = build_affordance(seg, annot, f"{OUT}/{key}_afford.glb")
        afford[key] = {"parts": parts_out}

        t = open(urdf[0]).read()
        mass = re.search(r'<mass value="([0-9.]+)"', t)
        mu = re.search(r"<mu1>([0-9.]+)</mu1>", t)
        aes = re.search(r"<ImageAestheticChecker>([0-9.]+)", t)
        cat = tag(t, "category", name)
        assets.append({
            "name": key,
            "label": (cat or name).replace("_", " ").title(),
            "model": f"{key}.glb",
            "collision": coll_name,
            "category": cat,
            "description": tag(t, "description"),
            "height": tag(t, "real_height"),
            "mass": mass.group(1) if mass else "",
            "friction": mu.group(1) if mu else "",
            "aesthetic": aes.group(1) if aes else "",
        })
        print(f"  -> {key}  parts={len(parts_out)} grasps={gi}")

    json.dump(assets, open(f"{OUT}/assets.json", "w"), indent=2)
    json.dump(afford, open(f"{OUT}/affordance.json", "w"), indent=2)
    print(f"\nDone: {len(assets)} assets -> {OUT}")


if __name__ == "__main__":
    main()
