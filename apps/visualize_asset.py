# Project EmbodiedGen
#
# Copyright (c) 2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.


import os

gradio_tmp_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "gradio_cache"
)
os.makedirs(gradio_tmp_dir, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = gradio_tmp_dir

import colorsys
import shutil
import uuid
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import trimesh
from app_style import custom_theme
from embodied_gen.utils.tags import VERSION

try:
    from embodied_gen.utils.gpt_clients import GPT_CLIENT as gpt_client

    gpt_client.check_connection()
    GPT_AVAILABLE = True
except Exception as e:
    gpt_client = None
    GPT_AVAILABLE = False
    print(
        f"Warning: GPT client could not be initialized. Search will be disabled. Error: {e}"
    )


# --- Configuration & Data Loading ---
RUNNING_MODE = "local"  # local or hf_remote
CSV_FILE = "dataset_index.csv"
HF_REPO_ID = "HorizonRobotics/EmbodiedGenData"
HF_LOCAL_DIR = "EmbodiedGenData"
CAMERA_ZOOM = 3.2

# Compatible with huggingface space zero GPU
import spaces
from huggingface_hub import snapshot_download


@spaces.GPU
def fake_gpu_init():
    pass


fake_gpu_init()

if RUNNING_MODE == "local":
    DATA_ROOT = "/horizon-bucket/robot_lab/datasets/embodiedgen/assets_v2"
elif RUNNING_MODE == "hf_remote":
    # Only fetch the index and preview videos up front; per-asset mesh/urdf/
    # usd/mjcf files are pulled lazily on demand (see `ensure_hf_files`).
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        allow_patterns=[
            f"dataset/{CSV_FILE}",
            "dataset/**/*.mp4",
        ],
        local_dir=HF_LOCAL_DIR,
        local_dir_use_symlinks=False,
    )
    DATA_ROOT = os.path.join(HF_LOCAL_DIR, "dataset")
else:
    raise ValueError(
        f"Unknown RUNNING_MODE: {RUNNING_MODE}, must be 'local' or 'hf_remote'."
    )


def ensure_hf_files(rel_patterns: list[str]) -> None:
    """Lazily download files for one asset in hf_remote mode.

    Args:
        rel_patterns: Glob patterns relative to ``DATA_ROOT`` (the repo's
            ``dataset/`` folder), e.g. ``"cat/sub/uid/mesh/**"``.

    No-op when running locally, where all files already exist on disk.
    Progress toasts are emitted by the calling action handlers.
    """
    if RUNNING_MODE != "hf_remote":
        return
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        allow_patterns=[f"dataset/{p}" for p in rel_patterns],
        local_dir=HF_LOCAL_DIR,
        local_dir_use_symlinks=False,
    )


csv_path = os.path.join(DATA_ROOT, CSV_FILE)
df = pd.read_csv(csv_path)
TMP_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "sessions/asset_viewer"
)
os.makedirs(TMP_DIR, exist_ok=True)


# --- Custom CSS for Styling ---
css = """
.gradio-container .gradio-group { box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important; }
#asset-gallery { border: 1px solid #E5E7EB; border-radius: 8px; padding: 8px; background-color: #F9FAFB; }
"""

lighting_css = """
<style>
#visual_mesh canvas { filter: brightness(2.2) !important; }
#collision_mesh_a canvas, #collision_mesh_b canvas { filter: brightness(1.0) !important; }
</style>
"""

_prev_temp = {}


def _unique_path(
    src_path: str | None, session_hash: str, kind: str
) -> str | None:
    """Link/copy src to GRADIO_TEMP_DIR/session_hash with random filename. Always return a fresh URL."""
    if not src_path:
        return None
    tmp_root = (
        Path(os.environ.get("GRADIO_TEMP_DIR", "/tmp"))
        / "model3d-cache"
        / session_hash
    )
    tmp_root.mkdir(parents=True, exist_ok=True)

    # rolling cleanup for same kind
    prev = _prev_temp.get(session_hash, {})
    old = prev.get(kind)
    if old and old.exists():
        old.unlink()

    ext = Path(src_path).suffix or ".glb"
    dst = tmp_root / f"{kind}-{uuid.uuid4().hex}{ext}"
    shutil.copy2(src_path, dst)

    prev[kind] = dst
    _prev_temp[session_hash] = prev
    return str(dst)


def _bounding_radius(mesh: trimesh.Trimesh | trimesh.Scene) -> float:
    """Radius of the mesh/scene bounding sphere (half the diagonal)."""
    lo, hi = mesh.bounds
    return float(np.linalg.norm(np.asarray(hi) - np.asarray(lo)) / 2)


def _camera_position(radius: float | None) -> tuple:
    """Initial camera `(alpha, beta, radius)`; only push the distance."""
    if not radius or radius <= 0:
        return (None, None, None)
    return (None, None, CAMERA_ZOOM * radius)


def _visual_radius(visual_path: str | None) -> float | None:
    if not visual_path:
        return None
    return _bounding_radius(trimesh.load(visual_path, process=False))


def _pastel_color(i: int) -> list[int]:
    """Distinct pastel RGBA for the i-th convex piece (golden-ratio hues)."""
    h = (i * 0.6180339887498949) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.45, 0.98)
    return [int(r * 255), int(g * 255), int(b * 255), 255]


def _colored_collision_path(
    collision_path: str | None, session_hash: str
) -> tuple[str | None, float | None]:
    """Export the collision mesh as a GLB with one color per convex piece.

    The convex-decomposition pieces are stored as separate objects inside the
    single collision ``.obj``; they are recovered via connected-component split
    and each is tinted a distinct pastel color for visualization.

    Returns ``(glb_path, bounding_radius)`` for camera framing.
    """
    if not collision_path:
        return None, None

    tmp_root = (
        Path(os.environ.get("GRADIO_TEMP_DIR", "/tmp"))
        / "model3d-cache"
        / session_hash
    )
    tmp_root.mkdir(parents=True, exist_ok=True)

    # rolling cleanup for same kind
    prev = _prev_temp.get(session_hash, {})
    old = prev.get("collision")
    if old and old.exists():
        old.unlink()

    loaded = trimesh.load(collision_path, process=False)
    if isinstance(loaded, trimesh.Scene):
        parts = list(loaded.geometry.values())
    else:
        parts = loaded.split(only_watertight=False)
        if len(parts) == 0:
            parts = [loaded]

    scene = trimesh.Scene()
    for i, part in enumerate(parts):
        part.visual = trimesh.visual.ColorVisuals(
            mesh=part, vertex_colors=_pastel_color(i)
        )
        scene.add_geometry(part)

    dst = tmp_root / f"collision-{uuid.uuid4().hex}.glb"
    scene.export(str(dst))

    prev["collision"] = dst
    _prev_temp[session_hash] = prev
    return str(dst), _bounding_radius(scene)


# --- Helper Functions (data filtering) ---
def get_primary_categories():
    return sorted(df["primary_category"].dropna().unique())


def get_secondary_categories(primary):
    if not primary:
        return []
    return sorted(
        df[df["primary_category"] == primary]["secondary_category"]
        .dropna()
        .unique()
    )


def get_categories(primary, secondary):
    if not primary or not secondary:
        return []
    return sorted(
        df[
            (df["primary_category"] == primary)
            & (df["secondary_category"] == secondary)
        ]["category"]
        .dropna()
        .unique()
    )


def get_assets(primary, secondary, category):
    if not primary or not secondary:
        return [], gr.update(interactive=False), pd.DataFrame()

    subset = df[
        (df["primary_category"] == primary)
        & (df["secondary_category"] == secondary)
    ]
    if category:
        subset = subset[subset["category"] == category]

    items = []
    for row in subset.itertuples():
        asset_dir = os.path.join(DATA_ROOT, row.asset_dir)
        video_path = None
        if pd.notna(asset_dir) and os.path.exists(asset_dir):
            for f in os.listdir(asset_dir):
                if f.lower().endswith(".mp4"):
                    video_path = os.path.join(asset_dir, f)
                    break
        items.append(
            video_path
            if video_path
            else "https://dummyimage.com/512x512/cccccc/000000&text=No+Preview"
        )

    return items, gr.update(interactive=True), subset


def search_assets(query: str, top_k: int):
    if not GPT_AVAILABLE or not query:
        gr.Warning(
            "GPT client is not available or query is empty. Cannot perform search."
        )
        return [], gr.update(interactive=False), pd.DataFrame()

    gr.Info(f"Searching for assets matching: '{query}'...")

    keywords = query.split()
    keyword_filter = pd.Series([False] * len(df), index=df.index)
    for keyword in keywords:
        keyword_filter |= df['description'].str.contains(
            keyword, case=False, na=False
        )

    candidates = df[keyword_filter]

    if len(candidates) > 100:
        candidates = candidates.head(100)

    if candidates.empty:
        gr.Warning("No assets found matching the keywords.")
        return [], gr.update(interactive=True), pd.DataFrame()

    try:
        descriptions = [
            f"{idx}: {desc}" for idx, desc in candidates['description'].items()
        ]
        descriptions_text = "\n".join(descriptions)

        prompt = f"""
        A user is searching for 3D assets with the query: "{query}".
        Below is a list of available assets, each with an ID and a description.
        Please evaluate how well each asset description matches the user's query and rate them on a scale from 0 to 10, where 10 is a perfect match.

        Your task is to return a list of the top {top_k} asset IDs, sorted from the most relevant to the least relevant.
        The output format must be a simple comma-separated list of IDs, for example: "123,45,678". Do not add any other text.

        Asset Descriptions:
        {descriptions_text}

        User Query: "{query}"

        Top {top_k} sorted asset IDs:
        """
        response = gpt_client.query(prompt)
        sorted_ids_str = response.strip().split(',')
        sorted_ids = [
            int(id_str.strip())
            for id_str in sorted_ids_str
            if id_str.strip().isdigit()
        ]
        top_assets = df.loc[sorted_ids].head(top_k)
    except Exception as e:
        gr.Error(f"An error occurred while using GPT for ranking: {e}")
        top_assets = candidates.head(top_k)

    items = []
    for row in top_assets.itertuples():
        asset_dir = os.path.join(DATA_ROOT, row.asset_dir)
        video_path = None
        if pd.notna(row.asset_dir) and os.path.exists(asset_dir):
            for f in os.listdir(asset_dir):
                if f.lower().endswith(".mp4"):
                    video_path = os.path.join(asset_dir, f)
                    break
        items.append(
            video_path
            if video_path
            else "https://dummyimage.com/512x512/cccccc/000000&text=No+Preview"
        )

    gr.Info(f"Found {len(items)} assets.")
    return items, gr.update(interactive=True), top_assets


def _extract_mesh_paths(row) -> tuple[str | None, str | None, str]:
    desc = row["description"]
    urdf_path = os.path.join(DATA_ROOT, row["urdf_path"])
    asset_dir = os.path.join(DATA_ROOT, row["asset_dir"])
    visual_mesh_path = None
    collision_mesh_path = None

    if pd.notna(urdf_path) and os.path.exists(urdf_path):
        try:
            tree = ET.parse(urdf_path)
            root = tree.getroot()

            visual_mesh_element = root.find('.//visual/geometry/mesh')
            if visual_mesh_element is not None:
                visual_mesh_filename = visual_mesh_element.get('filename')
                if visual_mesh_filename:
                    glb_filename = (
                        os.path.splitext(visual_mesh_filename)[0] + ".glb"
                    )
                    potential_path = os.path.join(asset_dir, glb_filename)
                    if os.path.exists(potential_path):
                        visual_mesh_path = potential_path

            collision_mesh_element = root.find('.//collision/geometry/mesh')
            if collision_mesh_element is not None:
                collision_mesh_filename = collision_mesh_element.get(
                    'filename'
                )
                if collision_mesh_filename:
                    potential_collision_path = os.path.join(
                        asset_dir, collision_mesh_filename
                    )
                    if os.path.exists(potential_collision_path):
                        collision_mesh_path = potential_collision_path

        except ET.ParseError:
            desc = f"Error: Failed to parse URDF at {urdf_path}. {desc}"
        except Exception as e:
            desc = f"An error occurred while processing URDF: {str(e)}. {desc}"

    return visual_mesh_path, collision_mesh_path, desc


def show_asset_from_gallery(
    evt: gr.SelectData,
    primary: str,
    secondary: str,
    category: str,
    search_query: str,
    gallery_df: pd.DataFrame,
):
    """Parse the selected asset and return raw paths + metadata."""
    index = evt.index

    if search_query and gallery_df is not None and not gallery_df.empty:
        subset = gallery_df
    else:
        if not primary or not secondary:
            return (
                None,  # visual_path
                None,  # collision_path
                "Error: Primary or secondary category not selected.",
                None,  # asset_dir
                None,  # urdf_path
                "N/A",
                "N/A",
                "N/A",
                "N/A",
            )

        subset = df[
            (df["primary_category"] == primary)
            & (df["secondary_category"] == secondary)
        ]
        if category:
            subset = subset[subset["category"] == category]

    if subset.empty or index >= len(subset):
        return (
            None,
            None,
            "Error: Selection index is out of bounds or data is missing.",
            None,
            None,
            "N/A",
            "N/A",
            "N/A",
            "N/A",
        )

    row = subset.iloc[index]

    # In hf_remote mode, pull only what the two mesh viewers need: the visual
    # GLB + collision OBJ (mesh/), plus the tiny URDF used to locate them and
    # show metadata. USD/MJCF/USD-textures are fetched only on button click.
    gr.Info("⏳ Loading 3D model, please wait...")
    rel_dir = row["asset_dir"]
    urdf_name = os.path.basename(row["urdf_path"])
    ensure_hf_files(
        [
            f"{rel_dir}/{urdf_name}",
            f"{rel_dir}/mesh/**",
        ]
    )

    visual_path, collision_path, desc = _extract_mesh_paths(row)

    urdf_path = os.path.join(DATA_ROOT, row["urdf_path"])
    asset_dir = os.path.join(DATA_ROOT, row["asset_dir"])

    # read extra info
    est_type_text = "N/A"
    est_height_text = "N/A"
    est_mass_text = "N/A"
    est_mu_text = "N/A"

    if pd.notna(urdf_path) and os.path.exists(urdf_path):
        try:
            tree = ET.parse(urdf_path)
            root = tree.getroot()
            category_elem = root.find('.//extra_info/category')
            if category_elem is not None and category_elem.text:
                est_type_text = category_elem.text.strip()
            height_elem = root.find('.//extra_info/real_height')
            if height_elem is not None and height_elem.text:
                est_height_text = height_elem.text.strip()
            mass_elem = root.find('.//extra_info/min_mass')
            if mass_elem is not None and mass_elem.text:
                est_mass_text = mass_elem.text.strip()
            mu_elem = root.find('.//collision/gazebo/mu2')
            if mu_elem is not None and mu_elem.text:
                est_mu_text = mu_elem.text.strip()
        except Exception:
            pass

    return (
        visual_path,
        collision_path,
        desc,
        asset_dir,
        urdf_path,
        est_type_text,
        est_height_text,
        est_mass_text,
        est_mu_text,
    )


def render_meshes(
    visual_path: str | None,
    collision_path: str | None,
    switch_viewer: bool,
    req: gr.Request,
):
    session_hash = getattr(req, "session_hash", "default")

    if switch_viewer:
        yield (
            gr.update(value=None),
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=True),
            True,
        )
    else:
        yield (
            gr.update(value=None),
            gr.update(value=None, visible=True),
            gr.update(value=None, visible=False),
            True,
        )

    visual_unique = (
        _unique_path(visual_path, session_hash, "visual")
        if visual_path
        else None
    )
    visual_cam = _camera_position(_visual_radius(visual_path))

    if collision_path:
        collision_unique, collision_r = _colored_collision_path(
            collision_path, session_hash
        )
    else:
        collision_unique, collision_r = None, None
    collision_cam = _camera_position(collision_r)

    if visual_unique or collision_unique:
        gr.Info("✅ 3D model loaded.")

    if switch_viewer:
        yield (
            gr.update(value=visual_unique, camera_position=visual_cam),
            gr.update(value=None, visible=False),
            gr.update(
                value=collision_unique,
                visible=True,
                camera_position=collision_cam,
            ),
            False,
        )
    else:
        yield (
            gr.update(value=visual_unique, camera_position=visual_cam),
            gr.update(
                value=collision_unique,
                visible=True,
                camera_position=collision_cam,
            ),
            gr.update(value=None, visible=False),
            True,
        )


def _rel_dir(asset_dir: str) -> str:
    """Repo-relative asset dir (path under DATA_ROOT), for HF glob patterns."""
    return os.path.relpath(asset_dir, DATA_ROOT)


def _find_urdf_stem(asset_dir: str) -> str | None:
    for f in os.listdir(asset_dir):
        if f.lower().endswith(".urdf"):
            return os.path.splitext(f)[0]
    return None


def _zip_items(
    zip_path: str,
    items: list[tuple[str, str]],
    exclude_suffixes: tuple[str, ...] = (),
) -> str:
    """Write files/dirs into a zip.

    Args:
        zip_path: Output archive path.
        items: ``(src_abspath, arcname)`` pairs. Directories are walked and
            stored under ``arcname``.
        exclude_suffixes: Lowercase filename suffixes to skip when walking
            directories, e.g. ``(".glb",)``.
    """
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for src, arcname in items:
            if os.path.isdir(src):
                for root_d, _, files in os.walk(src):
                    for fn in files:
                        if exclude_suffixes and fn.lower().endswith(
                            exclude_suffixes
                        ):
                            continue
                        fp = os.path.join(root_d, fn)
                        rel = os.path.relpath(fp, src)
                        zf.write(fp, os.path.join(arcname, rel))
            elif os.path.isfile(src):
                zf.write(src, arcname)
    return zip_path


def _fmt_zip_path(
    asset_dir: str, fmt: str, req: gr.Request
) -> tuple[str, str, str]:
    """Build the per-format output zip path named ``<uid>_<fmt>.zip``.

    Returns ``(zip_path, uid, zip_name)``.
    """
    uid = os.path.basename(os.path.normpath(asset_dir))
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)
    zip_name = f"{uid}_{fmt}.zip"
    return os.path.join(user_dir, zip_name), uid, zip_name


def download_urdf(asset_dir: str, req: gr.Request) -> str | None:
    """Package the ``.urdf`` file(s) and the ``mesh/`` folder (minus ``.glb``).

    The visual/collision meshes referenced by the URDF live in ``mesh/`` as
    ``.obj`` + material files; the ``.glb`` (a separate viewer asset) and all
    other formats (USD/MJCF/video/...) are intentionally excluded.
    """
    if not asset_dir or not os.path.isdir(asset_dir):
        gr.Warning("Please select an asset first.")
        return None
    gr.Info("⏳ Preparing URDF asset for download, please wait...")
    rel = _rel_dir(asset_dir)
    ensure_hf_files([f"{rel}/*.urdf", f"{rel}/mesh/**"])

    items: list[tuple[str, str]] = [
        (os.path.join(asset_dir, f), f)
        for f in os.listdir(asset_dir)
        if f.lower().endswith(".urdf")
    ]
    mesh_dir = os.path.join(asset_dir, "mesh")
    if os.path.isdir(mesh_dir):
        items.append((mesh_dir, "mesh"))

    zip_path, uid, zip_name = _fmt_zip_path(asset_dir, "urdf", req)
    _zip_items(zip_path, items, exclude_suffixes=(".glb",))
    gr.Info(f"✅ {zip_name} is ready.")
    return zip_path


def download_usd(asset_dir: str, req: gr.Request) -> str | None:
    """Package the ``.usd`` file together with its ``textures/`` folder."""
    if not asset_dir or not os.path.isdir(asset_dir):
        gr.Warning("Please select an asset first.")
        return None

    gr.Info("⏳ Preparing USD asset for download, please wait...")
    rel = _rel_dir(asset_dir)
    stem = _find_urdf_stem(asset_dir)
    if stem is not None:
        ensure_hf_files([f"{rel}/{stem}.usd", f"{rel}/textures/**"])
    else:
        ensure_hf_files([f"{rel}/*.usd", f"{rel}/textures/**"])

    items: list[tuple[str, str]] = []
    for f in os.listdir(asset_dir):
        if f.lower().endswith(".usd"):
            items.append((os.path.join(asset_dir, f), f))
    tex_dir = os.path.join(asset_dir, "textures")
    if os.path.isdir(tex_dir):
        items.append((tex_dir, "textures"))

    if not any(s.lower().endswith(".usd") for s, _ in items):
        gr.Warning("No USD file available for this asset.")
        return None

    zip_path, uid, zip_name = _fmt_zip_path(asset_dir, "usd", req)
    _zip_items(zip_path, items)
    gr.Info(f"✅ {zip_name} is ready.")
    return zip_path


def download_mjcf(asset_dir: str, req: gr.Request) -> str | None:
    """Package the ``mjcf/`` folder."""
    if not asset_dir or not os.path.isdir(asset_dir):
        gr.Warning("Please select an asset first.")
        return None

    gr.Info("⏳ Preparing MJCF asset for download, please wait...")
    ensure_hf_files([f"{_rel_dir(asset_dir)}/mjcf/**"])
    mjcf_dir = os.path.join(asset_dir, "mjcf")
    if not os.path.isdir(mjcf_dir):
        gr.Warning("No MJCF folder available for this asset.")
        return None

    zip_path, uid, zip_name = _fmt_zip_path(asset_dir, "mjcf", req)
    _zip_items(zip_path, [(mjcf_dir, "mjcf")])
    gr.Info(f"✅ {zip_name} is ready.")
    return zip_path


# Download buttons keyed by format; order matches the outputs list used when
# locking/unlocking them together.
_DL_KEYS = ("urdf", "usd", "mjcf")
_DL_LABELS = {
    "urdf": "⬇️ Download URDF",
    "usd": "⬇️ Download USD",
    "mjcf": "⬇️ Download MJCF",
}


def _lock_downloads(active: str) -> tuple:
    """Disable all download buttons; show a persistent hint on the clicked one.

    Keeps the user from spamming clicks while the (possibly slow) zip is being
    prepared, since a toast alone neither blocks clicks nor stays put.
    """
    return tuple(
        gr.update(interactive=False, label=f"⏳ Preparing {k.upper()}...")
        if k == active
        else gr.update(interactive=False)
        for k in _DL_KEYS
    )


def _unlock_downloads() -> tuple:
    """Re-enable buttons and restore labels; keep the freshly built file value."""
    return tuple(
        gr.update(interactive=True, label=_DL_LABELS[k]) for k in _DL_KEYS
    )


def _reset_downloads() -> tuple:
    """Enable + restore labels + clear stale file value (used on asset switch)."""
    return tuple(
        gr.update(interactive=True, label=_DL_LABELS[k], value=None)
        for k in _DL_KEYS
    )


def start_session(req: gr.Request) -> None:
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)


def end_session(req: gr.Request) -> None:
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    if os.path.exists(user_dir):
        shutil.rmtree(user_dir)


# --- UI ---
with gr.Blocks(
    theme=custom_theme,
    css=css,
    title="3D Asset Library",
) as demo:
    gr.HTML(lighting_css, visible=False)
    gr.Markdown(
        """
        ## 🏛️ ***EmbodiedGen***: 3D Asset Gallery Explorer

        **🔖 Version**: {VERSION}
        <p style="display: flex; gap: 10px; flex-wrap: nowrap;">
            <a href="https://horizonrobotics.github.io/EmbodiedGen">
                <img alt="📖 Documentation" src="https://img.shields.io/badge/📖-Documentation-blue">
            </a>
            <a href="https://arxiv.org/abs/2506.10600">
                <img alt="📄 arXiv" src="https://img.shields.io/badge/📄-arXiv-b31b1b">
            </a>
            <a href="https://github.com/HorizonRobotics/EmbodiedGen">
                <img alt="💻 GitHub" src="https://img.shields.io/badge/GitHub-000000?logo=github">
            </a>
            <a href="https://www.youtube.com/watch?v=rG4odybuJRk">
                <img alt="🎥 Video" src="https://img.shields.io/badge/🎥-Video-red">
            </a>
        </p>

        Browse and visualize the EmbodiedGen 3D asset database. Select categories to filter and click on a preview to load the model.

        """.format(VERSION=VERSION),
        elem_classes=["header"],
    )

    primary_list = get_primary_categories()
    primary_val = primary_list[0] if primary_list else None
    secondary_list = get_secondary_categories(primary_val)
    secondary_val = secondary_list[0] if secondary_list else None
    category_list = get_categories(primary_val, secondary_val)
    category_val = category_list[0] if category_list else None
    asset_folder = gr.State(value=None)
    gallery_df_state = gr.State()

    switch_viewer_state = gr.State(value=False)

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=350):
            with gr.Group():
                gr.Markdown("### Search Asset with Descriptions")
                search_box = gr.Textbox(
                    label="🔎 Enter your search query",
                    placeholder="e.g., 'a red chair with four legs'",
                    interactive=GPT_AVAILABLE,
                )
                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=50,
                    value=10,
                    step=1,
                    label="Number of results",
                    interactive=GPT_AVAILABLE,
                )
                search_button = gr.Button(
                    "Search", variant="primary", interactive=GPT_AVAILABLE
                )
                if not GPT_AVAILABLE:
                    gr.Markdown(
                        "<p style='color: #ff4b4b;'>⚠️ GPT client not available. Search is disabled.</p>"
                    )

            with gr.Group():
                gr.Markdown("### Select Asset Category")
                primary = gr.Dropdown(
                    choices=primary_list,
                    value=primary_val,
                    label="🗂️ Primary Category",
                )
                secondary = gr.Dropdown(
                    choices=secondary_list,
                    value=secondary_val,
                    label="📂 Secondary Category",
                )
                category = gr.Dropdown(
                    choices=category_list,
                    value=category_val,
                    label="🏷️ Asset Category",
                )

            with gr.Group():
                initial_assets, _, initial_df = get_assets(
                    primary_val, secondary_val, category_val
                )
                gallery = gr.Gallery(
                    value=initial_assets,
                    label="🖼️ Asset Previews",
                    columns=3,
                    height="auto",
                    allow_preview=True,
                    elem_id="asset-gallery",
                    interactive=bool(category_val),
                )

        with gr.Column(scale=2, min_width=500):
            with gr.Group():
                with gr.Tabs():
                    with gr.TabItem("Visual Mesh") as t1:
                        viewer = gr.Model3D(
                            label="🧊 3D Model Viewer",
                            height=380,
                            clear_color=[0.95, 0.95, 0.95],
                            elem_id="visual_mesh",
                        )
                    with gr.TabItem("Collision Mesh") as t2:
                        collision_viewer_a = gr.Model3D(
                            label="🧊 Collision Mesh",
                            height=380,
                            clear_color=[0.95, 0.95, 0.95],
                            elem_id="collision_mesh_a",
                            visible=True,
                        )
                        collision_viewer_b = gr.Model3D(
                            label="🧊 Collision Mesh",
                            height=380,
                            clear_color=[0.95, 0.95, 0.95],
                            elem_id="collision_mesh_b",
                            visible=False,
                        )

                t1.select(
                    fn=lambda: None,
                    js="() => { window.dispatchEvent(new Event('resize')); }",
                )
                t2.select(
                    fn=lambda: None,
                    js="() => { window.dispatchEvent(new Event('resize')); }",
                )

                with gr.Row():
                    est_type_text = gr.Textbox(
                        label="Asset category", interactive=False
                    )
                    est_height_text = gr.Textbox(
                        label="Real height(.m)", interactive=False
                    )
                    est_mass_text = gr.Textbox(
                        label="Mass(.kg)", interactive=False
                    )
                    est_mu_text = gr.Textbox(
                        label="Friction coefficient", interactive=False
                    )
                with gr.Row():
                    desc_box = gr.Textbox(
                        label="📝 Asset Description", interactive=False
                    )
                with gr.Accordion(label="Asset Details", open=False):
                    urdf_file = gr.Textbox(
                        label="URDF File Path", interactive=False, lines=2
                    )
                with gr.Row():
                    # DownloadButtons that build their zip into their own value
                    # on click; a chained JS handler then triggers the browser
                    # download from that value (one-click). If the JS ever
                    # fails, the value is still populated, so a second manual
                    # click downloads it via the button's native behavior.
                    urdf_dl_btn = gr.DownloadButton(
                        label="⬇️ Download URDF",
                        variant="primary",
                        interactive=False,
                    )
                    usd_dl_btn = gr.DownloadButton(
                        label="⬇️ Download USD",
                        variant="primary",
                        interactive=False,
                    )
                    mjcf_dl_btn = gr.DownloadButton(
                        label="⬇️ Download MJCF",
                        variant="primary",
                        interactive=False,
                    )

    search_button.click(
        fn=search_assets,
        inputs=[search_box, top_k_slider],
        outputs=[gallery, gallery, gallery_df_state],
    )
    search_box.submit(
        fn=search_assets,
        inputs=[search_box, top_k_slider],
        outputs=[gallery, gallery, gallery_df_state],
    )

    def update_on_primary_change(p):
        s_choices = get_secondary_categories(p)
        initial_assets, gallery_update, initial_df = get_assets(p, None, None)
        return (
            gr.update(choices=s_choices, value=None),
            gr.update(choices=[], value=None),
            initial_assets,
            gallery_update,
            initial_df,
        )

    def update_on_secondary_change(p, s):
        c_choices = get_categories(p, s)
        asset_previews, gallery_update, gallery_df = get_assets(p, s, None)
        return (
            gr.update(choices=c_choices, value=None),
            asset_previews,
            gallery_update,
            gallery_df,
        )

    def update_assets(p, s, c):
        asset_previews, gallery_update, gallery_df = get_assets(p, s, c)
        return asset_previews, gallery_update, gallery_df

    primary.change(
        fn=update_on_primary_change,
        inputs=[primary],
        outputs=[secondary, category, gallery, gallery, gallery_df_state],
    )
    secondary.change(
        fn=update_on_secondary_change,
        inputs=[primary, secondary],
        outputs=[category, gallery, gallery, gallery_df_state],
    )
    category.change(
        fn=update_assets,
        inputs=[primary, secondary, category],
        outputs=[gallery, gallery, gallery_df_state],
    )

    gallery.select(
        fn=show_asset_from_gallery,
        inputs=[primary, secondary, category, search_box, gallery_df_state],
        outputs=[
            (visual_path_state := gr.State()),
            (collision_path_state := gr.State()),
            desc_box,
            asset_folder,
            urdf_file,
            est_type_text,
            est_height_text,
            est_mass_text,
            est_mu_text,
        ],
    ).then(
        fn=render_meshes,
        inputs=[visual_path_state, collision_path_state, switch_viewer_state],
        outputs=[
            viewer,
            collision_viewer_a,
            collision_viewer_b,
            switch_viewer_state,
        ],
    ).success(
        fn=_reset_downloads,
        outputs=[urdf_dl_btn, usd_dl_btn, mjcf_dl_btn],
    )

    # After the zip is built into the button's own value, pass that file value
    # straight into the JS handler and trigger the browser download from its
    # URL. Reading the value via `inputs` avoids DOM-timing/selector issues.
    download_js = """
    (f) => {
        if (f && f.url) {
            const a = document.createElement('a');
            a.href = f.url;
            a.download = (f.orig_name || 'asset.zip').split('/').pop();
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        }
        return [];
    }
    """

    dl_btns = [urdf_dl_btn, usd_dl_btn, mjcf_dl_btn]

    # Each flow: lock all buttons (prevent spam clicks) -> build the zip ->
    # JS-trigger the browser download -> unlock. Selecting another asset
    # resets the buttons independently (see `.success` above).
    urdf_dl_btn.click(
        fn=lambda: _lock_downloads("urdf"), outputs=dl_btns
    ).then(
        fn=download_urdf, inputs=[asset_folder], outputs=[urdf_dl_btn]
    ).then(fn=lambda *a: None, inputs=[urdf_dl_btn], js=download_js).then(
        fn=_unlock_downloads, outputs=dl_btns
    )

    usd_dl_btn.click(fn=lambda: _lock_downloads("usd"), outputs=dl_btns).then(
        fn=download_usd, inputs=[asset_folder], outputs=[usd_dl_btn]
    ).then(fn=lambda *a: None, inputs=[usd_dl_btn], js=download_js).then(
        fn=_unlock_downloads, outputs=dl_btns
    )

    mjcf_dl_btn.click(
        fn=lambda: _lock_downloads("mjcf"), outputs=dl_btns
    ).then(
        fn=download_mjcf, inputs=[asset_folder], outputs=[mjcf_dl_btn]
    ).then(fn=lambda *a: None, inputs=[mjcf_dl_btn], js=download_js).then(
        fn=_unlock_downloads, outputs=dl_btns
    )

    demo.load(start_session)
    demo.unload(end_session)


if __name__ == "__main__":
    # Serve gallery videos / meshes that live under DATA_ROOT (outside cwd).
    demo.launch(
        server_port=8088,
        allowed_paths=[os.path.abspath(DATA_ROOT)],
    )
