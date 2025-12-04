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

import shutil
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Tuple

import gradio as gr
import pandas as pd
from app_style import custom_theme, lighting_css
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

if RUNNING_MODE == "local":
    DATA_ROOT = "/horizon-bucket/robot_lab/datasets/embodiedgen/assets"
elif RUNNING_MODE == "hf_remote":
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="HorizonRobotics/EmbodiedGenData",
        repo_type="dataset",
        allow_patterns=f"dataset/**",
        local_dir="EmbodiedGenData",
        local_dir_use_symlinks=False,
    )
    DATA_ROOT = "EmbodiedGenData/dataset"
else:
    raise ValueError(
        f"Unknown RUNNING_MODE: {RUNNING_MODE}, must be 'local' or 'hf_remote'."
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


# --- Mesh extraction ---
def _extract_mesh_paths(row) -> Tuple[str | None, str | None, str]:
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
    collision_unique = (
        _unique_path(collision_path, session_hash, "collision")
        if collision_path
        else None
    )

    if switch_viewer:
        yield (
            gr.update(value=visual_unique),
            gr.update(value=None, visible=False),
            gr.update(value=collision_unique, visible=True),
            False,
        )
    else:
        yield (
            gr.update(value=visual_unique),
            gr.update(value=collision_unique, visible=True),
            gr.update(value=None, visible=False),
            True,
        )


def create_asset_zip(asset_dir: str, req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)

    asset_folder_name = os.path.basename(os.path.normpath(asset_dir))
    zip_path_base = os.path.join(user_dir, asset_folder_name)

    archive_path = shutil.make_archive(
        base_name=zip_path_base, format='zip', root_dir=asset_dir
    )
    gr.Info(f"✅ {asset_folder_name}.zip is ready and can be downloaded.")

    return archive_path


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

        """.format(
            VERSION=VERSION
        ),
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
                            height=500,
                            clear_color=[0.95, 0.95, 0.95],
                            elem_id="visual_mesh",
                        )
                    with gr.TabItem("Collision Mesh") as t2:
                        collision_viewer_a = gr.Model3D(
                            label="🧊 Collision Mesh",
                            height=500,
                            clear_color=[0.95, 0.95, 0.95],
                            elem_id="collision_mesh_a",
                            visible=True,
                        )
                        collision_viewer_b = gr.Model3D(
                            label="🧊 Collision Mesh",
                            height=500,
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
                    extract_btn = gr.Button(
                        "📥 Extract Asset",
                        variant="primary",
                        interactive=False,
                    )
                    download_btn = gr.DownloadButton(
                        label="⬇️ Download Asset",
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
        lambda: (gr.Button(interactive=True), gr.Button(interactive=False)),
        outputs=[extract_btn, download_btn],
    )

    extract_btn.click(
        fn=create_asset_zip, inputs=[asset_folder], outputs=[download_btn]
    ).success(fn=lambda: gr.update(interactive=True), outputs=download_btn)

    demo.load(start_session)
    demo.unload(end_session)


if __name__ == "__main__":
    demo.launch(
        server_port=8088,
        allowed_paths=[
            "/horizon-bucket/robot_lab/datasets/embodiedgen/assets"
        ],
    )
