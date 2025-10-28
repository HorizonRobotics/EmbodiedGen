import os
import shutil
import xml.etree.ElementTree as ET

import gradio as gr
import pandas as pd
from app_style import custom_theme, lighting_css

# --- Configuration & Data Loading ---
VERSION = "v0.1.5"
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
#lighter_mesh canvas {
    filter: brightness(2.2) !important;
}
</style>
"""


# --- Helper Functions ---
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
        return [], gr.update(interactive=False)

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

    return items, gr.update(interactive=True)


def show_asset_from_gallery(
    evt: gr.SelectData, primary: str, secondary: str, category: str
):
    index = evt.index
    subset = df[
        (df["primary_category"] == primary)
        & (df["secondary_category"] == secondary)
    ]
    if category:
        subset = subset[subset["category"] == category]

    est_type_text = "N/A"
    est_height_text = "N/A"
    est_mass_text = "N/A"
    est_mu_text = "N/A"

    if index >= len(subset):
        return (
            None,
            "Error: Selection index is out of bounds.",
            None,
            None,
            est_type_text,
            est_height_text,
            est_mass_text,
            est_mu_text,
        )

    row = subset.iloc[index]
    desc = row["description"]
    urdf_path = os.path.join(DATA_ROOT, row["urdf_path"])
    asset_dir = os.path.join(DATA_ROOT, row["asset_dir"])
    mesh_to_display = None
    if pd.notna(urdf_path) and os.path.exists(urdf_path):
        try:
            tree = ET.parse(urdf_path)
            root = tree.getroot()

            mesh_element = root.find('.//visual/geometry/mesh')
            if mesh_element is not None:
                mesh_filename = mesh_element.get('filename')
                if mesh_filename:
                    glb_filename = os.path.splitext(mesh_filename)[0] + ".glb"
                    potential_path = os.path.join(asset_dir, glb_filename)
                    if os.path.exists(potential_path):
                        mesh_to_display = potential_path

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

        except ET.ParseError:
            desc = f"Error: Failed to parse URDF at {urdf_path}. {desc}"
        except Exception as e:
            desc = f"An error occurred while processing URDF: {str(e)}. {desc}"

    return (
        gr.update(value=mesh_to_display),
        desc,
        asset_dir,
        urdf_path,
        est_type_text,
        est_height_text,
        est_mass_text,
        est_mu_text,
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


# --- Gradio UI Definition ---
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
            <a href="https://horizonrobotics.github.io/robot_lab/embodied_gen/index.html">
                <img alt="🌐 Project Page" src="https://img.shields.io/badge/🌐-Project_Page-blue">
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

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=350):
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
                gallery = gr.Gallery(
                    value=get_assets(primary_val, secondary_val, category_val)[
                        0
                    ],
                    label="🖼️ Asset Previews",
                    columns=3,
                    height="auto",
                    allow_preview=True,
                    elem_id="asset-gallery",
                    interactive=bool(category_val),
                )

        with gr.Column(scale=2, min_width=500):
            with gr.Group():
                viewer = gr.Model3D(
                    label="🧊 3D Model Viewer",
                    height=500,
                    clear_color=[0.95, 0.95, 0.95],
                    elem_id="lighter_mesh",
                )
                with gr.Row():
                    # TODO: Add more asset details if needed
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
                with gr.Accordion(label="Asset Details", open=False):
                    desc_box = gr.Textbox(
                        label="📝 Asset Description", interactive=False
                    )
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

    def update_on_primary_change(p):
        s_choices = get_secondary_categories(p)
        return (
            gr.update(choices=s_choices, value=None),
            gr.update(choices=[], value=None),
            [],
            gr.update(interactive=False),
        )

    def update_on_secondary_change(p, s):
        c_choices = get_categories(p, s)
        return (
            gr.update(choices=c_choices, value=None),
            [],
            gr.update(interactive=False),
        )

    def update_on_secondary_change(p, s):
        c_choices = get_categories(p, s)
        asset_previews, gallery_update = get_assets(p, s, None)
        return (
            gr.update(choices=c_choices, value=None),
            asset_previews,
            gallery_update,
        )

    primary.change(
        fn=update_on_primary_change,
        inputs=[primary],
        outputs=[secondary, category, gallery, gallery],
    )

    secondary.change(
        fn=update_on_secondary_change,
        inputs=[primary, secondary],
        outputs=[category, gallery, gallery],
    )

    category.change(
        fn=get_assets,
        inputs=[primary, secondary, category],
        outputs=[gallery, gallery],
    )

    gallery.select(
        fn=show_asset_from_gallery,
        inputs=[primary, secondary, category],
        outputs=[
            viewer,
            desc_box,
            asset_folder,
            urdf_file,
            est_type_text,
            est_height_text,
            est_mass_text,
            est_mu_text,
        ],
    ).success(
        lambda: tuple(
            [
                gr.Button(interactive=True),
                gr.Button(interactive=False),
            ]
        ),
        outputs=[extract_btn, download_btn],
    )

    extract_btn.click(
        fn=create_asset_zip, inputs=[asset_folder], outputs=[download_btn]
    ).success(
        fn=lambda: gr.update(interactive=True),
        outputs=download_btn,
    )

    demo.load(start_session)
    demo.unload(end_session)


if __name__ == "__main__":
    demo.launch(
        server_name="10.34.8.82",
        server_port=8088,
        allowed_paths=[
            "/horizon-bucket/robot_lab/datasets/embodiedgen/assets"
        ],
    )
