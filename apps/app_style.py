from gradio.themes import Soft
from gradio.themes.utils.colors import gray, neutral, slate, stone, teal, zinc

lighting_css = """
<style>
#lighter_mesh canvas {
    filter: brightness(2.0) !important;
}
</style>
"""

image_css = """
<style>
.image_fit .image-frame {
object-fit: contain !important;
height: 100% !important;
}
</style>
"""

custom_theme = Soft(
    primary_hue=stone,
    secondary_hue=gray,
    radius_size="md",
    text_size="sm",
    spacing_size="sm",
)
