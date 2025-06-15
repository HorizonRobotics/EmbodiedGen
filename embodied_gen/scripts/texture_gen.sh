#!/bin/bash

while [[ $# -gt 0 ]]; do
    case $1 in
        --mesh_path)
            mesh_path="$2"
            shift 2
            ;;
        --prompt)
            prompt="$2"
            shift 2
            ;;
        --uuid)
            uuid="$2"
            shift 2
            ;;
        --output_root)
            output_root="$2"
            shift 2
            ;;
        *)
            echo "unknown: $1"
            exit 1
            ;;
    esac
done


if [[ -z "$mesh_path" || -z "$prompt" || -z "$uuid" || -z "$output_root" ]]; then
    echo "params missing"
    echo "usage: bash run.sh --mesh_path <path> --prompt <text> --uuid <id> --output_root <path>"
    exit 1
fi

# Step 1: drender-cli for condition rendering
drender-cli --mesh_path ${mesh_path} \
    --output_root ${output_root}/condition \
    --uuid ${uuid}

# Step 2: multi-view rendering
python embodied_gen/scripts/render_mv.py \
    --index_file "${output_root}/condition/index.json" \
    --controlnet_cond_scale 0.7 \
    --guidance_scale 9 \
    --strength 0.9 \
    --num_inference_steps 40 \
    --ip_adapt_scale 0 \
    --ip_img_path None \
    --uid ${uuid} \
    --prompt "${prompt}" \
    --save_dir "${output_root}/multi_view" \
    --sub_idxs "[[0,1,2],[3,4,5]]" \
    --seed 0

# Step 3: backprojection
backproject-cli --mesh_path ${mesh_path} \
    --color_path ${output_root}/multi_view/color_sample0.png \
    --output_path "${output_root}/texture_mesh/${uuid}.obj" \
    --save_glb_path "${output_root}/texture_mesh/${uuid}.glb" \
    --skip_fix_mesh \
    --delight \
    --no_save_delight_img

# Step 4: final rendering of textured mesh
drender-cli --mesh_path "${output_root}/texture_mesh/${uuid}.obj" \
    --output_root ${output_root}/texture_mesh \
    --num_images 90 \
    --elevation 20 \
    --with_mtl \
    --gen_color_mp4 \
    --pbr_light_factor 1.2

# Organize folders
rm -rf ${output_root}/condition
video_path="${output_root}/texture_mesh/${uuid}/color.mp4"
if [ -f "${video_path}" ]; then
    cp "${video_path}" "${output_root}/texture_mesh/color.mp4"
    echo "Resave video to ${output_root}/texture_mesh/color.mp4"
fi
rm -rf ${output_root}/texture_mesh/${uuid}