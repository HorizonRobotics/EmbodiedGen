export CUDA_VISIBLE_DEVICES=2
source outputs/env.sh

output_dir="outputs/test_integration"

rm -rf ${output_dir}

text3d-cli --prompts "small bronze figurine of a lion" \
    --n_image_retry 2 --n_asset_retry 2 --n_pipe_retry 1 --seed_img 0 \
    --output_root ${output_dir}/textto3d

texture-cli --mesh_path "apps/assets/example_texture/meshes/horse.obj" \
--prompt "A gray horse head with flying mane and brown eyes" \
--output_root "${output_dir}/texture_gen" \
--seed 0

scene3d-cli --prompts "Art studio with easel and canvas" \
--output_dir ${output_dir}/bg_scenes/ \
--seed 0 --gs3d.max_steps 4000 \
--disable_pano_check

layout-cli --task_descs "Place the pen in the mug on the desk" \
--bg_list "outputs/bg_scenes/scene_list.txt" \
--output_root "${output_dir}/layouts_gen" --insert_robot


python embodied_gen/scripts/compose_layout.py \
--layout_path "outputs/layouts_gen/task_0000/layout.json" \
--output_dir "outputs/layouts_gen/task_0000/recompose" --insert_robot