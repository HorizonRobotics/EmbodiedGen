#!/bin/bash

# Initialize variables
prompts=()
asset_types=()
output_root=""
seed=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --prompts)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                prompts+=("$1")
                shift
            done
            ;;
        --asset_types)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                asset_types+=("$1")
                shift
            done
            ;;
        --output_root)
            output_root="$2"
            shift 2
            ;;
        --seed)
            seed="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ ${#prompts[@]} -eq 0 || -z "$output_root" ]]; then
    echo "Missing required arguments."
    echo "Usage: bash run_text2asset3d.sh --prompts \"Prompt1\" \"Prompt2\" \
    --asset_types \"type1\" \"type2\" --seed <seed_value> --output_root <path>"
    exit 1
fi

# If no asset_types provided, default to ""
if [[ ${#asset_types[@]} -eq 0 ]]; then
    for (( i=0; i<${#prompts[@]}; i++ )); do
        asset_types+=("")
    done
fi

# Ensure the number of asset_types matches the number of prompts
if [[ ${#prompts[@]} -ne ${#asset_types[@]} ]]; then
    echo "The number of asset types must match the number of prompts."
    exit 1
fi

# Print arguments (for debugging)
echo "Prompts:"
for p in "${prompts[@]}"; do
    echo "   - $p"
done
# echo "Asset types:"
# for at in "${asset_types[@]}"; do
#     echo "   - $at"
# done
echo "Output root: ${output_root}"
echo "Seed: ${seed}"

# Concatenate prompts and asset types for Python command
prompt_args=""
asset_type_args=""
for i in "${!prompts[@]}"; do
    prompt_args+="\"${prompts[$i]}\" "
    asset_type_args+="\"${asset_types[$i]}\" "
done


# Step 1: Text-to-Image
echo ${prompt_args}
eval python3 embodied_gen/scripts/text2image.py \
    --prompts ${prompt_args} \
    --output_root "${output_root}/images" \
    --seed ${seed}

# Step 2: Image-to-3D
python3 embodied_gen/scripts/imageto3d.py \
    --image_root "${output_root}/images" \
    --output_root "${output_root}/asset3d" \
    --asset_type ${asset_type_args}
