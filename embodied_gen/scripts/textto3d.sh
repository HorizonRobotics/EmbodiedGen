#!/bin/bash

# Initialize variables
prompts=()
output_root=""

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
        --output_root)
            output_root="$2"
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
    echo "Usage: bash run_text2asset3d.sh --prompts \"Prompt1\" \"Prompt2\" --output_root <path>"
    exit 1
fi

# Print arguments (for debugging)
echo "Prompts:"
for p in "${prompts[@]}"; do
    echo "   - $p"
done
echo "Output root: ${output_root}"

# Concatenate prompts for Python command
prompt_args=""
for p in "${prompts[@]}"; do
    prompt_args+="\"$p\" "
done

# Step 1: Text-to-Image
eval python3 embodied_gen/scripts/text2image.py \
    --prompts ${prompt_args} \
    --output_root "${output_root}/images"

# Step 2: Image-to-3D
python3 embodied_gen/scripts/imageto3d.py \
    --image_root "${output_root}/images" \
    --output_root "${output_root}/asset3d"
