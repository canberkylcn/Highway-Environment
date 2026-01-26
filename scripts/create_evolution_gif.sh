#!/bin/bash
# Convert MP4 evolution videos to GIF for GitHub README embedding
# Usage: ./scripts/create_evolution_gif.sh highway

ENV=${1:-highway}
VIDEO_DIR="logs/videos/${ENV}-v0"
OUTPUT_DIR="assets/videos"

mkdir -p "$OUTPUT_DIR"

# Convert each video to GIF
for prefix in "1_untrained" "2_half_trained" "3_fully_trained"; do
    input="${VIDEO_DIR}/${prefix}-episode-0.mp4"
    output="${OUTPUT_DIR}/${ENV}_${prefix}.gif"
    
    if [ -f "$input" ]; then
        echo "Converting $input to $output..."
        ffmpeg -i "$input" -vf "fps=10,scale=640:-1:flags=lanczos" -loop 0 "$output"
        echo "✓ Created $output"
    else
        echo "⚠ Warning: $input not found"
    fi
done

echo ""
echo "Done! Add these to README.md:"
echo "![Untrained](assets/videos/${ENV}_1_untrained.gif)"
echo "![Half-Trained](assets/videos/${ENV}_2_half_trained.gif)"
echo "![Fully Trained](assets/videos/${ENV}_3_fully_trained.gif)"
