#!/bin/bash
set -e

# Function to display usage information
usage() {
    echo "Usage: denoise.sh <input> <output> <naturalness> <mingain> [output_sample_rate]"
    echo "  input: Input file or folder"
    echo "  output: Output file (if input is a file) or folder (if input is a folder)"
    echo "  naturalness: [0,1] (0 = max noise suppression, 1 = more natural)"
    echo "  mingain: [-60, 0] dB (0 dB = no suppression)"
    echo "  output_sample_rate: (optional) Output sample rate in Hz (default: same as input)"
}

# Function to process a single file
process_file() {
    local input_file="$1"
    local output_file="$2"
    local naturalness="$3"
    local mingain="$4"
    local output_sample_rate="$5"

    # Get input sample rate
    local input_sample_rate=$(sox --i -r "$input_file")

    # Set output sample rate to input sample rate if not specified
    if [ -z "$output_sample_rate" ]; then
        output_sample_rate="$input_sample_rate"
    fi

    # Determine if resampling is needed
    if [ "$input_sample_rate" -eq 8000 ] || [ "$input_sample_rate" -eq 16000 ]; then
        # No resampling needed
        sox "$input_file" -c 1 -b 16 -e signed-integer -L "$RAW_INPUT_FNAME"
        process_sample_rate="$input_sample_rate"
    else
        # Resample to 16kHz
        sox "$input_file" -c 1 -b 16 -e signed-integer -L -r 16000 "$RAW_INPUT_FNAME"
        process_sample_rate=16000
    fi

    # Run denoising algorithm
    "$TESTNR" -n "$naturalness" -m "$mingain" -i "$RAW_INPUT_FNAME" -o "$RAW_OUTPUT_FNAME" -r "$process_sample_rate"

    # Convert output to desired format and sample rate
    sox -r "$process_sample_rate" -b 16 -e signed-integer -L "$RAW_OUTPUT_FNAME" -r "$output_sample_rate" -b 16 -e signed-integer "$output_file"

    # Clean up temporary files
    rm -f "$RAW_INPUT_FNAME" "$RAW_OUTPUT_FNAME"

    echo "Processed: $input_file -> $output_file"
}

# Check for correct number of arguments
if [ $# -lt 4 ] || [ $# -gt 5 ]; then
    usage
    exit 1
fi

# Assign arguments to variables
input="$1"
output="$2"
naturalness="$3"
mingain="$4"
output_sample_rate="${5:-}" # Optional argument, empty if not provided

# Set constants
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
RAW_INPUT_FNAME="/tmp/_temp_input.raw"
RAW_OUTPUT_FNAME="/tmp/_temp_output.raw"
TESTNR="$SCRIPT_DIR/../build/testnr"

# Check if input is a file or directory
if [ -f "$input" ]; then
    # Input is a file
    process_file "$input" "$output" "$naturalness" "$mingain" "$output_sample_rate"
elif [ -d "$input" ]; then
    # Input is a directory
    if [ ! -d "$output" ]; then
        mkdir -p "$output"
    fi

    for file in "$input"/*.wav; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            output_file="$output/$filename"
            process_file "$file" "$output_file" "$naturalness" "$mingain" "$output_sample_rate"
        fi
    done
else
    echo "Error: Input '$input' is not a valid file or directory."
    exit 1
fi

echo "Denoising completed."