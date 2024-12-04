#!/bin/bash

# Function to print usage information
usage() {
    echo "Usage: $0 -i input_file.wav -o output_file.wav -s {8000,16000} [-h]"
    echo "  -i: Input WAV file"
    echo "  -o: Output WAV file"
    echo "  -s: Output sample rate (8000 or 16000)"
    echo "  -h: Use high-quality Sox resampler (optional)"
    exit 1
}

# Parse command-line arguments
while getopts "i:o:s:h" opt; do
    case $opt in
        i) input_file="$OPTARG" ;;
        o) output_file="$OPTARG" ;;
        s) output_rate="$OPTARG" ;;
        h) use_sox=1 ;;
        *) usage ;;
    esac
done

# Check if required arguments are provided
if [ -z "$input_file" ] || [ -z "$output_file" ] || [ -z "$output_rate" ]; then
    usage
fi

# Check if output rate is valid
if [ "$output_rate" != "8000" ] && [ "$output_rate" != "16000" ]; then
    echo "Error: Output sample rate must be either 8000 or 16000"
    exit 1
fi

# Get input sample rate
input_rate=$(sox --i -r "$input_file")

# Function to convert WAV to RAW
wav_to_raw() {
    sox "$1" -c 1 -b 16 -e signed-integer -L "$2"
}

# Function to convert RAW to WAV
raw_to_wav() {
    sox -r "$1" -c 1 -b 16 -e signed-integer -L "$2" "$3"
}

# Temporary files
temp_dir=$(mktemp -d)
raw_input="$temp_dir/input.raw"
raw_output="$temp_dir/output.raw"

# Convert input WAV to RAW
wav_to_raw "$input_file" "$raw_input"

# Resample using Sox or test_resample
if [ -n "$use_sox" ]; then
    # Use Sox for high-quality resampling
    sox "$input_file" -r "$output_rate" "$output_file"
else
    # Use test_resample
    if [ "$input_rate" == "8000" ] || [ "$input_rate" == "16000" ]; then
        # Use test_resample directly
        build/test_resample "$raw_input" "$raw_output" "$input_rate" "$output_rate"
    else
        # Resample to 16000 using Sox, then use test_resample
        temp_16k="$temp_dir/temp_16k.raw"
        sox "$input_file" -r 16000 -c 1 -b 16 -e signed-integer -L "$temp_16k"
        build/test_resample "$temp_16k" "$raw_output" 16000 "$output_rate"
    fi
    
    # Convert RAW output to WAV
    raw_to_wav "$output_rate" "$raw_output" "$output_file"
fi

# Clean up temporary files
rm -rf "$temp_dir"

echo "Resampling complete: $input_file -> $output_file"