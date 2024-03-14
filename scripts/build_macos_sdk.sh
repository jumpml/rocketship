#!/bin/bash

# Destination directory
dest_dir="/tmp/jumpml_macos_sdk"

# Create the destination directory and its subdirectories
mkdir -p "$dest_dir"/{include,src,build,data}

# List of files to copy (replace with the actual file names)
files=(
  "include/common_def.h"
  "include/dsp_processing.h"
  "include/dsplib.h"
  "include/fixed_point_math.h"
  "include/jumpml_nr_tuning.h"
  "include/jumpml_nr.h"
  "include/nn_layers_tools.h"
  "include/nn_layers.h"
  "include/nnlib_fixedpt.h"
  "include/nnlib_float.h"
  "include/nnlib_types.h"
  "include/noise_reduction.h"
  "include/signalsifter_config.h"
  "include/signalsifter_monitor.h"
  "include/signalsifter.h"
  "include/tanh_table_S16.h"
  "include/tanh_table.h"
  "include/utils.h"
  "third-party/kissFFT/kiss_fftr.h"
  "third-party/kissFFT/kiss_fft.h"
  "src/jumpml_nr.c"
  "test/testnr/JumpML_NR_demo.c"
  "data/outdoor_mix.wav"
  "build/libjumpmlnr_arm64.a"
  "scripts/denoise.sh"
  "scripts/makefile_macos_sdk.mk"
  "docs/macos_sdk/readme.txt"
)

# Copy and move files based on their extensions
for file in "${files[@]}"; do
  if [[ "$file" == *.h ]]; then
    cp "$file" "$dest_dir/include/"
  elif [[ "$file" == *.c ]]; then
    cp "$file" "$dest_dir/src/"
  elif [[ "$file" == *.wav ]]; then
    cp "$file" "$dest_dir/data/"
  elif [[ "$file" == *.a ]]; then
    cp "$file" "$dest_dir/build/"  
  elif [[ "$file" == *.mk ]]; then
    cp "$file" "$dest_dir/Makefile" 
  else
    cp "$file" "$dest_dir/"
  fi
done

# Get the current Git commit hash (if in a Git repository)
commit_hash=$(git rev-parse HEAD 2>/dev/null)

# Get the current date in YYYY-MM-DD format
current_date=$(date +"%Y-%m-%d")

# Extract relevant details from the latest commit message
commit_message=$(git show -s --format=%B HEAD 2>/dev/null)
subject=$(echo "$commit_message" | head -n 1)
body=$(echo "$commit_message" | tail -n +2)

# Create version.txt file with commit hash, date, and extracted details
echo "Commit Hash: $commit_hash" > "$dest_dir/version.txt"
echo "Build Date: $current_date" >> "$dest_dir/version.txt"
echo "" >> "$dest_dir/version.txt" # Add an empty line for separation
echo "Commit Subject: $subject" >> "$dest_dir/version.txt"
echo "Commit Body:" >> "$dest_dir/version.txt"
echo "$body" >> "$dest_dir/version.txt"
