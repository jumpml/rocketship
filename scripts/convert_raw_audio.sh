#!/bin/bash

# Function to display help
show_help() {
  echo "Usage: $0 <input_file> <input_sample_rate> <input_format> <output_file> [output_sample_rate] [output_format]"
  echo ""
  echo "Converts RAW audio files between different formats and sample rates."
  echo ""
  echo "Arguments:"
  echo "  <input_file>          Path to the input RAW audio file."
  echo "  <input_sample_rate>   Sample rate of the input audio (e.g., 44100)."
  echo "  <input_format>        Format of the input audio. Supported formats are:"
  echo "                        - float32          : 32-bit floating-point"
  echo "                        - signed-int16     : 16-bit signed integer PCM"
  echo "                        - unsigned-int8    : 8-bit unsigned integer PCM"
  echo "                        - signed-int8      : 8-bit signed integer PCM"
  echo "                        - int32            : 32-bit signed integer PCM"
  echo "                        - ulaw             : µ-law companding"
  echo "                        - alaw             : A-law companding"
  echo "  <output_file>         Path to the output RAW audio file."
  echo "  [output_sample_rate]  Sample rate of the output audio (optional, defaults to input sample rate)."
  echo "  [output_format]       Format of the output audio (optional, defaults to input format)."
  echo ""
  echo "Example Usage:"
  echo "  $0 input.raw 44100 signed-int16 output.raw 48000 float32"
  echo "  $0 input.raw 8000 ulaw output.raw 8000 alaw"
  echo "  $0 input.raw 22050 signed-int8 output.raw"
  echo ""
  exit 0
}

# Check if the user asked for help
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  show_help
fi

# Check if minimum arguments are provided
if [ $# -lt 4 ]; then
  show_help
fi

# Assign input arguments to variables
INPUT_FILE="$1"
INPUT_SAMPLE_RATE="$2"
INPUT_FORMAT="$3"
OUTPUT_FILE="$4"
OUTPUT_SAMPLE_RATE="${5:-$INPUT_SAMPLE_RATE}"  # Default to input sample rate
OUTPUT_FORMAT="${6:-$INPUT_FORMAT}"            # Default to input format

# Function to validate the format
validate_format() {
  case "$1" in
    "float32"|"signed-int16"|"unsigned-int8"|"signed-int8"|"int32"|"ulaw"|"alaw")
      return 0
      ;;
    *)
      echo "Invalid format: $1. Supported formats are: float32, signed-int16, unsigned-int8, signed-int8, int32, ulaw, alaw."
      exit 1
      ;;
  esac
}

# Validate input and output formats
validate_format "$INPUT_FORMAT"
validate_format "$OUTPUT_FORMAT"

# Map the user-friendly format names to SoX encoding options
case "$INPUT_FORMAT" in
  "float32") SOX_INPUT_FORMAT="floating-point" ;;
  "signed-int16") SOX_INPUT_FORMAT="signed-integer" ;;
  "unsigned-int8") SOX_INPUT_FORMAT="unsigned-integer" ;;
  "signed-int8") SOX_INPUT_FORMAT="signed-integer" ;;
  "int32") SOX_INPUT_FORMAT="signed-integer" ;;
  "ulaw") SOX_INPUT_FORMAT="u-law" ;;
  "alaw") SOX_INPUT_FORMAT="a-law" ;;
esac

case "$OUTPUT_FORMAT" in
  "float32") SOX_OUTPUT_FORMAT="floating-point" ;;
  "signed-int16") SOX_OUTPUT_FORMAT="signed-integer" ;;
  "unsigned-int8") SOX_OUTPUT_FORMAT="unsigned-integer" ;;
  "signed-int8") SOX_OUTPUT_FORMAT="signed-integer" ;;
  "int32") SOX_OUTPUT_FORMAT="signed-integer" ;;
  "ulaw") SOX_OUTPUT_FORMAT="u-law" ;;
  "alaw") SOX_OUTPUT_FORMAT="a-law" ;;
esac

# Set the number of bits per sample for specific formats
case "$INPUT_FORMAT" in
  "signed-int16") INPUT_BITS="16" ;;
  "unsigned-int8"|"signed-int8") INPUT_BITS="8" ;;
  "int32") INPUT_BITS="32" ;;
  "float32") INPUT_BITS="32" ;;
  "ulaw"|"alaw") INPUT_BITS="8" ;;
esac

case "$OUTPUT_FORMAT" in
  "signed-int16") OUTPUT_BITS="16" ;;
  "unsigned-int8"|"signed-int8") OUTPUT_BITS="8" ;;
  "int32") OUTPUT_BITS="32" ;;
  "float32") OUTPUT_BITS="32" ;;
  "ulaw"|"alaw") OUTPUT_BITS="8" ;;
esac

# Convert the RAW audio file using SoX
sox -r "$INPUT_SAMPLE_RATE" -e "$SOX_INPUT_FORMAT" -b "$INPUT_BITS" -t raw "$INPUT_FILE" \
    -r "$OUTPUT_SAMPLE_RATE" -e "$SOX_OUTPUT_FORMAT" -b "$OUTPUT_BITS" -t raw "$OUTPUT_FILE"

# Check if the conversion was successful
if [ $? -eq 0 ]; then
  echo "Conversion successful!"
else
  echo "Conversion failed!"
  exit 1
fi
