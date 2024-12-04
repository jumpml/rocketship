#!/bin/bash
set -e

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
TESTNR="$SCRIPT_DIR/../build/testnr"
CHECKAUDIO="$SCRIPT_DIR/../build/compare_audio"
RAW_OUTPUT_FNAME="/tmp/_temp_output.raw"

INPUT="$SCRIPT_DIR/../data/sample01-orig.raw"
REFOUT="$SCRIPT_DIR/../data/sample01-orig_out.raw"

"$TESTNR" -i "$INPUT" -o "$RAW_OUTPUT_FNAME"
echo "Checking NR output vs. golden reference output"
"$CHECKAUDIO" "$RAW_OUTPUT_FNAME" "$REFOUT" 

INPUT="$SCRIPT_DIR/../data/sample01-orig_8khz.raw"
REFOUT="$SCRIPT_DIR/../data/sample01-orig_8khz_out.raw"

"$TESTNR" -i "$INPUT" -o "$RAW_OUTPUT_FNAME" -r 8000
echo "Checking NR output vs. golden reference output"
"$CHECKAUDIO" "$RAW_OUTPUT_FNAME" "$REFOUT" 