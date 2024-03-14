#!/bin/bash
set -e 

if [ $# -ne 4 ]; then
	echo "Incorrect number of arguments. Usage: denoise.sh input.wav output.wav naturalness mingain"
	exit
fi


# To use different values, PLEASE MODIFY THESE. BUT THESE ARE GOOD DEFAULTS.
naturalness=$3
mingain=$4

# NATURALNESS in [0,1]. 0 means maximum noise suppression, 1 means more natural  (more speech preservation) 
# MINGAIN in [-60, 0] dB. 0 dB means no suppression.


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

SR1=`sox --i -r "$1"`
JUMPML_NR_SR=16000
RAW_INPUT_FNAME="/tmp/_temp_input.raw"
RAW_OUTPUT_FNAME="/tmp/_temp_output.raw"

TESTNR=build/testnr
`sox "$1" -c 1 --bits 16 --encoding signed-integer --endian little --rate $JUMPML_NR_SR $RAW_INPUT_FNAME`
echo `$TESTNR -n $naturalness -m $mingain -i "$RAW_INPUT_FNAME" -o "$RAW_OUTPUT_FNAME"`
`sox -r $JUMPML_NR_SR -b 16 -e signed-integer $RAW_OUTPUT_FNAME -r $SR1 -b 16 -e signed-integer "$2"`


# if [[ "$OSTYPE" == "darwin"* ]]; then
# 	afplay $2
# else
# 	aplay $2
# fi
