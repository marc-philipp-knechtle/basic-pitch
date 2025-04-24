#!/usr/bin/env bash

if [ "$#" -lt 2 ]; then
  echo "This script needs 2 parameters."
  exit 1
fi

OUTPUT_DIR=$1
INPUT_DIR=$2

if [ ! -d "$INPUT_DIR" ]; then
  echo "Error: $INPUT_DIR is not a valid directory"
  exit 1
fi

FILES=$(find "$INPUT_DIR" -type f)

for FILE in $FILES; do
  basic-pitch "$OUTPUT_DIR" "$FILE" --save-midi --sonify-midi --save-model-outputs
done

# basic-pitch "$OUTPUT_DIR" "$FILES" --save-midi --sonify-midi
