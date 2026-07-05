#!/bin/bash

set -e

ffmpeg -v quiet -i "${1}" -ar 16000 -ac 1 -c:a pcm_s16le "${2}"
echo "DONE"
