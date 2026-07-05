#!/bin/bash

set -e

mkdir -p "${2}"
ffmpeg -v quiet -skip_frame nokey -i "${1}" -vf 'scale=iw*0.5:ih*0.5' -r 1/10 "${2}/f%08d.jpg"
echo "DONE"
