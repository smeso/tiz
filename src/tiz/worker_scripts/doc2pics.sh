#!/bin/bash

set -e

mkdir -p "${2}"
unoconv --format=pdf -o "${2}/output.pdf" "${1}"
pdftoppm -jpeg -forcenum -r 200 -- "${2}/output.pdf" "${2}/page"
rm "${2}/output.pdf"
echo "DONE"
