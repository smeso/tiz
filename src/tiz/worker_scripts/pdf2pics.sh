#!/bin/bash

set -e

mkdir -p "${2}"
pdftoppm -jpeg -forcenum -r 200 -- "${1}" "${2}/page"
echo "DONE"
