#!/bin/bash
# Run the project checks inside a uv-managed virtualenv.
# For all supported Python versions

set -euo pipefail

projdir="$(dirname "$(realpath "${BASH_SOURCE[0]}")")/../"

"${projdir}scripts/uv_checks.sh" 3.11
"${projdir}scripts/uv_checks.sh" 3.12
"${projdir}scripts/uv_checks.sh" 3.13
"${projdir}scripts/uv_checks.sh" 3.14
