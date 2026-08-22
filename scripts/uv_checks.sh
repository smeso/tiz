#!/bin/bash
# Run the project checks inside a uv-managed virtualenv.
#
# Creates a virtualenv with a specific Python version using uv, installs the
# project with all its dependencies (including the dev extras needed by
# scripts/checks.sh) into it and runs scripts/checks.sh with that virtualenv
# active. The exit code of the checks is propagated to the caller.

set -euo pipefail

projdir="$(dirname "$(realpath "${BASH_SOURCE[0]}")")/../"
python_version="${1:-3.14}"
venv="${projdir}.venv-${python_version}"
venv_python="${venv}/bin/python"

cd "${projdir}"

uv venv --clear --python "${python_version}" "${venv}"
uv pip install --python "${venv_python}" -e ".[dev]"

export VIRTUAL_ENV="${venv}"
export PATH="${venv}/bin:${PATH}"
unset PYTHONHOME

"${projdir}scripts/checks.sh"
