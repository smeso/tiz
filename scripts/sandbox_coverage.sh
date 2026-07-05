#!/bin/bash

set -euo pipefail

cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/../"

rm -f .coverage
pytest --cov-fail-under=0 tests/test_integration.py
python3 -m coverage report --include="*/src/tiz/sandbox_worker.py" --show-missing --fail-under=89
