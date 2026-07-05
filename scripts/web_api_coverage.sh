#!/bin/bash

set -euo pipefail

cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/../"

rm -f .coverage
pytest --cov-fail-under=0 tests/test_web_api_integration.py
python3 -m coverage report --include="*/src/tiz/web_api.py" --show-missing --fail-under=89
