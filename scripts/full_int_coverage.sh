#!/bin/bash

set -euo pipefail

cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/../"

rm -f .coverage
pytest --cov-fail-under=0 tests/test_full_integration.py
