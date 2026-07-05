#!/bin/bash

set -euo pipefail

cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/../"

pytest
pytest --accurate-cov
./scripts/sandbox_coverage.sh
./scripts/web_api_coverage.sh
./scripts/full_int_coverage.sh
ruff check
mypy src
ruff format --diff

echo OK
