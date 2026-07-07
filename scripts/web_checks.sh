#!/bin/bash
#
# web_checks.sh - Run frontend linting/formatting checks on web_static files.
#
# Runs eslint, stylelint, htmlhint, and prettier (in check mode) against
# files in src/tiz/data/web_static/.

set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
REPO_ROOT="$(realpath "$SCRIPT_DIR/..")"

export PATH="$REPO_ROOT/node_modules/.bin:$PATH"

cd "$REPO_ROOT/src/tiz/data/web_static"

WEB_DIR="$PWD"

eslint --max-warnings=0 "$WEB_DIR"/**/*.js

stylelint "$WEB_DIR"/**/*.css

htmlhint "$WEB_DIR"/**/*.html

prettier --check "$WEB_DIR"/**/*.js "$WEB_DIR"/**/*.css "$WEB_DIR"/**/*.html

echo "OK"
