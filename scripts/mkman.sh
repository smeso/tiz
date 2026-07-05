#!/bin/bash

export PYTHONPATH
PYTHONPATH="$(dirname "$(realpath "${BASH_SOURCE[0]}")")/../src/"
exec argparse-manpage --module tiz.cli --function get_parser --prog tiz  --output "${1}"
