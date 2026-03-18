#!/usr/bin/env bash
set -euo pipefail

# One-command training entry with the paper default config.
python scripts/train.py --config configs/assr_default.yaml
