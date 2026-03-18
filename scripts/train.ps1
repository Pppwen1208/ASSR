#!/usr/bin/env pwsh
$ErrorActionPreference = "Stop"

# One-command training entry with the paper default config.
python scripts/train.py --config configs/assr_default.yaml
