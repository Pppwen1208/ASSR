from __future__ import annotations

from pathlib import Path

import yaml

from assr.config.schema import ASSRConfig


def load_config(path: str | Path) -> ASSRConfig:
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return ASSRConfig.from_dict(data)

