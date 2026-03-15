from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from assr.config import load_config
from assr.data.text_tokens import TextEmbeddingAdapter
from assr.inference import tiled_infer
from assr.models.assr import ASSR


def _load_img(path: str, channels: int) -> torch.Tensor:
    img = Image.open(path)
    if channels == 1:
        arr = np.array(img.convert("L"), dtype=np.float32)[None, ...] / 255.0
    else:
        arr = np.array(img.convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _save_img(t: torch.Tensor, path: str) -> None:
    x = t.detach().cpu().clamp(0.0, 1.0)[0]
    if x.shape[0] == 1:
        arr = (x[0].numpy() * 255.0).astype(np.uint8)
        img = Image.fromarray(arr, mode="L")
    else:
        arr = (x.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        img = Image.fromarray(arr, mode="RGB")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def _save_risk_map(t: torch.Tensor, path: str) -> None:
    r = t.detach().cpu().clamp(0.0, 1.0)[0, 0]
    arr = (r.numpy() * 255.0).astype(np.uint8)
    img = Image.fromarray(arr, mode="L")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def _parse_text(
    text: str | None,
    adapter: TextEmbeddingAdapter,
) -> tuple[torch.Tensor, torch.Tensor]:
    if text is None or text.strip() == "":
        return adapter.from_ids([])
    p = Path(text)
    if p.exists():
        if p.suffix.lower() == ".json":
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], int):
                return adapter.from_ids(data)
            if isinstance(data, list):
                return adapter.from_array(np.asarray(data, dtype=np.float32))
        if p.suffix.lower() == ".npy":
            return adapter.from_path(p)
    # fallback: comma-separated ids
    ids = [int(x.strip()) for x in text.split(",") if x.strip()]
    return adapter.from_ids(ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True, help="S2 LR image path")
    parser.add_argument("--s1", type=str, default="", help="Optional S1 aligned LR image")
    parser.add_argument("--text", type=str, default="", help="Token json/npy path or comma ids")
    parser.add_argument("--scale", type=float, default=4.0)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--risk-output", type=str, default="", help="Optional risk map output path")
    parser.add_argument("--enable-risk-gate", action="store_true", help="Enable deployment-time risk gate")
    parser.add_argument("--risk-gate-strength", type=float, default=0.35)
    args = parser.parse_args()

    cfg = load_config(args.config)

    device = torch.device(
        cfg.device if cfg.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    model = ASSR(
        cfg=cfg.model,
        s2_channels=cfg.data.s2_channels,
        s1_channels=cfg.data.s1_channels,
        use_s1=cfg.data.use_s1,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()

    s2 = _load_img(args.input, channels=cfg.data.s2_channels).to(device)
    s1 = None
    if args.s1:
        s1 = _load_img(args.s1, channels=cfg.data.s1_channels).to(device)

    adapter = TextEmbeddingAdapter(max_len=cfg.data.max_text_len, embed_dim=768)
    te, tm = _parse_text(args.text, adapter)
    te = te.unsqueeze(0).to(device)
    tm = tm.unsqueeze(0).to(device)

    with torch.no_grad():
        out = tiled_infer(
            model=model,
            s2_lr=s2,
            s1_lr=s1,
            scale=args.scale,
            tile_size=cfg.infer.tile_size,
            hr_overlap=cfg.infer.hr_overlap,
            text_embed=te,
            text_mask=tm,
            enable_risk_gate=bool(args.enable_risk_gate),
            risk_gate_strength=float(args.risk_gate_strength),
            return_aux=True,
        )
    _save_img(out["sr"], args.output)
    if args.risk_output:
        _save_risk_map(out["risk_map"], args.risk_output)
    sem = float(out["semantic_consistency"].mean().item())
    print(f"semantic_consistency: {sem:.6f}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
