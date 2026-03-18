from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from assr.config import load_config
from assr.data.dataset import ASSRDataset, assr_collate
from assr.engine.evaluator import evaluate_model, evaluate_scale_stability
from assr.models.assr import ASSR

DEFAULT_SCALES = "1.5,2,3,4,5.5,6"


def _parse_scales(raw: str) -> list[float]:
    scales = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if len(scales) == 0:
        raise ValueError("scales must contain at least one float value")
    return scales


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument(
        "--scales",
        type=str,
        default=DEFAULT_SCALES,
        help=(
            "Scale sweep for SAS/SCE, e.g. 1.5,2,3,4,5.5,6. "
            "Default computes SCE by default."
        ),
    )
    parser.add_argument(
        "--skip-scale-metrics",
        action="store_true",
        help="Skip SAS/SCE computation and only report PSNR/SSIM/LPIPS/Edge-F1.",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default="",
        help="Optional path to save metrics as JSON.",
    )
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
    state = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)
    model.eval()

    ds = ASSRDataset(
        manifest_path=cfg.data.test_manifest,
        data_cfg=cfg.data,
        degradation_cfg=cfg.degradation,
        training=False,
    )
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=assr_collate,
    )
    metrics = evaluate_model(
        model=model,
        loader=loader,
        device=device,
        use_amp=cfg.amp,
        max_batches=args.max_batches if args.max_batches > 0 else None,
    )

    if not args.skip_scale_metrics:
        scales = _parse_scales(args.scales)
        sas_sce = evaluate_scale_stability(
            model=model,
            loader=loader,
            scales=scales,
            device=device,
            use_amp=cfg.amp,
            max_batches=args.max_batches if args.max_batches > 0 else None,
        )
        metrics.update(sas_sce)

    # Public-facing alias for checklist wording.
    if "sce_l1" in metrics:
        metrics["sce"] = float(metrics["sce_l1"])

    order = [
        "psnr",
        "ssim",
        "lpips",
        "edge_f1",
        "sce",
        "sce_l1",
        "sce_lpips",
        "sas_ssim",
        "sas_edge_f1",
    ]
    printed: set[str] = set()
    for k in order:
        if k not in metrics:
            continue
        printed.add(k)
        v = float(metrics[k])
        print(f"{k}: {v:.6f}")

    for k in sorted(metrics.keys()):
        if k in printed:
            continue
        v = float(metrics[k])
        print(f"{k}: {v:.6f}")

    if args.save_json.strip():
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({k: float(v) for k, v in metrics.items()}, indent=2),
            encoding="utf-8",
        )
        print(f"saved_json: {out_path}")


if __name__ == "__main__":
    main()
