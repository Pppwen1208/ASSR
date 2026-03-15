from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from assr.config.schema import DataConfig, DegradationConfig
from assr.data.degradation import ScaleAwareDegradation
from assr.data.text_tokens import TextEmbeddingAdapter


def _load_manifest(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"manifest not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("manifest must be a JSON list")
    return data


def _pil_to_tensor(img: Image.Image, channels: int) -> torch.Tensor:
    if channels == 1:
        img = img.convert("L")
        arr = np.array(img, dtype=np.float32)[None, ...] / 255.0
    elif channels == 3:
        img = img.convert("RGB")
        arr = np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
    else:
        raise ValueError(f"unsupported channel count: {channels}")
    return torch.from_numpy(arr)


def _load_mask_tensor(path: str | Path) -> torch.Tensor:
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.float32)[None, ...] / 255.0
    return torch.from_numpy(arr)


def _resize_tensor(
    x: torch.Tensor | None,
    out_h: int,
    out_w: int,
    mode: str,
) -> torch.Tensor | None:
    if x is None:
        return None
    if x.shape[-2:] == (out_h, out_w):
        return x
    x4 = x.unsqueeze(0)
    if mode in ("bicubic", "bilinear"):
        y4 = F.interpolate(x4, size=(out_h, out_w), mode=mode, align_corners=False)
    else:
        y4 = F.interpolate(x4, size=(out_h, out_w), mode=mode)
    return y4.squeeze(0)


def _crop(x: torch.Tensor | None, top: int, left: int, crop: int) -> torch.Tensor | None:
    if x is None:
        return None
    return x[:, top : top + crop, left : left + crop]


def _augment(
    x: torch.Tensor | None,
    flip_x: bool,
    flip_y: bool,
    k_rot: int,
) -> torch.Tensor | None:
    if x is None:
        return None
    if flip_x:
        x = torch.flip(x, dims=[1])
    if flip_y:
        x = torch.flip(x, dims=[2])
    if k_rot > 0:
        x = torch.rot90(x, k_rot, dims=[1, 2])
    return x


class ASSRDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        data_cfg: DataConfig,
        degradation_cfg: DegradationConfig,
        training: bool = True,
    ) -> None:
        self.samples = _load_manifest(manifest_path)
        self.data_cfg = data_cfg
        self.training = training
        self.degradation = ScaleAwareDegradation(degradation_cfg)
        self.text_adapter = TextEmbeddingAdapter(
            max_len=data_cfg.max_text_len, embed_dim=768
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_text(self, sample: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        if "text_embed_path" in sample:
            return self.text_adapter.from_path(sample["text_embed_path"])
        if "text_embed" in sample:
            arr = np.asarray(sample["text_embed"], dtype=np.float32)
            return self.text_adapter.from_array(arr)
        if "text_tokens" in sample:
            return self.text_adapter.from_ids(sample["text_tokens"])
        return self.text_adapter.from_ids([])

    def _load_valid_mask(self, sample: dict[str, Any], h: int, w: int) -> torch.Tensor:
        valid = torch.ones(1, h, w, dtype=torch.float32)

        combined_path = sample.get("valid_mask", "") or sample.get("mask", "")
        cloud_path = sample.get("cloud_mask", "") or sample.get("cloud_mask_path", "")
        coh_path = sample.get("coherence_mask", "") or sample.get("coherence_mask_path", "")

        if isinstance(combined_path, str) and combined_path:
            valid = (_load_mask_tensor(combined_path) > 0.5).float()

        if isinstance(cloud_path, str) and cloud_path:
            cloud = (_load_mask_tensor(cloud_path) > 0.5).float()
            valid = valid * (1.0 - cloud)

        if isinstance(coh_path, str) and coh_path:
            coh = (_load_mask_tensor(coh_path) > 0.5).float()
            valid = valid * coh

        valid = _resize_tensor(valid, out_h=h, out_w=w, mode="nearest")
        return (valid > 0.5).float()

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        s2 = _pil_to_tensor(
            Image.open(sample["s2_hr"]),
            channels=self.data_cfg.s2_channels,
        )
        h, w = s2.shape[-2:]

        s1 = None
        s1_path = sample.get("s1_hr", "") or sample.get("s1_lr", "")
        if self.data_cfg.use_s1 and isinstance(s1_path, str) and s1_path:
            s1 = _pil_to_tensor(Image.open(s1_path), channels=self.data_cfg.s1_channels)
            s1 = _resize_tensor(s1, out_h=h, out_w=w, mode="bicubic")

        valid_mask = self._load_valid_mask(sample, h=h, w=w)

        crop = int(self.data_cfg.hr_crop_size)
        if h < crop or w < crop:
            nh = max(h, crop)
            nw = max(w, crop)
            s2 = _resize_tensor(s2, out_h=nh, out_w=nw, mode="bicubic")
            s1 = _resize_tensor(s1, out_h=nh, out_w=nw, mode="bicubic")
            valid_mask = _resize_tensor(valid_mask, out_h=nh, out_w=nw, mode="nearest")
            h, w = nh, nw

        top = random.randint(0, h - crop)
        left = random.randint(0, w - crop)
        s2_hr = _crop(s2, top=top, left=left, crop=crop)
        s1_hr = _crop(s1, top=top, left=left, crop=crop)
        valid_mask_hr = _crop(valid_mask, top=top, left=left, crop=crop)

        if self.training:
            flip_x = random.random() < 0.5
            flip_y = random.random() < 0.5
            k_rot = random.randint(0, 3)
            s2_hr = _augment(s2_hr, flip_x=flip_x, flip_y=flip_y, k_rot=k_rot)
            s1_hr = _augment(s1_hr, flip_x=flip_x, flip_y=flip_y, k_rot=k_rot)
            valid_mask_hr = _augment(
                valid_mask_hr,
                flip_x=flip_x,
                flip_y=flip_y,
                k_rot=k_rot,
            )

        force_x4_prob = float(
            min(max(self.data_cfg.pixelshuffle_supervision_prob, 0.0), 1.0)
        )
        if (
            random.random() < force_x4_prob
            and self.data_cfg.scale_min <= 4.0 <= self.data_cfg.scale_max
        ):
            scale = 4.0
        else:
            scale = random.uniform(self.data_cfg.scale_min, self.data_cfg.scale_max)
        s2_deg = self.degradation.degrade_s2(s2_hr, scale)

        s1_lr = None
        if s1_hr is not None:
            s1_deg = self.degradation.degrade_s1(s1_hr, scale)
            s1_lr = s1_deg.lr

        valid_mask_lr = F.interpolate(
            valid_mask_hr.unsqueeze(0),
            size=s2_deg.lr.shape[-2:],
            mode="nearest",
        ).squeeze(0)
        text_embed, text_mask = self._load_text(sample)

        return {
            "id": sample.get("id", str(index)),
            "s2_lr": s2_deg.lr,
            "s2_lr_ref": s2_deg.lr_ref,
            "s1_lr": s1_lr,
            "s2_hr": s2_hr,
            "valid_mask": valid_mask_hr,
            "valid_mask_lr": valid_mask_lr,
            "scale": torch.tensor(scale, dtype=torch.float32),
            "resize_meta": {"kappa": s2_deg.kappa, "eta": s2_deg.eta},
            "text_embed": text_embed,
            "text_mask": text_mask,
        }


def assr_collate(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Variable-scale samples have variable LR size; keep list form for micro-batch execution.
    return batch
