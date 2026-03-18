from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image

CAPTION_MODEL_ID = "Salesforce/blip-image-captioning-base"
CAPTION_PROMPT = "Describe this image in one short sentence."
CAPTION_DO_SAMPLE = False
CAPTION_NUM_BEAMS = 3
CAPTION_MAX_NEW_TOKENS = 16
CAPTION_MAX_TOKENS = 16
CAPTION_IMAGE_SIZE = 384
CAPTION_FALLBACK = "a low-resolution satellite image"


def _load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str | Path, data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _normalize_caption(text: str) -> str:
    # Table A2: lowercase + whitespace stripping.
    return " ".join((text or "").strip().lower().split())


def _extract_text_from_pipeline_output(obj: Any) -> str:
    # HF image-to-text pipeline may return:
    # [{'generated_text': '...'}] OR [[{'generated_text': '...'}], ...]
    if isinstance(obj, list):
        if len(obj) == 0:
            return ""
        if isinstance(obj[0], dict):
            return str(obj[0].get("generated_text", ""))
        return _extract_text_from_pipeline_output(obj[0])
    if isinstance(obj, dict):
        return str(obj.get("generated_text", ""))
    return str(obj)


def _resize_for_caption(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    if hasattr(Image, "Resampling"):
        return image.resize((CAPTION_IMAGE_SIZE, CAPTION_IMAGE_SIZE), Image.Resampling.BICUBIC)
    return image.resize((CAPTION_IMAGE_SIZE, CAPTION_IMAGE_SIZE), Image.BICUBIC)


def _run_caption_inference(caption_pipe: Any, images: list[Image.Image]) -> list[str]:
    kwargs = {
        "max_new_tokens": CAPTION_MAX_NEW_TOKENS,
        "do_sample": CAPTION_DO_SAMPLE,
        "num_beams": CAPTION_NUM_BEAMS,
    }
    try:
        pred = caption_pipe(images, prompt=CAPTION_PROMPT, **kwargs)
    except TypeError:
        pred = caption_pipe(images, **kwargs)

    if len(images) == 1:
        pred = [pred]

    captions: list[str] = []
    for one_out in pred:
        caption = _normalize_caption(_extract_text_from_pipeline_output(one_out))
        if not caption:
            caption = CAPTION_FALLBACK
        captions.append(caption)
    return captions


def _resolve_image_path(
    entry: dict[str, Any],
    image_key: str,
    fallback_keys: list[str],
) -> str:
    keys = [image_key] + [k for k in fallback_keys if k != image_key]
    for k in keys:
        v = entry.get(k, "")
        if isinstance(v, str) and v.strip():
            return v
    raise KeyError(
        f"missing image path in entry id={entry.get('id', '<unknown>')}; "
        f"checked keys={keys}"
    )


def _load_caption_model(
    local_files_only: bool,
    device: str,
) -> tuple[Any, Any]:
    try:
        from transformers import AutoTokenizer, pipeline
    except Exception as e:
        raise RuntimeError(
            "transformers is required. Please install requirements.txt first."
        ) from e

    if device == "auto":
        use_cuda = torch.cuda.is_available()
    else:
        use_cuda = device.startswith("cuda")
    pipe_device = 0 if use_cuda else -1
    dtype = torch.float16 if use_cuda else torch.float32

    cap_pipe = pipeline(
        task="image-to-text",
        model=CAPTION_MODEL_ID,
        device=pipe_device,
        torch_dtype=dtype,
        local_files_only=local_files_only,
    )
    tok = AutoTokenizer.from_pretrained(
        CAPTION_MODEL_ID,
        use_fast=True,
        local_files_only=local_files_only,
    )
    return cap_pipe, tok


def _truncate_to_token_ids(tokenizer: Any, text: str, max_tokens: int = CAPTION_MAX_TOKENS) -> list[int]:
    return tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_tokens,
    )


def _batch_chunks(items: list[Any], size: int) -> list[list[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _run_manifest_mode(
    manifest: list[dict[str, Any]],
    caption_pipe: Any,
    tokenizer: Any,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    outputs = [dict(x) for x in manifest]
    index_and_paths: list[tuple[int, str]] = []
    for i, entry in enumerate(outputs):
        img_path = _resolve_image_path(
            entry,
            image_key=args.image_key,
            fallback_keys=args.fallback_keys,
        )
        index_and_paths.append((i, img_path))

    for chunk in _batch_chunks(index_and_paths, args.batch_size):
        imgs = [_resize_for_caption(Image.open(p)) for _, p in chunk]
        captions = _run_caption_inference(caption_pipe, imgs)
        for (idx, _), caption in zip(chunk, captions):
            token_ids = _truncate_to_token_ids(tokenizer, caption)
            outputs[idx][args.caption_key] = caption
            outputs[idx][args.token_key] = token_ids
    return outputs


def _run_glob_mode(
    paths: list[str],
    caption_pipe: Any,
    tokenizer: Any,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for chunk in _batch_chunks(paths, args.batch_size):
        imgs = [_resize_for_caption(Image.open(p)) for p in chunk]
        captions = _run_caption_inference(caption_pipe, imgs)
        for path, caption in zip(chunk, captions):
            token_ids = _truncate_to_token_ids(tokenizer, caption)
            rows.append(
                {
                    "image": path,
                    args.caption_key: caption,
                    args.token_key: token_ids,
                }
            )
    return rows


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Offline LR caption generation for ASSR. "
            "Captions are truncated to max token length and exported as JSON."
        )
    )
    p.add_argument("--manifest", type=str, default="", help="Input manifest JSON list")
    p.add_argument("--images-glob", type=str, default="", help="Image glob pattern")
    p.add_argument("--output", type=str, required=True, help="Output JSON path")

    p.add_argument(
        "--model-id",
        type=str,
        default=CAPTION_MODEL_ID,
        help=f"Ignored. Fixed by protocol to {CAPTION_MODEL_ID}.",
    )
    p.add_argument(
        "--tokenizer-id",
        type=str,
        default="",
        help="Ignored. Tokenizer is fixed to model-id by protocol.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=CAPTION_MAX_NEW_TOKENS,
        help=f"Ignored. Fixed by protocol to {CAPTION_MAX_NEW_TOKENS}.",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=CAPTION_MAX_TOKENS,
        help=f"Ignored. Fixed by protocol to {CAPTION_MAX_TOKENS}.",
    )
    p.add_argument("--local-files-only", action="store_true")

    p.add_argument("--image-key", type=str, default="s2_lr")
    p.add_argument(
        "--fallback-keys",
        type=str,
        default="s2_hr,image",
        help="Comma-separated fallback keys for image path resolution",
    )
    p.add_argument("--caption-key", type=str, default="text_caption")
    p.add_argument("--token-key", type=str, default="text_tokens")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    args.fallback_keys = [k.strip() for k in args.fallback_keys.split(",") if k.strip()]

    if (not args.manifest) and (not args.images_glob):
        raise ValueError("either --manifest or --images-glob must be provided")

    caption_pipe, tokenizer = _load_caption_model(
        local_files_only=args.local_files_only,
        device=args.device,
    )

    if args.manifest:
        manifest_data = _load_json(args.manifest)
        if not isinstance(manifest_data, list):
            raise ValueError("manifest JSON must be a list of sample objects")
        out = _run_manifest_mode(
            manifest=manifest_data,
            caption_pipe=caption_pipe,
            tokenizer=tokenizer,
            args=args,
        )
    else:
        paths = sorted(glob.glob(args.images_glob))
        if len(paths) == 0:
            raise FileNotFoundError(f"no images matched: {args.images_glob}")
        out = _run_glob_mode(
            paths=paths,
            caption_pipe=caption_pipe,
            tokenizer=tokenizer,
            args=args,
        )

    _save_json(args.output, out)
    print(f"Saved captions: {args.output} ({len(out)} records)")


if __name__ == "__main__":
    main()
