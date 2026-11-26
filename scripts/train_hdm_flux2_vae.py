"""
Self-contained training script to adapt pretrained HDM (XUDiT-340M) to FLUX.2 VAE
(32ch -> 2x2 patchified 128ch + BatchNorm) latent space.

Features:
- Loads pretrained `KBlueLeaf/HDM-xut-340M-anime` backbone (XUDiT) + Qwen3-0.6B text encoder.
- Replaces only `in_patch` (4ch p=2 -> 128ch p=1) and `out_patch` (4ch p=2 -> 128ch p=1),
  preserving all ~343M pretrained transformer weights and identical token count (16x total downsample).
- Optional 2-stage warmup: trains only `in_patch`/`out_patch` for `--warmup_patch_steps`,
  then unfreezes the full backbone.
- Encodes/decodes in exact Diffusers FLUX.2 VAE latent space (`AutoencoderKLFlux2`:
  encode -> 2x2 patchify -> normalize with `vae.bn` running stats).
- Streams `nyanko-devs/danbooru2026` tar shards directly over HTTP (no full tar download),
  joining captions from `metadata/posts-snapshot.parquet` with Shifted Square Crop.
- Runs 1,000 optimizer steps and saves 2x2 validation grids using 4 `rating: g` prompts.

Usage:
    python scripts/train_hdm_flux2_vae.py \
        --resolution 512 \
        --batch_size 4 \
        --grad_accum 4 \
        --steps 1000 \
        --val_every 250 \
        --flux2_vae_repo black-forest-labs/FLUX.2-dev
"""

import argparse
import io
import math
import os
import queue
import random
import sys
import tarfile
import threading
import time
import urllib.request
from pathlib import Path

# Allow running standalone from any working directory
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from diffusers import AutoencoderKL, AutoencoderKLFlux2
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import AutoTokenizer, Qwen3Model

import xut.env
xut.env.TORCH_COMPILE = False  # disable per-op compile for fast startup & Windows safety
from xut.modules.axial_rope import make_axial_pos_no_cache, make_cropped_pos
from xut.xut import XUDiT

# ---------------------------------------------------------------------------
# 4 Rating-G Validation Prompts
# ---------------------------------------------------------------------------
VAL_PROMPTS = [
    (
        "1girl, solo, silver hair, blue eyes, white sundress, sunflower field, "
        "blue sky, clouds, wind, smile, looking at viewer, masterpiece, newest, general, safe"
    ),
    (
        "1girl, solo, witch hat, library, floating books, warm candlelight, "
        "cozy atmosphere, detailed background, masterpiece, newest, general, safe"
    ),
    (
        "1girl, solo, school uniform, standing on train station platform, sunset, "
        "golden hour, lens flare, anime scenery, masterpiece, newest, general, safe"
    ),
    (
        "1boy, 1girl, sitting at a cafe table, coffee cups, rain outside window, "
        "warm indoor lighting, slice of life, masterpiece, newest, general, safe"
    ),
]
VAL_NEG_PROMPT = "worst quality, low quality, bad anatomy, old, early, watermark, signature"


# ---------------------------------------------------------------------------
# FLUX.2 VAE Helpers (exact Diffusers Flux2Pipeline convention)
# ---------------------------------------------------------------------------
def patchify_flux2_latents(latents: torch.Tensor) -> torch.Tensor:
    """[B, 32, H/8, W/8] -> [B, 128, H/16, W/16]"""
    b, c, h, w = latents.shape
    latents = latents.view(b, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    return latents.reshape(b, c * 4, h // 2, w // 2)


def unpatchify_flux2_latents(latents: torch.Tensor) -> torch.Tensor:
    """[B, 128, H/16, W/16] -> [B, 32, H/8, W/8]"""
    b, c4, h, w = latents.shape
    c = c4 // 4
    latents = latents.view(b, c, 2, 2, h, w)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    return latents.reshape(b, c, h * 2, w * 2)


def get_eq_sdxl_stats(vae: AutoencoderKL, device, dtype):
    mean = torch.tensor(vae.config.latents_mean, device=device, dtype=dtype).view(1, -1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=device, dtype=dtype).view(1, -1, 1, 1)
    return mean, std


@torch.no_grad()
def encode_eq_sdxl(
    vae: AutoencoderKL,
    images: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    sample_mode: str = "sample",
) -> torch.Tensor:
    """images in [-1, 1], [B, 3, H, W] -> normalized [B, 4, H/8, W/8]."""
    dist = vae.encode(images).latent_dist
    latents = dist.sample() if sample_mode == "sample" else dist.mode()
    return (latents - mean) / std


@torch.no_grad()
def decode_eq_sdxl(
    vae: AutoencoderKL,
    latents: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> list[Image.Image]:
    """normalized [B, 4, H/8, W/8] -> list of PIL.Image."""
    latents = latents * std + mean
    imgs = torch.cat([vae.decode(z.unsqueeze(0)).sample for z in latents], dim=0)
    imgs = (imgs.float() / 2.0 + 0.5).clamp(0.0, 1.0)
    imgs = (imgs.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype("uint8")
    return [Image.fromarray(arr) for arr in imgs]


def get_flux2_bn_stats(vae: AutoencoderKLFlux2, device, dtype):
    bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(device=device, dtype=dtype)
    bn_std = torch.sqrt(
        vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
    ).to(device=device, dtype=dtype)
    return bn_mean, bn_std


@torch.no_grad()
def encode_flux2(
    vae: AutoencoderKLFlux2,
    images: torch.Tensor,
    bn_mean: torch.Tensor,
    bn_std: torch.Tensor,
    sample_mode: str = "sample",
) -> torch.Tensor:
    """images in [-1, 1], [B, 3, H, W] -> normalized [B, 128, H/16, W/16]."""
    dist = vae.encode(images).latent_dist
    latents = dist.sample() if sample_mode == "sample" else dist.mode()
    latents = patchify_flux2_latents(latents)
    latents = (latents - bn_mean) / bn_std
    return latents


@torch.no_grad()
def decode_flux2(
    vae: AutoencoderKLFlux2,
    latents: torch.Tensor,
    bn_mean: torch.Tensor,
    bn_std: torch.Tensor,
) -> list[Image.Image]:
    """normalized [B, 128, H/16, W/16] -> list of PIL.Image."""
    latents = latents * bn_std + bn_mean
    latents = unpatchify_flux2_latents(latents)
    imgs = torch.cat(
        [vae.decode(z.unsqueeze(0)).sample for z in latents], dim=0
    )
    imgs = (imgs.float() / 2.0 + 0.5).clamp(0.0, 1.0)
    imgs = (imgs.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype("uint8")
    return [Image.fromarray(arr) for arr in imgs]


# ---------------------------------------------------------------------------
# Danbooru2026 Streaming Data Source (HTTP tar + Parquet metadata join)
# ---------------------------------------------------------------------------
KAOMOJI_CHARS = set("^><=;:")


def clean_tag(tag: str) -> str:
    if any(ch in tag for ch in KAOMOJI_CHARS):
        return tag
    return tag.replace("_", " ")


def build_caption(row: dict, tag_shuffle: bool = True) -> str:
    parts = []
    for col in ("tag_string_character", "tag_string_copyright", "tag_string_artist"):
        val = row.get(col) or ""
        tags = [clean_tag(t) for t in val.split() if t]
        if tags:
            parts.extend(tags)

    gen = [clean_tag(t) for t in (row.get("tag_string_general") or "").split() if t]
    if tag_shuffle:
        random.shuffle(gen)
    parts.extend(gen)

    rating = row.get("rating", "")
    if rating == "g":
        parts.extend(["general", "safe"])
    elif rating == "s":
        parts.append("sensitive")

    score = row.get("score", 0) or 0
    if score >= 50:
        parts.append("masterpiece")
    elif score >= 15:
        parts.append("best quality")

    return ", ".join(parts)


def load_danbooru2026_metadata(
    meta_parquet: str | None = None,
    rating_filter: str | None = "g",
    max_rows: int | None = None,
) -> dict[int, dict]:
    """Download (once, cached by HF Hub) and index posts-snapshot.parquet by post ID."""
    if meta_parquet is None:
        print("Downloading/locating nyanko-devs/danbooru2026 metadata/posts-snapshot.parquet...")
        meta_parquet = hf_hub_download(
            repo_id="nyanko-devs/danbooru2026",
            filename="metadata/posts-snapshot.parquet",
            repo_type="dataset",
        )
    cols = [
        "id",
        "rating",
        "score",
        "tag_string_artist",
        "tag_string_character",
        "tag_string_copyright",
        "tag_string_general",
    ]
    print(f"Reading metadata columns from {meta_parquet}...")
    pf = pq.ParquetFile(meta_parquet)
    id_to_meta: dict[int, dict] = {}
    for rg in tqdm(range(pf.num_row_groups), desc="Indexing parquet"):
        tbl = pf.read_row_group(rg, columns=cols)
        ids = tbl["id"].to_pylist()
        ratings = tbl["rating"].to_pylist()
        scores = tbl["score"].to_pylist()
        artists = tbl["tag_string_artist"].to_pylist()
        chars = tbl["tag_string_character"].to_pylist()
        copys = tbl["tag_string_copyright"].to_pylist()
        gens = tbl["tag_string_general"].to_pylist()
        for pid, rat, sc, art, ch, cp, gn in zip(
            ids, ratings, scores, artists, chars, copys, gens
        ):
            if rating_filter and rat != rating_filter:
                continue
            id_to_meta[pid] = {
                "rating": rat,
                "score": sc,
                "tag_string_artist": art,
                "tag_string_character": ch,
                "tag_string_copyright": cp,
                "tag_string_general": gn,
            }
            if max_rows and len(id_to_meta) >= max_rows:
                print(f"Indexed {len(id_to_meta):,} posts (capped at max_rows).")
                return id_to_meta
    print(f"Indexed {len(id_to_meta):,} posts (rating_filter={rating_filter}).")
    return id_to_meta


def shifted_square_crop(img: Image.Image, size: int, latent_stride: int = 16):
    """HDM Shifted Square Crop -> (img_tensor [-1,1], pos_map [H/16*W/16, 2], log_ar)."""
    img = img.convert("RGB")
    resize_tf = T.Compose(
        [
            T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    img_t = resize_tf(img)
    _, target_h, target_w = img_t.shape
    # Ensure both dims are divisible by latent_stride (16)
    target_h = (target_h // latent_stride) * latent_stride
    target_w = (target_w // latent_stride) * latent_stride
    img_t = img_t[:, :target_h, :target_w]

    log_ar = math.log(target_w / target_h)
    crop_h, crop_w = 0, 0
    if target_h > target_w:
        max_steps = (target_h - target_w) // latent_stride
        crop_h = random.randint(0, max_steps) * latent_stride if max_steps > 0 else 0
        img_crop = img_t[:, crop_h : crop_h + target_w, :]
    elif target_w > target_h:
        max_steps = (target_w - target_h) // latent_stride
        crop_w = random.randint(0, max_steps) * latent_stride if max_steps > 0 else 0
        img_crop = img_t[:, :, crop_w : crop_w + target_h]
    else:
        img_crop = img_t

    pos_map = make_cropped_pos(
        crop_h // latent_stride,
        crop_w // latent_stride,
        target_h // latent_stride,
        target_w // latent_stride,
    )
    return img_crop, pos_map, log_ar


def stream_danbooru2026_batches(
    id_to_meta: dict[int, dict],
    resolution: int,
    batch_size: int,
    shards: list[int],
    hf_token: str | None = None,
    latent_stride: int = 16,
):
    """Infinite generator yielding (images, captions, pos_maps, log_ars) by streaming HF tars."""
    headers = {"User-Agent": "hdm-flux2-adapt/1.0"}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    buf_imgs, buf_caps, buf_pos, buf_ars = [], [], [], []
    shard_order = list(shards)

    while True:
        random.shuffle(shard_order)
        for shard_id in shard_order:
            url = (
                f"https://huggingface.co/datasets/nyanko-devs/danbooru2026/"
                f"resolve/main/original/data-{shard_id:04d}.tar"
            )
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=30) as resp:
                    with tarfile.open(fileobj=resp, mode="r|") as tf:
                        for member in tf:
                            if not member.isfile() or not member.name.endswith(".webp"):
                                continue
                            try:
                                pid = int(member.name.rsplit("/", 1)[-1].split(".")[0])
                            except ValueError:
                                continue
                            meta = id_to_meta.get(pid)
                            if meta is None:
                                continue
                            raw = tf.extractfile(member).read()
                            try:
                                img = Image.open(io.BytesIO(raw))
                                img_t, pos_map, log_ar = shifted_square_crop(
                                    img, resolution, latent_stride=latent_stride
                                )
                            except Exception:
                                continue

                            caption = build_caption(meta, tag_shuffle=True)
                            buf_imgs.append(img_t)
                            buf_caps.append(caption)
                            buf_pos.append(pos_map)
                            buf_ars.append(log_ar)

                            if len(buf_imgs) == batch_size:
                                yield (
                                    torch.stack(buf_imgs),
                                    list(buf_caps),
                                    torch.stack(buf_pos),
                                    torch.tensor(buf_ars, dtype=torch.float32),
                                )
                                buf_imgs.clear()
                                buf_caps.clear()
                                buf_pos.clear()
                                buf_ars.clear()
            except Exception as e:
                print(f"[warn] Stream error on shard {shard_id:04d}: {e}. Moving to next shard.")
                continue


def prefetched_danbooru2026_batches(
    id_to_meta: dict[int, dict],
    resolution: int,
    batch_size: int,
    shards: list[int],
    hf_token: str | None = None,
    latent_stride: int = 16,
    num_workers: int = 4,
    queue_size: int = 8,
):
    """Multi-threaded prefetch wrapper over stream_danbooru2026_batches so GPU never stalls on HTTP/PIL."""
    q: queue.Queue = queue.Queue(maxsize=queue_size)
    num_workers = max(1, min(num_workers, len(shards)))

    def worker(worker_shards):
        gen = stream_danbooru2026_batches(
            id_to_meta=id_to_meta,
            resolution=resolution,
            batch_size=batch_size,
            shards=worker_shards,
            hf_token=hf_token,
            latent_stride=latent_stride,
        )
        for batch in gen:
            q.put(batch)

    for w in range(num_workers):
        w_shards = shards[w::num_workers]
        t = threading.Thread(target=worker, args=(w_shards,), daemon=True)
        t.start()

    while True:
        yield q.get()


# ---------------------------------------------------------------------------
# Model Loading & Adaptation (HDM 4ch p=2 -> FLUX.2 128ch p=1)
# ---------------------------------------------------------------------------
def load_adapted_hdm(
    hdm_repo: str,
    device: torch.device,
    vae_mode: str = "flux2",
    grad_ckpt: bool = True,
) -> XUDiT:
    import json

    cfg_path = hf_hub_download(hdm_repo, "transformer/config.json")
    ckpt_path = hf_hub_download(hdm_repo, "transformer/diffusion_pytorch_model.safetensors")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Remove diffusers metadata keys
    for k in list(cfg.keys()):
        if k.startswith("_"):
            cfg.pop(k)

    if vae_mode == "flux2":
        # Adapt input/output interface to FLUX.2 patchified latent space: [B, 128, H/16, W/16]
        cfg["input_dim"] = 128
        cfg["patch_size"] = 1

    model = XUDiT(**cfg)

    raw_sd = load_file(ckpt_path, device="cpu")
    state_dict = {}
    dropped = []
    for k, v in raw_sd.items():
        clean_k = k[len("model."):] if k.startswith("model.") else k
        if vae_mode == "flux2" and (clean_k.startswith("in_patch.") or clean_k.startswith("out_patch.")):
            dropped.append(clean_k)
            continue
        state_dict[clean_k] = v

    missing, unexpected = model.load_state_dict(state_dict, strict=(vae_mode == "eq_sdxl"))
    print(f"Loaded pretrained HDM weights from {hdm_repo} (vae_mode={vae_mode}).")
    if dropped:
        print(f"  Dropped old patch layers : {dropped}")
    if missing:
        print(f"  Re-initialized keys      : {missing}")
    if unexpected:
        print(f"  Unexpected keys          : {unexpected}")

    if grad_ckpt:
        model.set_grad_ckpt(True)

    return model.to(device)


def build_mup_optimizer(
    model: nn.Module,
    base_lr: float,
    patch_lr_mult: float = 1.0,
    weight_decay: float = 0.01,
) -> torch.optim.Optimizer:
    """HDM muP per-parameter LR scaling (lr / fan_in), matching BaseTrainer."""
    param_groups = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2:
            fan_in = param.numel()
        elif param.ndim > 2:
            fan_in = 1
            for s in param.shape[1:]:
                fan_in *= s
        else:
            fan_in = param.shape[1]

        mult = patch_lr_mult if ("in_patch" in name or "out_patch" in name) else 1.0
        param_groups.append(
            {
                "params": [param],
                "lr": (base_lr * mult) / fan_in,
                "weight_decay": 0.0 if param.ndim < 2 else weight_decay,
            }
        )
    return torch.optim.AdamW(param_groups, lr=base_lr, betas=(0.9, 0.98))


# ---------------------------------------------------------------------------
# Validation Sampler (Euler Flow Matching + CFG + TREAD Auto-Guidance)
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_validation(
    model: XUDiT,
    text_encoder: Qwen3Model,
    tokenizer,
    vae,
    bn_mean: torch.Tensor,
    bn_std: torch.Tensor,
    out_dir: Path,
    step: int,
    resolution: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    vae_mode: str = "flux2",
    num_steps: int = 24,
    cfg_scale: float = 3.5,
    tread_gamma2: float = 0.25,
    seed: int = 42,
):
    model.eval()
    prompts = VAL_PROMPTS
    neg_prompts = [VAL_NEG_PROMPT] * len(prompts)

    pos_tok = tokenizer(prompts, padding="longest", truncation=True, max_length=256, return_tensors="pt")
    neg_tok = tokenizer(neg_prompts, padding="longest", truncation=True, max_length=256, return_tensors="pt")

    with torch.autocast(device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
        pos_emb = text_encoder(
            input_ids=pos_tok.input_ids.to(device),
            attention_mask=pos_tok.attention_mask.to(device),
        ).last_hidden_state
        neg_emb = text_encoder(
            input_ids=neg_tok.input_ids.to(device),
            attention_mask=neg_tok.attention_mask.to(device),
        ).last_hidden_state

    if vae_mode == "eq_sdxl":
        lat_c, lat_h, lat_w = 4, resolution // 8, resolution // 8
    else:
        lat_c, lat_h, lat_w = 128, resolution // 16, resolution // 16

    gen = torch.Generator(device=device).manual_seed(seed)
    z = torch.randn((len(prompts), lat_c, lat_h, lat_w), generator=gen, device=device, dtype=torch.float32)

    aspect_ratio = torch.zeros(len(prompts), device=device, dtype=amp_dtype)
    pos_map = make_axial_pos_no_cache(lat_h, lat_w, device=device)[None].expand(len(prompts), -1, -1).to(amp_dtype)

    dt = 1.0 / num_steps
    t_val = 1.0
    for _ in range(num_steps):
        t = torch.full((len(prompts),), t_val, device=device, dtype=amp_dtype)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
            cond = model(
                z.to(amp_dtype),
                t,
                ctx=pos_emb,
                pos_map=pos_map,
                addon_info=aspect_ratio,
                tread_rate=None,
            ).float()
            uncond = model(
                z.to(amp_dtype),
                t,
                ctx=neg_emb,
                pos_map=pos_map,
                addon_info=aspect_ratio,
                tread_rate=tread_gamma2 if tread_gamma2 > 0 else None,
            ).float()
        v = uncond + cfg_scale * (cond - uncond)
        z = z - dt * v
        t_val -= dt

    if vae_mode == "eq_sdxl":
        imgs = decode_eq_sdxl(vae, z.to(bn_mean.dtype), bn_mean, bn_std)
    else:
        imgs = decode_flux2(vae, z.to(bn_mean.dtype), bn_mean, bn_std)

    # Save 2x2 grid (both PNG and compressed JPEG for low-bandwidth download)
    grid = Image.new("RGB", (resolution * 2, resolution * 2))
    for idx, im in enumerate(imgs):
        im.save(out_dir / f"val_step_{step:04d}_p{idx}.png")
        grid.paste(im, ((idx % 2) * resolution, (idx // 2) * resolution))
    grid_path = out_dir / f"val_step_{step:04d}_grid.png"
    grid_jpg = out_dir / f"val_step_{step:04d}_grid.jpg"
    grid.save(grid_path)
    grid.save(grid_jpg, "JPEG", quality=80, optimize=True)
    print(f"[val] Saved 4 rating-g samples -> {grid_path} & {grid_jpg}")
    model.train()


# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Adapt HDM-xut-340M to FLUX.2 VAE latent space")
    p.add_argument("--vae_mode", type=str, default="flux2", choices=["flux2", "eq_sdxl"], help="flux2 (128ch p=1) or eq_sdxl (native 4ch p=2 sanity check)")
    p.add_argument("--hdm_repo", type=str, default="KBlueLeaf/HDM-xut-340M-anime")
    p.add_argument("--flux2_vae_repo", type=str, default="black-forest-labs/FLUX.2-dev")
    p.add_argument("--flux2_vae_subfolder", type=str, default="vae")
    p.add_argument("--meta_parquet", type=str, default=None, help="Optional local path to posts-snapshot.parquet")
    p.add_argument("--rating_filter", type=str, default="g", help="Danbooru rating filter for training (g/s/q/e or '' for all)")
    p.add_argument("--num_shards", type=int, default=32, help="Number of danbooru2026 tar shards to stream from")
    p.add_argument("--num_workers", type=int, default=4, help="Concurrent HTTP/PIL prefetch threads")
    p.add_argument("--resolution", type=int, default=512, help="Square crop size (must be multiple of 16)")
    p.add_argument("--batch_size", type=int, default=4, help="Micro-batch size per forward")
    p.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    p.add_argument("--steps", type=int, default=1000, help="Total optimizer steps")
    p.add_argument("--warmup_patch_steps", type=int, default=200, help="Steps to train ONLY in_patch/out_patch first (0 to disable)")
    p.add_argument("--lr", type=float, default=0.1, help="Base muP LR (HDM uses 0.1-0.5 because it divides by fan_in)")
    p.add_argument("--patch_lr_mult", type=float, default=3.0, help="Extra LR multiplier on fresh in_patch/out_patch")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--tread_rate", type=float, default=0.5, help="TREAD token drop ratio during training (0.0 to disable)")
    p.add_argument("--val_every", type=int, default=250)
    p.add_argument("--val_steps", type=int, default=24)
    p.add_argument("--val_cfg", type=float, default=3.5)
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--out_dir", type=str, default="./outputs/hdm_flux2_adapt")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    assert args.resolution % 16 == 0, "resolution must be divisible by 16"

    # Disable torch.compile inside unit modules by default for fast startup on Windows
    xut.env.TORCH_COMPILE = False

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load VAE
    if args.vae_mode == "eq_sdxl":
        print(f"Loading native EQ-SDXL-VAE from {args.hdm_repo} (subfolder=vae)...")
        vae = AutoencoderKL.from_pretrained(
            args.hdm_repo, subfolder="vae", torch_dtype=amp_dtype
        ).to(device).eval().requires_grad_(False)
        bn_mean, bn_std = get_eq_sdxl_stats(vae, device, amp_dtype)
    else:
        print(f"Loading FLUX.2 VAE from {args.flux2_vae_repo} (subfolder={args.flux2_vae_subfolder})...")
        vae = AutoencoderKLFlux2.from_pretrained(
            args.flux2_vae_repo,
            subfolder=args.flux2_vae_subfolder or None,
            torch_dtype=amp_dtype,
        ).to(device).eval().requires_grad_(False)
        bn_mean, bn_std = get_flux2_bn_stats(vae, device, amp_dtype)

    # 2. Load Qwen3-0.6B Text Encoder + Tokenizer from HDM repo
    print(f"Loading Qwen3 text encoder & tokenizer from {args.hdm_repo}...")
    tokenizer = AutoTokenizer.from_pretrained(args.hdm_repo, subfolder="tokenizer")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    text_encoder = Qwen3Model.from_pretrained(
        args.hdm_repo, subfolder="text_encoder", torch_dtype=amp_dtype
    ).to(device).eval().requires_grad_(False)

    # 3. Load & Adapt HDM XUDiT Backbone
    model = load_adapted_hdm(args.hdm_repo, device=device, vae_mode=args.vae_mode, grad_ckpt=True)
    if getattr(model, "use_tread", False):
        model.dropout_ratio = args.tread_rate

    # Step 0 baseline validation (runs before parquet indexing if --steps 0 for instant sanity check)
    run_validation(
        model, text_encoder, tokenizer, vae, bn_mean, bn_std,
        out_dir, step=0, resolution=args.resolution, device=device,
        amp_dtype=amp_dtype, vae_mode=args.vae_mode,
        num_steps=args.val_steps, cfg_scale=args.val_cfg, seed=args.seed,
    )
    if args.steps <= 0:
        print("steps <= 0: validation-only sanity check complete.")
        return

    # Phase 1: optional patch-only warmup (only used in flux2 mode)
    def set_backbone_trainable(trainable: bool):
        for name, p in model.named_parameters():
            is_patch = ("in_patch" in name) or ("out_patch" in name)
            p.requires_grad = True if is_patch else trainable

    in_patch_warmup = (args.vae_mode == "flux2") and (args.warmup_patch_steps > 0)
    if in_patch_warmup:
        print(f"Phase 1 (steps 0..{args.warmup_patch_steps}): training ONLY in_patch + out_patch.")
        set_backbone_trainable(False)
    else:
        set_backbone_trainable(True)

    optimizer = build_mup_optimizer(model, base_lr=args.lr, patch_lr_mult=args.patch_lr_mult)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.dtype == "fp16"))

    # 4. Metadata & Streaming Data Generator
    id_to_meta = load_danbooru2026_metadata(
        meta_parquet=args.meta_parquet,
        rating_filter=args.rating_filter or None,
    )
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    latent_stride = 8 if args.vae_mode == "eq_sdxl" else 16
    data_iter = prefetched_danbooru2026_batches(
        id_to_meta=id_to_meta,
        resolution=args.resolution,
        batch_size=args.batch_size,
        shards=list(range(min(args.num_shards, 1000))),
        hf_token=hf_token,
        latent_stride=latent_stride,
        num_workers=args.num_workers,
    )

    # 5. Training Loop (1,000 optimizer steps)
    model.train()
    ema_loss = None
    pbar = tqdm(range(1, args.steps + 1), desc="HDM -> FLUX.2 VAE")
    t0 = time.time()

    for step in pbar:
        # Transition from Phase 1 (patch-only) -> Phase 2 (full backbone)
        if in_patch_warmup and step == args.warmup_patch_steps + 1:
            print(f"\n[step {step}] Unfreezing full HDM backbone!")
            set_backbone_trainable(True)
            optimizer = build_mup_optimizer(model, base_lr=args.lr * 0.5, patch_lr_mult=1.0)
            in_patch_warmup = False

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0

        for _ in range(args.grad_accum):
            imgs, captions, pos_maps, log_ars = next(data_iter)
            imgs = imgs.to(device=device, dtype=amp_dtype, non_blocking=True)
            pos_maps = pos_maps.to(device=device, dtype=amp_dtype, non_blocking=True)
            log_ars = log_ars.to(device=device, dtype=amp_dtype, non_blocking=True)

            # Encode latents
            if args.vae_mode == "eq_sdxl":
                latents = encode_eq_sdxl(vae, imgs, bn_mean, bn_std, sample_mode="sample")
            else:
                latents = encode_flux2(vae, imgs, bn_mean, bn_std, sample_mode="sample")

            # Encode captions with Qwen3-0.6B (10% CFG dropout)
            captions_drop = ["" if random.random() < 0.1 else c for c in captions]
            toks = tokenizer(
                captions_drop,
                padding="max_length",
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype):
                ctx = text_encoder(
                    input_ids=toks.input_ids.to(device),
                    attention_mask=toks.attention_mask.to(device),
                ).last_hidden_state

            # Flow matching: noisy = t * noise + (1 - t) * x, target = noise - x
            b = latents.shape[0]
            noise = torch.randn_like(latents, dtype=torch.float32)
            t = torch.sigmoid(torch.randn(b, 1, 1, 1, device=device))
            noisy = t * noise + (1.0 - t) * latents.float()
            target = noise - latents.float()

            with torch.autocast(device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                pred = model(
                    noisy.to(amp_dtype),
                    t.view(b).to(amp_dtype),
                    ctx=ctx,
                    pos_map=pos_maps,
                    addon_info=log_ars,
                )

            loss = F.mse_loss(pred.float(), target) / args.grad_accum
            scaler.scale(loss).backward()
            step_loss += loss.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        ema_loss = step_loss if ema_loss is None else 0.98 * ema_loss + 0.02 * step_loss
        pbar.set_postfix(
            loss=f"{step_loss:.4f}",
            ema=f"{ema_loss:.4f}",
            phase="patch" if in_patch_warmup else "full",
            s_per_it=f"{(time.time() - t0) / step:.2f}",
        )

        if step % args.val_every == 0 or step == args.steps:
            run_validation(
                model, text_encoder, tokenizer, vae, bn_mean, bn_std,
                out_dir, step=step, resolution=args.resolution, device=device,
                amp_dtype=amp_dtype, vae_mode=args.vae_mode,
                num_steps=args.val_steps, cfg_scale=args.val_cfg, seed=args.seed,
            )

    # Save final adapted weights
    ckpt_out = out_dir / f"hdm_xut_flux2_step_{args.steps}.pt"
    torch.save(model.state_dict(), ckpt_out)
    print(f"Done! Saved adapted checkpoint to {ckpt_out}")


if __name__ == "__main__":
    main()
