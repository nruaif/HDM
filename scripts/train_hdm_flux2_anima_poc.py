"""
FLUX.2-VAE PoC Trainer for HDM (XUDiT-340M, 128ch p=1)
on `Shio-Koube/anima-turbo-512-wds` (or `Shio-Koube/Anima-gen`).

Supports TWO data modes seamlessly:
  1. Cached Latents Mode (`--cache_dir ./cache/anima_flux2_latents`, DEFAULT):
     - Resumable streaming `.tar` -> `AutoencoderKLFlux2` FP16 (or `--cache_dtype fp8`)
       AR-bucketed `.safetensors` (~1.86 GB/shard FP16, ~0.93 GB/shard FP8, <300 MB RAM during caching).
     - RAM-safe `ShardStreamingLatentStore` (`--max_ram_shards 8`) so even 50 shards (94 GB)
       never OOMs 30 GB system RAM, while keeping VAE offloaded to CPU during training!
  2. Zero-Disk On-the-Fly Mode (`--cache_dir ""`):
     - Multi-threaded HTTP WebDataset streaming + on-the-fly GPU FLUX.2 VAE encoding (0 GB disk).

Features:
  - Pretrained `KBlueLeaf/HDM-xut-340M-anime` backbone adapted to `input_dim=128, patch_size=1`
    with 2-stage warmup (train fresh `in_patch`/`out_patch` first, then unfreeze full model)
  - Tier-3 XUDiT optimizations (`compile_blocks(dynamic=True)`, selective `grad_ckpt`,
    fused muP AdamW, `padding="longest"` Qwen3-0.6B)
  - Contrastive Flow Matching (DeltaFM, arXiv:2506.05350) after patch warmup
  - W&B logging, clean `XUDiT` safetensors checkpoints, and `--resume` support

Usage (Cached 4-shard PoC, ~7.4 GB FP16 or ~3.7 GB FP8):
    python3 -u scripts/train_hdm_flux2_anima_poc.py \
        --hf_repo Shio-Koube/anima-turbo-512-wds \
        --num_shards 4 \
        --cache_dir /workspace/cache/anima_flux2_latents \
        --cache_dtype fp16 \
        --steps 5000 \
        --warmup_patch_steps 250 \
        --batch_size 16 \
        --val_every 500 \
        --save_every 1000 \
        --wandb_project hdm-flux2-poc

Usage (Zero-disk on-the-fly encoding):
    python3 -u scripts/train_hdm_flux2_anima_poc.py --cache_dir "" ...
"""

import argparse
import io
import json
import math
import os
import queue
import random
import re
import sys
import tarfile
import threading
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKLFlux2
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import get_token
from PIL import Image
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from transformers import AutoTokenizer, Qwen3Model

import xut.env
xut.env.TORCH_COMPILE = False
from xut.modules.axial_rope import make_axial_pos_no_cache
from xut.xut import XUDiT


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
# 1. FLUX.2 VAE Helpers (exact Diffusers Flux2Pipeline convention)
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


def get_flux2_bn_stats(vae: AutoencoderKLFlux2, device: torch.device, dtype: torch.dtype):
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
    chunk_size: int = 0,
) -> torch.Tensor:
    """images in [-1, 1], [B, 3, H, W] -> normalized [B, 128, H/16, W/16]."""
    if chunk_size > 0 and images.shape[0] > chunk_size:
        outs = []
        for s in range(0, images.shape[0], chunk_size):
            dist = vae.encode(images[s : s + chunk_size]).latent_dist
            z = dist.sample() if sample_mode == "sample" else dist.mode()
            outs.append(z)
        latents = torch.cat(outs, dim=0)
    else:
        dist = vae.encode(images).latent_dist
        latents = dist.sample() if sample_mode == "sample" else dist.mode()
    latents = patchify_flux2_latents(latents)
    return (latents - bn_mean) / bn_std


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
    imgs = torch.cat([vae.decode(z.unsqueeze(0)).sample for z in latents], dim=0)
    imgs = (imgs.float() / 2.0 + 0.5).clamp(0.0, 1.0)
    imgs = (imgs.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype("uint8")
    return [Image.fromarray(arr) for arr in imgs]


# ---------------------------------------------------------------------------
# 2. Zero-Disk HTTP WebDataset Streaming + On-the-Fly AR Bucketing
# ---------------------------------------------------------------------------
def normalize_anima_caption(raw: str, tag_shuffle: bool = True, tag_dropout: float = 0.1) -> str:
    """Convert Anima `@artist` tags to HDM-style `artist` tags, with optional shuffle/dropout."""
    tags = [t.strip() for t in raw.split(",") if t.strip()]
    clean = [t[1:] if t.startswith("@") else t for t in tags]
    if tag_shuffle and len(clean) > 2:
        head = clean[:2]
        tail = clean[2:]
        random.shuffle(tail)
        clean = head + tail
    if tag_dropout > 0.0 and len(clean) > 3:
        kept = [t for t in clean if random.random() > tag_dropout]
        if kept:
            clean = kept
    return ", ".join(clean)


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img.convert("RGB"), dtype=np.uint8, copy=True)
    return torch.from_numpy(arr).permute(2, 0, 1).float() / 127.5 - 1.0


def list_wds_shards(hf_repo: str, num_shards: int, hf_token: str | None) -> list[str]:
    api = HfApi(token=hf_token)
    files = sorted(
        f for f in api.list_repo_files(repo_id=hf_repo, repo_type="dataset")
        if f.endswith(".tar") and ("data-" in f or "shard" in f)
    )
    assert files, f"No .tar shards found in {hf_repo}!"
    return files[:num_shards] if num_shards > 0 else files


def stream_wds_ar_batches(
    hf_repo: str,
    shard_paths: list[str],
    batch_size: int,
    hf_token: str | None = None,
    tag_dropout: float = 0.1,
    max_bucket_wait: int = 8,
):
    """
    Infinite generator streaming .tar shards directly over HTTP (0 GB disk).
    Groups decoded images by exact (W, H) aspect-ratio bucket and yields homogeneous
    batches `(imgs [B, 3, H, W], captions, (H, W), log_ar)` as soon as a bucket fills.
    """
    headers = {"User-Agent": "hdm-flux2-poc/1.0"}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    buckets_imgs: dict[tuple[int, int], list[torch.Tensor]] = defaultdict(list)
    buckets_caps: dict[tuple[int, int], list[str]] = defaultdict(list)
    order = list(shard_paths)

    while True:
        random.shuffle(order)
        for rel_path in order:
            url = f"https://huggingface.co/datasets/{hf_repo}/resolve/main/{rel_path}"
            pending: dict[str, dict] = {}
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=45) as resp:
                    with tarfile.open(fileobj=resp, mode="r|") as tf:
                        for m in tf:
                            if not m.isfile():
                                continue
                            name = m.name.rsplit("/", 1)[-1]
                            if "." not in name:
                                continue
                            key, ext = name.rsplit(".", 1)
                            if ext not in ("webp", "png", "jpg", "jpeg", "txt"):
                                continue
                            item = pending.setdefault(key, {})
                            item[ext] = tf.extractfile(m).read()
                            img_ext = next((e for e in ("webp", "png", "jpg", "jpeg") if e in item), None)
                            if img_ext and "txt" in item:
                                try:
                                    im = Image.open(io.BytesIO(item[img_ext])).convert("RGB")
                                    w, h = im.size
                                    # FLUX.2 patchified stride = 16
                                    w16 = (w // 16) * 16
                                    h16 = (h // 16) * 16
                                    if w16 < 64 or h16 < 64:
                                        pending.pop(key, None)
                                        continue
                                    if w != w16 or h != h16:
                                        im = im.crop((0, 0, w16, h16))
                                    img_t = pil_to_tensor(im)
                                    cap_raw = item["txt"].decode("utf-8", errors="replace").strip()
                                    cap = normalize_anima_caption(cap_raw, tag_shuffle=True, tag_dropout=tag_dropout)
                                except Exception:
                                    pending.pop(key, None)
                                    continue
                                pending.pop(key, None)

                                bkey = (h16, w16)
                                buckets_imgs[bkey].append(img_t)
                                buckets_caps[bkey].append(cap)

                                if len(buckets_imgs[bkey]) >= batch_size:
                                    imgs_b = torch.stack(buckets_imgs[bkey][:batch_size])
                                    caps_b = buckets_caps[bkey][:batch_size]
                                    del buckets_imgs[bkey][:batch_size]
                                    del buckets_caps[bkey][:batch_size]
                                    log_ar = math.log(w16 / h16)
                                    yield imgs_b, caps_b, bkey, log_ar
            except Exception as e:
                print(f"[stream warn] {rel_path}: {e}. Moving to next shard.", flush=True)
                continue


def prefetched_wds_batches(
    hf_repo: str,
    shard_paths: list[str],
    batch_size: int,
    hf_token: str | None = None,
    tag_dropout: float = 0.1,
    num_workers: int = 4,
    queue_size: int = 8,
):
    """Multi-threaded prefetch wrapper so GPU never waits on HTTP/PIL decoding."""
    q: queue.Queue = queue.Queue(maxsize=queue_size)
    n_workers = max(1, min(num_workers, len(shard_paths)))

    def _worker(w_shards: list[str], wid: int):
        random.seed(1337 + wid)
        gen = stream_wds_ar_batches(
            hf_repo=hf_repo,
            shard_paths=w_shards,
            batch_size=batch_size,
            hf_token=hf_token,
            tag_dropout=tag_dropout,
        )
        for batch in gen:
            q.put(batch)

    for w in range(n_workers):
        t = threading.Thread(target=_worker, args=(shard_paths[w::n_workers], w), daemon=True)
        t.start()

    while True:
        yield q.get()


# ---------------------------------------------------------------------------
# 2b. Resumable Offline FLUX.2 Latent Caching + RAM-Safe Chunked Loader
# ---------------------------------------------------------------------------
@torch.no_grad()
def cache_wds_shards_to_flux2_latents(
    hf_repo: str,
    shard_paths: list[str],
    cache_dir: Path,
    vae: AutoencoderKLFlux2,
    bn_mean: torch.Tensor,
    bn_std: torch.Tensor,
    device: torch.device,
    amp_dtype: torch.dtype,
    hf_token: str | None = None,
    vae_batch_size: int = 16,
    cache_dtype: str = "fp16",
) -> list[tuple[Path, Path]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    save_dt = torch.float8_e4m3fn if cache_dtype == "fp8" else torch.float16
    cached_pairs: list[tuple[Path, Path]] = []

    vae.to(device)
    for rel_path in shard_paths:
        stem = Path(rel_path).stem
        lat_path = cache_dir / f"{stem}_latents.safetensors"
        meta_path = cache_dir / f"{stem}_meta.json"
        if lat_path.exists() and meta_path.exists():
            cached_pairs.append((lat_path, meta_path))
            continue

        t0 = time.time()
        print(f"[cache] Downloading & FLUX.2-encoding {rel_path} ({cache_dtype})...", flush=True)
        local_tar = hf_hub_download(repo_id=hf_repo, filename=rel_path, repo_type="dataset", token=hf_token)

        buckets_imgs: dict[str, list[torch.Tensor]] = defaultdict(list)
        buckets_lats: dict[str, list[torch.Tensor]] = defaultdict(list)
        buckets_caps: dict[str, list[str]] = defaultdict(list)
        total_imgs = 0

        def _flush(bkey: str):
            imgs = buckets_imgs[bkey]
            if not imgs:
                return
            batch_t = torch.stack(imgs).to(device=device, dtype=amp_dtype)
            imgs.clear()
            with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                z = encode_flux2(vae, batch_t, bn_mean, bn_std, sample_mode="sample")
            if save_dt == torch.float8_e4m3fn:
                z = z.float().clamp(-448.0, 448.0).to(save_dt)
            else:
                z = z.to(save_dt)
            buckets_lats[bkey].append(z.cpu())

        pending: dict[str, dict] = {}
        with tarfile.open(local_tar, "r") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                name = m.name.rsplit("/", 1)[-1]
                if "." not in name:
                    continue
                key, ext = name.rsplit(".", 1)
                if ext not in ("webp", "png", "jpg", "jpeg", "txt"):
                    continue
                item = pending.setdefault(key, {})
                item[ext] = tf.extractfile(m).read()
                img_ext = next((e for e in ("webp", "png", "jpg", "jpeg") if e in item), None)
                if img_ext and "txt" in item:
                    try:
                        im = Image.open(io.BytesIO(item[img_ext])).convert("RGB")
                        w, h = im.size
                        w16, h16 = (w // 16) * 16, (h // 16) * 16
                        if w16 < 64 or h16 < 64:
                            pending.pop(key, None)
                            continue
                        if w != w16 or h != h16:
                            im = im.crop((0, 0, w16, h16))
                        bkey = f"{w16}x{h16}"
                        buckets_imgs[bkey].append(pil_to_tensor(im))
                        buckets_caps[bkey].append(item["txt"].decode("utf-8", errors="replace").strip())
                        total_imgs += 1
                    except Exception:
                        pass
                    pending.pop(key, None)
                    if img_ext and f"{w16}x{h16}" in buckets_imgs and len(buckets_imgs[f"{w16}x{h16}"]) >= vae_batch_size:
                        _flush(f"{w16}x{h16}")

        for bkey in list(buckets_imgs.keys()):
            _flush(bkey)

        tensors = {}
        meta = {}
        for bkey, chunks in buckets_lats.items():
            lat = torch.cat(chunks, dim=0)
            tensors[f"lat_{bkey}"] = lat
            meta[bkey] = {"count": int(lat.shape[0]), "captions": buckets_caps[bkey]}

        save_file(tensors, str(lat_path))
        meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        sz_mb = lat_path.stat().st_size / (1024**2)
        print(f"[cache] Saved {stem}: {total_imgs} imgs across {len(meta)} AR buckets ({sz_mb:.1f} MB) in {time.time() - t0:.1f}s", flush=True)
        cached_pairs.append((lat_path, meta_path))

        # Remove downloaded raw tar from HF cache if possible to save disk
        try:
            real_tar = Path(local_tar).resolve()
            if real_tar.exists():
                real_tar.unlink()
        except Exception:
            pass

    vae.to("cpu")
    torch.cuda.empty_cache()
    return cached_pairs


class ChunkedFlux2LatentStore:
    """
    RAM-safe AR-bucketed FLUX.2 latent store.
    Loads up to `max_ram_shards` shards at a time (e.g. 8 shards = ~15 GB FP16 or ~7.5 GB FP8),
    and seamlessly rotates shard chunks if total shards > `max_ram_shards`.
    """

    def __init__(self, cached_pairs: list[tuple[Path, Path]], max_ram_shards: int = 8):
        self.cached_pairs = list(cached_pairs)
        self.max_ram_shards = max(1, max_ram_shards)
        self.buckets: dict[str, dict] = {}
        self.bucket_keys: list[str] = []
        self.bucket_weights: list[int] = []
        self._chunk_idx = 0
        self._chunks = [
            self.cached_pairs[i : i + self.max_ram_shards]
            for i in range(0, len(self.cached_pairs), self.max_ram_shards)
        ]
        self._load_chunk(0)

    def _load_chunk(self, idx: int):
        self.buckets.clear()
        self.bucket_keys.clear()
        self.bucket_weights.clear()
        pairs = self._chunks[idx % len(self._chunks)]
        total = 0
        for lat_path, meta_path in pairs:
            sd = load_file(str(lat_path), device="cpu")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            for bkey, info in meta.items():
                t = sd[f"lat_{bkey}"]  # [N, 128, H/16, W/16] fp16 or fp8
                if bkey not in self.buckets:
                    w, h = map(int, bkey.split("x"))
                    self.buckets[bkey] = {
                        "latents": [t],
                        "captions": list(info["captions"]),
                        "hw_px": (h, w),
                        "log_ar": math.log(w / h),
                    }
                else:
                    self.buckets[bkey]["latents"].append(t)
                    self.buckets[bkey]["captions"].extend(info["captions"])
                total += len(info["captions"])
        for bkey, b in self.buckets.items():
            b["latents"] = torch.cat(b["latents"], dim=0)
            self.bucket_keys.append(bkey)
            self.bucket_weights.append(len(b["captions"]))
        print(
            f"[cache-store] Loaded chunk {(idx % len(self._chunks)) + 1}/{len(self._chunks)} "
            f"({len(pairs)} shards, {total:,} latents, {len(self.bucket_keys)} AR buckets)",
            flush=True,
        )

    def iter_batches(self, batch_size: int, tag_dropout: float = 0.1, seed: int = 42):
        rng = np.random.default_rng(seed)
        epoch_idx = 0
        while True:
            chunk_order = list(range(len(self._chunks)))
            rng.shuffle(chunk_order)
            for cidx in chunk_order:
                if len(self._chunks) > 1 or epoch_idx > 0:
                    if len(self._chunks) > 1:
                        self._load_chunk(cidx)
                all_batches = []
                for bkey in self.bucket_keys:
                    n = len(self.buckets[bkey]["captions"])
                    perm = rng.permutation(n)
                    for start in range(0, n - batch_size + 1, batch_size):
                        all_batches.append((bkey, perm[start : start + batch_size]))
                rng.shuffle(all_batches)
                for bkey, idx_arr in all_batches:
                    b = self.buckets[bkey]
                    idx_t = torch.from_numpy(idx_arr)
                    lats = b["latents"][idx_t]
                    caps = [
                        normalize_anima_caption(b["captions"][int(i)], tag_shuffle=True, tag_dropout=tag_dropout)
                        for i in idx_arr
                    ]
                    yield lats, caps, b["hw_px"], b["log_ar"]
            epoch_idx += 1


# ---------------------------------------------------------------------------
# 3. Model Loading (HDM 4ch p=2 -> FLUX.2 128ch p=1), Optimizer & Validation
# ---------------------------------------------------------------------------
def clean_state_dict(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        if k.startswith("model."):
            k = k[len("model."):]
        k = k.replace("._orig_mod.", ".")
        out[k] = v
    return out


def load_hdm_for_flux2(
    hdm_repo: str,
    device: torch.device,
    grad_ckpt: str | bool = "half",
    resume_path: str = "",
) -> XUDiT:
    cfg_path = hf_hub_download(hdm_repo, "transformer/config.json")
    cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    for k in list(cfg.keys()):
        if k.startswith("_"):
            cfg.pop(k)

    # Adapt patch interface to FLUX.2 patchified latents: [B, 128, H/16, W/16]
    cfg["input_dim"] = 128
    cfg["patch_size"] = 1

    model = XUDiT(**cfg)

    if resume_path:
        raw_sd = load_file(resume_path, device="cpu")
        state_dict = clean_state_dict(raw_sd)
        model.load_state_dict(state_dict, strict=True)
        print(f"Resumed FLUX.2-adapted HDM-340M from {resume_path} (grad_ckpt={grad_ckpt}).", flush=True)
    else:
        ckpt_path = hf_hub_download(hdm_repo, "transformer/diffusion_pytorch_model.safetensors")
        raw_sd = clean_state_dict(load_file(ckpt_path, device="cpu"))
        state_dict = {}
        dropped = []
        for k, v in raw_sd.items():
            if k.startswith("in_patch.") or k.startswith("out_patch."):
                dropped.append(k)
                continue
            state_dict[k] = v
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained HDM-340M backbone from {hdm_repo} -> adapted to FLUX.2 (128ch, p=1).", flush=True)
        print(f"  Re-initialized patch layers ({len(missing)} keys): {missing}", flush=True)
        if unexpected:
            print(f"  Unexpected keys: {unexpected}", flush=True)

    model.set_grad_ckpt(grad_ckpt)
    return model.to(device)


def build_mup_optimizer(
    model: nn.Module,
    base_lr: float,
    patch_lr_mult: float = 3.0,
    weight_decay: float = 0.01,
    use_cuda: bool = True,
) -> torch.optim.Optimizer:
    param_groups = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2:
            fan_in = param.numel()
        elif param.ndim > 2:
            fan_in = math.prod(param.shape[1:])
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
    return torch.optim.AdamW(param_groups, lr=base_lr, betas=(0.9, 0.98), fused=use_cuda)


def set_backbone_trainable(model: nn.Module, trainable: bool):
    for name, param in model.named_parameters():
        if "in_patch" in name or "out_patch" in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(trainable)


@torch.no_grad()
def run_validation(
    model: XUDiT,
    text_encoder: Qwen3Model,
    tokenizer,
    vae: AutoencoderKLFlux2,
    bn_mean: torch.Tensor,
    bn_std: torch.Tensor,
    out_dir: Path,
    step: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    prompts: list[str] | None = None,
    resolution: int = 512,
    num_steps: int = 24,
    cfg_scale: float = 3.5,
    tread_gamma2: float = 0.25,
    seed: int = 42,
) -> Path:
    model.eval()
    prompts = prompts or VAL_PROMPTS
    neg_prompts = [VAL_NEG_PROMPT] * len(prompts)
    pos_tok = tokenizer(prompts, padding="longest", truncation=True, max_length=128, return_tensors="pt")
    neg_tok = tokenizer(neg_prompts, padding="longest", truncation=True, max_length=128, return_tensors="pt")

    with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
        pos_emb = text_encoder(
            input_ids=pos_tok.input_ids.to(device),
            attention_mask=pos_tok.attention_mask.to(device),
        ).last_hidden_state
        neg_emb = text_encoder(
            input_ids=neg_tok.input_ids.to(device),
            attention_mask=neg_tok.attention_mask.to(device),
        ).last_hidden_state

    lat_h, lat_w = resolution // 16, resolution // 16
    gen = torch.Generator(device=device).manual_seed(seed)
    z = torch.randn((len(prompts), 128, lat_h, lat_w), generator=gen, device=device, dtype=torch.float32)
    aspect_ratio = torch.zeros(len(prompts), device=device, dtype=amp_dtype)
    pos_map = make_axial_pos_no_cache(lat_h, lat_w, device=device, dtype=amp_dtype)[None].expand(len(prompts), -1, -1)

    dt = 1.0 / num_steps
    t_val = 1.0
    for _ in range(num_steps):
        t = torch.full((len(prompts),), t_val, device=device, dtype=amp_dtype)
        with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
            cond = model(z.to(amp_dtype), t, ctx=pos_emb, pos_map=pos_map, addon_info=aspect_ratio, tread_rate=0.0).float()
            uncond = model(z.to(amp_dtype), t, ctx=neg_emb, pos_map=pos_map, addon_info=aspect_ratio, tread_rate=tread_gamma2).float()
        z = z - dt * (uncond + cfg_scale * (cond - uncond))
        t_val -= dt

    vae_was_cpu = next(vae.parameters()).device.type == "cpu"
    if vae_was_cpu:
        vae.to(device)
    imgs = decode_flux2(vae, z.to(bn_mean.dtype), bn_mean, bn_std)
    if vae_was_cpu:
        vae.to("cpu")
    torch.cuda.empty_cache()

    cols = 4 if len(imgs) >= 4 else len(imgs)
    rows = math.ceil(len(imgs) / cols)
    grid = Image.new("RGB", (resolution * cols, resolution * rows))
    for idx, im in enumerate(imgs):
        grid.paste(im, ((idx % cols) * resolution, (idx // cols) * resolution))
    grid_jpg = out_dir / f"val_step_{step:05d}_grid.jpg"
    grid.save(grid_jpg, "JPEG", quality=88, optimize=True)
    print(f"[val] Saved {len(imgs)} FLUX.2 samples ({cols}x{rows} @ {resolution}px) -> {grid_jpg}", flush=True)
    model.train()
    return grid_jpg


def load_val_prompts(prompts_file: str) -> list[str]:
    if prompts_file and Path(prompts_file).exists():
        lines = [l.strip() for l in Path(prompts_file).read_text(encoding="utf-8").splitlines() if l.strip()]
        return [re.sub(r"^\[\d+\]\s*", "", l) for l in lines]
    return list(VAL_PROMPTS)


# ---------------------------------------------------------------------------
# 4. Main PoC Training Loop
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="On-the-fly FLUX.2-VAE PoC trainer for HDM-340M on Anima WDS")
    p.add_argument("--hf_repo", type=str, default="Shio-Koube/anima-turbo-512-wds")
    p.add_argument("--hdm_repo", type=str, default="KBlueLeaf/HDM-xut-340M-anime")
    p.add_argument("--flux2_vae_repo", type=str, default="black-forest-labs/FLUX.2-dev")
    p.add_argument("--flux2_vae_subfolder", type=str, default="vae")
    p.add_argument("--num_shards", type=int, default=4, help="Number of WDS shards to cache/stream (0 = all)")
    p.add_argument("--cache_dir", type=str, default="./cache/anima_flux2_latents", help="Latent cache dir (set '' for zero-disk on-the-fly encoding)")
    p.add_argument("--cache_dtype", type=str, default="fp16", choices=["fp16", "fp8"], help="Cached latent precision (fp16=~1.86GB/shard, fp8=~0.93GB/shard)")
    p.add_argument("--max_ram_shards", type=int, default=8, help="Max cached shards kept in RAM at once (prevents RAM OOM)")
    p.add_argument("--vae_batch_size", type=int, default=16, help="Batch size during offline latent caching")
    p.add_argument("--num_workers", type=int, default=4, help="Concurrent HTTP/PIL prefetch threads (on-the-fly mode)")
    p.add_argument("--queue_size", type=int, default=8, help="Prefetched AR-bucketed batches in RAM (on-the-fly mode)")
    p.add_argument("--vae_chunk_size", type=int, default=0, help="Optional micro-chunk for on-the-fly VAE encode (0 = full batch)")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--steps", type=int, default=5000, help="Total optimizer steps for PoC")
    p.add_argument("--warmup_patch_steps", type=int, default=250, help="Steps to train ONLY in_patch/out_patch first (0 to disable)")
    p.add_argument("--lr", type=float, default=0.05, help="Base muP LR (divides by fan_in)")
    p.add_argument("--patch_lr_mult", type=float, default=3.0, help="Extra LR multiplier on fresh 128ch in_patch/out_patch")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--tread_rate", type=float, default=0.5)
    p.add_argument("--delta_fm_lambda", type=float, default=0.05, help="Contrastive Flow Matching (DeltaFM) weight after patch warmup (0 to disable)")
    p.add_argument("--tag_dropout", type=float, default=0.1)
    p.add_argument("--max_ctx_len", type=int, default=128)
    p.add_argument("--grad_ckpt", type=str, default="auto", choices=["auto", "true", "half", "false"])
    p.add_argument("--compile", action="store_true", default=True)
    p.add_argument("--no_compile", dest="compile", action="store_false")
    p.add_argument("--resume", type=str, default="", help="Optional FLUX.2-adapted .safetensors checkpoint to resume")
    p.add_argument("--val_prompts_file", type=str, default="", help="Optional val_prompts.txt to reuse exact prompts")
    p.add_argument("--val_res", type=int, default=512)
    p.add_argument("--val_every", type=int, default=500)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--val_steps", type=int, default=24)
    p.add_argument("--val_cfg", type=float, default=3.5)
    p.add_argument("--out_dir", type=str, default="./outputs/hdm_flux2_anima_poc")
    p.add_argument("--wandb_project", type=str, default="")
    p.add_argument("--wandb_run_name", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        bf16_ok = torch.cuda.is_bf16_supported() and torch.cuda.get_device_capability(device)[0] >= 8
        amp_dtype = torch.bfloat16 if bf16_ok else torch.float16
        vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    else:
        amp_dtype = torch.float32
        vram_gb = 0.0

    if args.grad_ckpt == "auto":
        grad_ckpt = False if vram_gb >= 22.0 else "half"
    elif args.grad_ckpt == "true":
        grad_ckpt = True
    elif args.grad_ckpt == "half":
        grad_ckpt = "half"
    else:
        grad_ckpt = False

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or get_token()
    print(f"Device: {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'} ({vram_gb:.1f} GB) | AMP={amp_dtype} | grad_ckpt={grad_ckpt}", flush=True)

    wb = None
    if args.wandb_project:
        import wandb
        wb = wandb.init(project=args.wandb_project, name=args.wandb_run_name or None, config=vars(args))
        print(f"W&B run initialized: {wb.url}", flush=True)

    # 1. Load FLUX.2 VAE (stays on GPU in amp_dtype for fast on-the-fly encoding)
    print(f"Loading FLUX.2 VAE from {args.flux2_vae_repo}/{args.flux2_vae_subfolder}...", flush=True)
    vae = AutoencoderKLFlux2.from_pretrained(
        args.flux2_vae_repo,
        subfolder=args.flux2_vae_subfolder,
        torch_dtype=amp_dtype,
        token=hf_token,
    ).to(device).eval().requires_grad_(False)
    bn_mean, bn_std = get_flux2_bn_stats(vae, device, amp_dtype)

    # 2. Load Qwen3-0.6B Text Encoder + FLUX.2-adapted HDM-340M
    tokenizer = AutoTokenizer.from_pretrained(args.hdm_repo, subfolder="tokenizer")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    text_encoder = Qwen3Model.from_pretrained(
        args.hdm_repo, subfolder="text_encoder", torch_dtype=amp_dtype
    ).to(device).eval().requires_grad_(False)

    model = load_hdm_for_flux2(args.hdm_repo, device=device, grad_ckpt=grad_ckpt, resume_path=args.resume)
    if getattr(model, "use_tread", False):
        model.dropout_ratio = args.tread_rate
    if args.compile and device.type == "cuda":
        print("Compiling XUDiT blocks (dynamic=True)...", flush=True)
        model.compile_blocks(mode="default", dynamic=True)

    # 3. Setup Data Source: Cached FLUX.2 Latents (RAM-safe) OR Zero-Disk On-the-Fly Streaming
    shard_paths = list_wds_shards(args.hf_repo, args.num_shards, hf_token)
    use_cache = bool(args.cache_dir and args.cache_dir.strip())
    if use_cache:
        print(f"[mode] CACHED FLUX.2 latents ({args.cache_dtype}) -> {args.cache_dir} ({len(shard_paths)} shards, max_ram_shards={args.max_ram_shards})", flush=True)
        cached_pairs = cache_wds_shards_to_flux2_latents(
            hf_repo=args.hf_repo,
            shard_paths=shard_paths,
            cache_dir=Path(args.cache_dir),
            vae=vae,
            bn_mean=bn_mean,
            bn_std=bn_std,
            device=device,
            amp_dtype=amp_dtype,
            hf_token=hf_token,
            vae_batch_size=args.vae_batch_size,
            cache_dtype=args.cache_dtype,
        )
        store = ChunkedFlux2LatentStore(cached_pairs, max_ram_shards=args.max_ram_shards)
        data_iter = store.iter_batches(args.batch_size, tag_dropout=args.tag_dropout, seed=args.seed)
    else:
        print(f"[mode] ZERO-DISK ON-THE-FLY FLUX.2 encoding: streaming {len(shard_paths)} shards ({args.num_workers} workers, queue={args.queue_size})...", flush=True)
        data_iter = prefetched_wds_batches(
            hf_repo=args.hf_repo,
            shard_paths=shard_paths,
            batch_size=args.batch_size,
            hf_token=hf_token,
            tag_dropout=args.tag_dropout,
            num_workers=args.num_workers,
            queue_size=args.queue_size,
        )

    val_prompts = load_val_prompts(args.val_prompts_file)
    (out_dir / "val_prompts.txt").write_text("\n".join(f"[{i+1}] {p}" for i, p in enumerate(val_prompts)), encoding="utf-8")

    # Cache axial pos_maps per (H/16, W/16) bucket on GPU
    pos_map_cache: dict[tuple[int, int], torch.Tensor] = {}

    def get_pos_map(h_px: int, w_px: int, b: int) -> torch.Tensor:
        key = (h_px // 16, w_px // 16)
        pm = pos_map_cache.get(key)
        if pm is None:
            pm = make_axial_pos_no_cache(key[0], key[1], device=device, dtype=amp_dtype)
            pos_map_cache[key] = pm
        return pm.unsqueeze(0).expand(b, -1, -1)

    # 4. Stage 1 (Patch Warmup) vs Stage 2 (Full Fine-tune + DeltaFM)
    in_warmup = (args.warmup_patch_steps > 0) and (not args.resume)
    if in_warmup:
        print(f"=== Stage 1: Patch Warmup (steps 1..{args.warmup_patch_steps}) — training ONLY in_patch & out_patch ===", flush=True)
        set_backbone_trainable(model, False)
    else:
        set_backbone_trainable(model, True)

    optimizer = build_mup_optimizer(
        model,
        base_lr=args.lr,
        patch_lr_mult=args.patch_lr_mult,
        use_cuda=(device.type == "cuda"),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and amp_dtype == torch.float16))

    model.train()
    ema_fm = None
    ema_tot = None
    t0 = time.time()
    pbar = tqdm(range(1, args.steps + 1), desc="HDM-FLUX2 PoC")

    for step in pbar:
        if in_warmup and step == args.warmup_patch_steps + 1:
            in_warmup = False
            print(f"\n=== Stage 2: Unfreezing full XUDiT backbone at step {step} (DeltaFM lambda={args.delta_fm_lambda}) ===", flush=True)
            set_backbone_trainable(model, True)
            optimizer = build_mup_optimizer(
                model,
                base_lr=args.lr,
                patch_lr_mult=args.patch_lr_mult,
                use_cuda=(device.type == "cuda"),
            )

        optimizer.zero_grad(set_to_none=True)
        step_tot = 0.0
        step_fm = 0.0
        step_neg = 0.0
        use_dfm = (not in_warmup) and (args.delta_fm_lambda > 0.0)

        for _ in range(args.grad_accum):
            batch_x, captions, (h_px, w_px), log_ar = next(data_iter)
            b = batch_x.shape[0]
            if use_cache:
                latents = batch_x.to(device=device, dtype=torch.float32, non_blocking=True)
            else:
                imgs = batch_x.to(device=device, dtype=amp_dtype, non_blocking=True)
                # On-the-fly FLUX.2 VAE encode -> [B, 128, H/16, W/16]
                with torch.no_grad(), torch.amp.autocast(device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                    latents = encode_flux2(vae, imgs, bn_mean, bn_std, sample_mode="sample", chunk_size=args.vae_chunk_size)

            captions_drop = ["" if random.random() < 0.1 else c for c in captions]
            toks = tokenizer(captions_drop, padding="longest", truncation=True, max_length=args.max_ctx_len, return_tensors="pt")
            with torch.no_grad(), torch.amp.autocast(device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                ctx = text_encoder(
                    input_ids=toks.input_ids.to(device, non_blocking=True),
                    attention_mask=toks.attention_mask.to(device, non_blocking=True),
                ).last_hidden_state

            pos_map = get_pos_map(h_px, w_px, b)
            log_ars = torch.full((b,), log_ar, device=device, dtype=amp_dtype)

            noise = torch.randn_like(latents, dtype=torch.float32)
            t = torch.sigmoid(torch.randn(b, 1, 1, 1, device=device))
            noisy = t * noise + (1.0 - t) * latents.float()
            target = noise - latents.float()

            with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                pred = model(noisy.to(amp_dtype), t.view(b).to(amp_dtype), ctx=ctx, pos_map=pos_map, addon_info=log_ars)

            pred_f = pred.float()
            fm_loss = F.mse_loss(pred_f, target)
            scale_loss = 1.0 / args.grad_accum
            if use_dfm and b > 1:
                neg_idx = (torch.arange(b, device=device) + torch.randint(1, b, (b,), device=device)) % b
                neg_loss = F.mse_loss(pred_f, target[neg_idx])
                loss = (fm_loss - args.delta_fm_lambda * neg_loss) * scale_loss
                step_neg += float(neg_loss.item()) * scale_loss
            else:
                loss = fm_loss * scale_loss

            scaler.scale(loss).backward()
            step_tot += float(loss.item())
            step_fm += float(fm_loss.item()) * scale_loss

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        ema_fm = step_fm if ema_fm is None else 0.98 * ema_fm + 0.02 * step_fm
        ema_tot = step_tot if ema_tot is None else 0.98 * ema_tot + 0.02 * step_tot
        img_s = (step * args.batch_size * args.grad_accum) / max(1e-3, time.time() - t0)
        stage_tag = "warm" if in_warmup else "full"
        pbar.set_postfix(st=stage_tag, fm=f"{step_fm:.4f}", ema=f"{ema_fm:.4f}", img_s=f"{img_s:.1f}")

        if wb is not None and step % 25 == 0:
            wb.log(
                {
                    "train/fm_loss": step_fm,
                    "train/loss": step_tot,
                    "train/contrastive_loss": step_neg,
                    "train/ema_loss": ema_fm,
                    "train/ema_total_loss": ema_tot,
                    "train/img_s": img_s,
                    "train/stage": 1 if in_warmup else 2,
                },
                step=step,
            )

        if args.val_every > 0 and step % args.val_every == 0:
            g_path = run_validation(
                model, text_encoder, tokenizer, vae, bn_mean, bn_std,
                out_dir, step, device, amp_dtype,
                prompts=val_prompts, resolution=args.val_res,
                num_steps=args.val_steps, cfg_scale=args.val_cfg, seed=args.seed,
            )
            if wb is not None:
                import wandb
                wb.log({"val/grid": wandb.Image(str(g_path), caption=f"FLUX.2 PoC Step {step}")}, step=step)

        if args.save_every > 0 and step % args.save_every == 0:
            clean_sd = clean_state_dict(model.state_dict())
            save_file(clean_sd, str(out_dir / "hdm_flux2_latest.safetensors"))
            save_file(clean_sd, str(out_dir / f"hdm_flux2_step_{step:05d}.safetensors"))

    save_file(clean_state_dict(model.state_dict()), str(out_dir / "hdm_flux2_final.safetensors"))
    print(f"Done in {time.time() - t0:.1f}s! Checkpoints & grids in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
