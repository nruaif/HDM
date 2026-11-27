"""
Fast Tier-2 + Tier-3 PoC Trainer for HDM (XUDiT-340M + native EQ-SDXL-VAE)
on `Shio-Koube/anima-turbo-512-wds`.

Pipeline:
  1. Offline AR-Bucketed Latent Caching (resumable):
     Streams `shards/data-XXXX.tar` from `Shio-Koube/anima-turbo-512-wds`, encodes
     images with `EQ-SDXL-VAE` into FP16 bucketed `.safetensors` (~230 MB / 5k shard),
     then removes VAE from GPU memory during training.
  2. Tier-3 Optimized XUDiT-340M Training:
     - Batched gather/scatter_ TREAD (zero CPU-GPU syncs) + fused half-dim AxialRoPE
     - `model.compile_blocks(dynamic=True)` + selective `grad_ckpt="half"` (or `False` on >=24GB)
     - Live Qwen3-0.6B with `@artist` -> `artist` normalization, tag shuffle/dropout,
       and `padding="longest"` (`max_length=128`) for 2-3x faster text conditioning
     - Hardware-aware AMP (`bf16` on Ampere+, `fp16 + GradScaler` on T4)

Usage (PoC on 4 shards = 20,000 images, ~920 MB cached latents):
    export HF_TOKEN=hf_...
    python3 -u scripts/train_hdm_anima_poc.py \
        --hf_repo Shio-Koube/anima-turbo-512-wds \
        --num_shards 4 \
        --steps 1000 \
        --batch_size 8 \
        --grad_accum 2 \
        --val_every 250
"""

import argparse
import io
import json
import math
import os
import random
import sys
import tarfile
import time
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
from diffusers import AutoencoderKL
from huggingface_hub import HfApi, hf_hub_download
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
# 1. Offline AR-Bucketed EQ-SDXL-VAE Latent Caching
# ---------------------------------------------------------------------------
def get_eq_sdxl_stats(vae: AutoencoderKL, device, dtype):
    mean = torch.tensor(vae.config.latents_mean, device=device, dtype=dtype).view(1, -1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=device, dtype=dtype).view(1, -1, 1, 1)
    return mean, std


@torch.no_grad()
def decode_eq_sdxl(vae: AutoencoderKL, latents: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> list[Image.Image]:
    latents = latents * std + mean
    imgs = torch.cat([vae.decode(z.unsqueeze(0)).sample for z in latents], dim=0)
    imgs = (imgs.float() / 2.0 + 0.5).clamp(0.0, 1.0)
    imgs = (imgs.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype("uint8")
    return [Image.fromarray(arr) for arr in imgs]


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img.convert("RGB"), dtype=np.uint8, copy=True)
    return torch.from_numpy(arr).permute(2, 0, 1).float() / 127.5 - 1.0


@torch.no_grad()
def cache_wds_shards_to_latents(
    hf_repo: str,
    cache_dir: Path,
    num_shards: int,
    vae: AutoencoderKL,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    amp_dtype: torch.dtype,
    vae_batch_size: int = 32,
) -> list[tuple[Path, Path]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    api = HfApi(token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    hf_files = sorted(f for f in api.list_repo_files(repo_id=hf_repo, repo_type="dataset") if f.startswith("shards/data-") and f.endswith(".tar"))
    target_shards = hf_files[:num_shards]
    assert target_shards, f"No shards found in {hf_repo}!"

    cached_pairs = []
    vae.to(device)
    for rel_path in target_shards:
        stem = Path(rel_path).stem  # e.g. data-0000
        lat_path = cache_dir / f"{stem}_latents.safetensors"
        meta_path = cache_dir / f"{stem}_meta.json"
        if lat_path.exists() and meta_path.exists():
            cached_pairs.append((lat_path, meta_path))
            continue

        t0 = time.time()
        print(f"[cache] Downloading {rel_path} from {hf_repo}...", flush=True)
        local_tar = hf_hub_download(repo_id=hf_repo, filename=rel_path, repo_type="dataset")

        buckets_imgs: dict[str, list[torch.Tensor]] = defaultdict(list)
        buckets_lats: dict[str, list[torch.Tensor]] = defaultdict(list)
        buckets_caps: dict[str, list[str]] = defaultdict(list)
        buckets_ids: dict[str, list[str]] = defaultdict(list)
        total_imgs = 0

        def _flush_bucket(bkey: str):
            imgs = buckets_imgs[bkey]
            if not imgs:
                return
            batch_t = torch.stack(imgs).to(device=device, dtype=amp_dtype)
            imgs.clear()
            with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                z = vae.encode(batch_t).latent_dist.sample()
            buckets_lats[bkey].append(((z - mean) / std).to(torch.float16).cpu())

        # Stream tar sequentially and encode each bucket as soon as it reaches vae_batch_size (<300MB RAM!)
        pending: dict[str, dict] = {}
        with tarfile.open(local_tar, "r") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                name = m.name.rsplit("/", 1)[-1]
                if "." not in name:
                    continue
                key, ext = name.rsplit(".", 1)
                if ext not in ("webp", "png", "jpg", "txt"):
                    continue
                item = pending.setdefault(key, {})
                item[ext] = tf.extractfile(m).read()
                img_ext = next((e for e in ("webp", "png", "jpg") if e in item), None)
                if img_ext and "txt" in item:
                    im = Image.open(io.BytesIO(item[img_ext])).convert("RGB")
                    w, h = im.size
                    bkey = f"{w}x{h}"
                    buckets_imgs[bkey].append(pil_to_tensor(im))
                    buckets_caps[bkey].append(item["txt"].decode("utf-8", errors="replace").strip())
                    buckets_ids[bkey].append(key)
                    total_imgs += 1
                    pending.pop(key)
                    if len(buckets_imgs[bkey]) >= vae_batch_size:
                        _flush_bucket(bkey)

        for key, item in pending.items():
            img_ext = next((e for e in ("webp", "png", "jpg") if e in item), None)
            if not img_ext:
                continue
            im = Image.open(io.BytesIO(item[img_ext])).convert("RGB")
            w, h = im.size
            bkey = f"{w}x{h}"
            buckets_imgs[bkey].append(pil_to_tensor(im))
            buckets_caps[bkey].append(item.get("txt", b"").decode("utf-8", errors="replace").strip())
            buckets_ids[bkey].append(key)
            total_imgs += 1

        for bkey in list(buckets_imgs.keys()):
            _flush_bucket(bkey)

        tensors_out = {f"lat_{bkey}": torch.cat(chunks, dim=0) for bkey, chunks in buckets_lats.items()}
        meta_out = {bkey: {"ids": buckets_ids[bkey], "captions": buckets_caps[bkey]} for bkey in buckets_lats}

        save_file(tensors_out, str(lat_path))
        meta_path.write_text(json.dumps(meta_out, ensure_ascii=False), encoding="utf-8")
        # Clean downloaded .tar blob so 50+ shards don't double-use disk space
        try:
            real_tar = Path(local_tar).resolve()
            Path(local_tar).unlink(missing_ok=True)
            if real_tar != Path(local_tar) and real_tar.exists():
                real_tar.unlink(missing_ok=True)
        except Exception:
            pass
        dt = time.time() - t0
        size_mb = lat_path.stat().st_size / (1024 * 1024)
        print(f"[cache] Encoded {total_imgs:,} imgs -> {lat_path.name} ({size_mb:.1f} MB) in {dt:.1f}s ({total_imgs/dt:.1f} img/s)", flush=True)
        cached_pairs.append((lat_path, meta_path))

    # Offload VAE to CPU so 100% of VRAM is freed for XUDiT training
    vae.to("cpu")
    torch.cuda.empty_cache()
    return cached_pairs


# ---------------------------------------------------------------------------
# 2. Fast Bucketed Latent Dataset & Caption Augmentation
# ---------------------------------------------------------------------------
def normalize_anima_caption(cap: str, tag_shuffle: bool = True, tag_dropout: float = 0.1) -> str:
    """Convert Anima `@artist` tags to HDM's plain `artist` format + shuffle/drop general tags."""
    tags = [t.strip() for t in cap.split(",") if t.strip()]
    cleaned = [t[1:] if t.startswith("@") else t for t in tags]
    if tag_dropout > 0.0 and len(cleaned) > 6:
        # Keep first ~5 structural/character tags, randomly drop tail tags
        head, tail = cleaned[:5], cleaned[5:]
        tail = [t for t in tail if random.random() >= tag_dropout]
        if tag_shuffle:
            random.shuffle(tail)
        cleaned = head + tail
    elif tag_shuffle and len(cleaned) > 5:
        head, tail = cleaned[:5], cleaned[5:]
        random.shuffle(tail)
        cleaned = head + tail
    return ", ".join(cleaned)


class BucketedLatentStore:
    """Loads cached FP16 shard files into CPU RAM (mmap/fast) and yields homogeneous AR batches."""

    def __init__(self, cached_pairs: list[tuple[Path, Path]], device: torch.device, amp_dtype: torch.dtype):
        self.buckets: dict[str, dict] = {}
        self.bucket_keys: list[str] = []
        self.bucket_weights: list[int] = []
        total = 0

        for lat_path, meta_path in cached_pairs:
            sd = load_file(str(lat_path), device="cpu")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            for bkey, info in meta.items():
                t = sd[f"lat_{bkey}"]  # [N, 4, H/8, W/8] fp16
                if bkey not in self.buckets:
                    w, h = map(int, bkey.split("x"))
                    lat_h, lat_w = t.shape[2], t.shape[3]
                    pos_map = make_axial_pos_no_cache(lat_h // 2, lat_w // 2, device=device, dtype=amp_dtype)
                    log_ar = math.log(w / h)
                    self.buckets[bkey] = {
                        "latents": [t],
                        "captions": list(info["captions"]),
                        "pos_map": pos_map,
                        "log_ar": log_ar,
                    }
                else:
                    self.buckets[bkey]["latents"].append(t)
                    self.buckets[bkey]["captions"].extend(info["captions"])
                total += len(info["captions"])

        pin_all = (device.type == "cuda") and (total <= 50_000)
        for bkey, b in self.buckets.items():
            cat = torch.cat(b["latents"], dim=0)
            b["latents"] = cat.pin_memory() if pin_all else cat
            self.bucket_keys.append(bkey)
            self.bucket_weights.append(len(b["captions"]))

        print(f"[data] Loaded {total:,} cached latents across {len(self.bucket_keys)} AR buckets: { {k: len(self.buckets[k]['captions']) for k in self.bucket_keys} }", flush=True)

    def sample_batch(self, batch_size: int, tag_dropout: float = 0.1):
        bkey = random.choices(self.bucket_keys, weights=self.bucket_weights, k=1)[0]
        b = self.buckets[bkey]
        n = len(b["captions"])
        idx = torch.randint(0, n, (batch_size,))
        latents = b["latents"][idx]
        captions = [normalize_anima_caption(b["captions"][int(i)], tag_shuffle=True, tag_dropout=tag_dropout) for i in idx]
        pos_map = b["pos_map"].unsqueeze(0).expand(batch_size, -1, -1)
        log_ars = torch.full((batch_size,), b["log_ar"], dtype=pos_map.dtype)
        return latents, captions, pos_map, log_ars

    def iter_epoch_batches(self, batch_size: int, seed: int, rank: int = 0, world_size: int = 1, tag_dropout: float = 0.1):
        """Yield shuffled homogeneous-AR batches covering 1 full epoch across DDP ranks."""
        rng = np.random.default_rng(seed)
        all_batches = []
        for bkey in self.bucket_keys:
            n = len(self.buckets[bkey]["captions"])
            perm = rng.permutation(n)
            for start in range(0, n - batch_size + 1, batch_size):
                all_batches.append((bkey, perm[start : start + batch_size]))
        rng.shuffle(all_batches)
        # Trim so all DDP ranks execute identical number of batches
        usable = (len(all_batches) // world_size) * world_size
        for bkey, idx_arr in all_batches[rank:usable:world_size]:
            b = self.buckets[bkey]
            idx_t = torch.from_numpy(idx_arr)
            latents = b["latents"][idx_t]
            captions = [normalize_anima_caption(b["captions"][int(i)], tag_shuffle=True, tag_dropout=tag_dropout) for i in idx_arr]
            pos_map = b["pos_map"].unsqueeze(0).expand(batch_size, -1, -1)
            log_ars = torch.full((batch_size,), b["log_ar"], dtype=pos_map.dtype)
            yield latents, captions, pos_map, log_ars

    def epoch_num_batches(self, batch_size: int, world_size: int = 1) -> int:
        tot = sum(len(self.buckets[k]["captions"]) // batch_size for k in self.bucket_keys)
        return tot // world_size

    def sample_val_prompts(self, n: int = 12, seed: int = 42) -> list[str]:
        """Deterministically sample n diverse prompts from the cached dataset."""
        rng = random.Random(seed)
        prompts = []
        keys = sorted(self.bucket_keys)
        for i in range(n):
            bkey = keys[i % len(keys)]
            caps = self.buckets[bkey]["captions"]
            raw = caps[rng.randrange(len(caps))]
            prompts.append(normalize_anima_caption(raw, tag_shuffle=False, tag_dropout=0.0))
        return prompts


# ---------------------------------------------------------------------------
# 3. Model Loading, muP Optimizer & Validation
# ---------------------------------------------------------------------------
def load_hdm_sdxl(hdm_repo: str, device: torch.device, grad_ckpt: str | bool = "half") -> XUDiT:
    cfg_path = hf_hub_download(hdm_repo, "transformer/config.json")
    ckpt_path = hf_hub_download(hdm_repo, "transformer/diffusion_pytorch_model.safetensors")
    cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    for k in list(cfg.keys()):
        if k.startswith("_"):
            cfg.pop(k)

    model = XUDiT(**cfg)
    raw_sd = load_file(ckpt_path, device="cpu")
    state_dict = {(k[len("model."):] if k.startswith("model.") else k): v for k, v in raw_sd.items()}
    model.load_state_dict(state_dict, strict=True)
    model.set_grad_ckpt(grad_ckpt)
    print(f"Loaded HDM-xut-340M from {hdm_repo} (grad_ckpt={grad_ckpt}).", flush=True)
    return model.to(device)


def build_mup_optimizer(model: nn.Module, base_lr: float, weight_decay: float = 0.01, use_cuda: bool = True) -> torch.optim.Optimizer:
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
        param_groups.append(
            {
                "params": [param],
                "lr": base_lr / fan_in,
                "weight_decay": 0.0 if param.ndim < 2 else weight_decay,
            }
        )
    return torch.optim.AdamW(param_groups, lr=base_lr, betas=(0.9, 0.98), fused=use_cuda)


@torch.no_grad()
def run_validation(
    model: XUDiT,
    text_encoder: Qwen3Model,
    tokenizer,
    vae: AutoencoderKL,
    mean: torch.Tensor,
    std: torch.Tensor,
    out_dir: Path,
    step: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    prompts: list[str] | None = None,
    num_steps: int = 24,
    cfg_scale: float = 3.5,
    tread_gamma2: float = 0.25,
    seed: int = 42,
):
    model.eval()
    resolution = 512
    prompts = prompts or VAL_PROMPTS
    neg_prompts = [VAL_NEG_PROMPT] * len(prompts)
    pos_tok = tokenizer(prompts, padding="longest", truncation=True, max_length=128, return_tensors="pt")
    neg_tok = tokenizer(neg_prompts, padding="longest", truncation=True, max_length=128, return_tensors="pt")

    with torch.autocast(device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
        pos_emb = text_encoder(input_ids=pos_tok.input_ids.to(device), attention_mask=pos_tok.attention_mask.to(device)).last_hidden_state
        neg_emb = text_encoder(input_ids=neg_tok.input_ids.to(device), attention_mask=neg_tok.attention_mask.to(device)).last_hidden_state

    lat_h, lat_w = resolution // 8, resolution // 8
    gen = torch.Generator(device=device).manual_seed(seed)
    z = torch.randn((len(prompts), 4, lat_h, lat_w), generator=gen, device=device, dtype=torch.float32)
    aspect_ratio = torch.zeros(len(prompts), device=device, dtype=amp_dtype)
    pos_map = make_axial_pos_no_cache(lat_h // 2, lat_w // 2, device=device, dtype=amp_dtype)[None].expand(len(prompts), -1, -1)

    dt = 1.0 / num_steps
    t_val = 1.0
    for _ in range(num_steps):
        t = torch.full((len(prompts),), t_val, device=device, dtype=amp_dtype)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
            cond = model(z.to(amp_dtype), t, ctx=pos_emb, pos_map=pos_map, addon_info=aspect_ratio, tread_rate=0.0).float()
            uncond = model(z.to(amp_dtype), t, ctx=neg_emb, pos_map=pos_map, addon_info=aspect_ratio, tread_rate=tread_gamma2).float()
        z = z - dt * (uncond + cfg_scale * (cond - uncond))
        t_val -= dt

    vae.to(device)
    imgs = decode_eq_sdxl(vae, z.to(mean.dtype), mean, std)
    vae.to("cpu")
    torch.cuda.empty_cache()

    cols = 4 if len(imgs) >= 4 else len(imgs)
    rows = math.ceil(len(imgs) / cols)
    grid = Image.new("RGB", (resolution * cols, resolution * rows))
    for idx, im in enumerate(imgs):
        grid.paste(im, ((idx % cols) * resolution, (idx // cols) * resolution))
    grid_jpg = out_dir / f"val_step_{step:05d}_grid.jpg"
    grid.save(grid_jpg, "JPEG", quality=88, optimize=True)
    print(f"[val] Saved {len(imgs)} samples ({cols}x{rows}) -> {grid_jpg}", flush=True)
    model.train()
    return grid_jpg


def parse_args():
    p = argparse.ArgumentParser(description="Fast PoC HDM-340M training on Anima WDS with Tier-2+3 optimizations")
    p.add_argument("--hf_repo", type=str, default="Shio-Koube/anima-turbo-512-wds")
    p.add_argument("--hdm_repo", type=str, default="KBlueLeaf/HDM-xut-340M-anime")
    p.add_argument("--cache_dir", type=str, default="./cache/anima_eqsdxl_latents")
    p.add_argument("--out_dir", type=str, default="./outputs/hdm_anima_poc")
    p.add_argument("--num_shards", type=int, default=50, help="Number of 5k WDS shards to cache & train on")
    p.add_argument("--vae_batch_size", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--epochs", type=int, default=100, help="Number of full dataset epochs (>0 overrides --steps)")
    p.add_argument("--steps", type=int, default=0, help="Optional fixed step count (used only if --epochs 0)")
    p.add_argument("--lr", type=float, default=0.05, help="muP LR (divides by fan_in)")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--tread_rate", type=float, default=0.5)
    p.add_argument("--tag_dropout", type=float, default=0.1)
    p.add_argument("--max_ctx_len", type=int, default=128)
    p.add_argument("--grad_ckpt", type=str, default="auto", choices=["auto", "true", "half", "false"])
    p.add_argument("--compile", action="store_true", default=True)
    p.add_argument("--no_compile", dest="compile", action="store_false")
    p.add_argument("--val_every", type=int, default=0, help="Step-based val interval (0 = eval every epoch when --epochs > 0)")
    p.add_argument("--save_every", type=int, default=5, help="Epoch interval (when --epochs>0) or step interval to save checkpoints")
    p.add_argument("--val_steps", type=int, default=24)
    p.add_argument("--val_cfg", type=float, default=3.5)
    p.add_argument("--val_num_prompts", type=int, default=12, help="Number of dataset prompts for validation grid")
    p.add_argument("--wandb_project", type=str, default="", help="W&B project name (empty = disabled)")
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
    print(f"Device: {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'} ({vram_gb:.1f} GB) | AMP={amp_dtype} | grad_ckpt={grad_ckpt}", flush=True)

    wb = None
    if args.wandb_project:
        import wandb
        wb = wandb.init(project=args.wandb_project, name=args.wandb_run_name or None, config=vars(args))
        print(f"W&B run initialized: {wb.url}", flush=True)

    # 1. Load EQ-SDXL-VAE and cache WDS shards offline
    vae = AutoencoderKL.from_pretrained(args.hdm_repo, subfolder="vae", torch_dtype=amp_dtype).eval().requires_grad_(False)
    mean, std = get_eq_sdxl_stats(vae, device, amp_dtype)
    cached_pairs = cache_wds_shards_to_latents(
        hf_repo=args.hf_repo,
        cache_dir=Path(args.cache_dir),
        num_shards=args.num_shards,
        vae=vae,
        mean=mean,
        std=std,
        device=device,
        amp_dtype=amp_dtype,
        vae_batch_size=args.vae_batch_size,
    )
    store = BucketedLatentStore(cached_pairs, device=device, amp_dtype=amp_dtype)

    # 2. Load Qwen3-0.6B Text Encoder + HDM-340M
    tokenizer = AutoTokenizer.from_pretrained(args.hdm_repo, subfolder="tokenizer")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    text_encoder = Qwen3Model.from_pretrained(args.hdm_repo, subfolder="text_encoder", torch_dtype=amp_dtype).to(device).eval().requires_grad_(False)

    model = load_hdm_sdxl(args.hdm_repo, device=device, grad_ckpt=grad_ckpt)
    if getattr(model, "use_tread", False):
        model.dropout_ratio = args.tread_rate
    if args.compile and device.type == "cuda":
        print("Compiling XUDiT blocks (dynamic=True)...", flush=True)
        model.compile_blocks(mode="default", dynamic=True)

    # Sample deterministic validation prompts from the dataset
    val_prompts = store.sample_val_prompts(n=args.val_num_prompts, seed=args.seed)
    (out_dir / "val_prompts.txt").write_text("\n".join(f"[{i+1}] {p}" for i, p in enumerate(val_prompts)), encoding="utf-8")

    # Step 0 baseline validation
    g0 = run_validation(model, text_encoder, tokenizer, vae, mean, std, out_dir, 0, device, amp_dtype, prompts=val_prompts, num_steps=args.val_steps, cfg_scale=args.val_cfg, seed=args.seed)
    if wb is not None:
        import wandb
        wb.log({"val/grid": wandb.Image(str(g0), caption="Step 0 (baseline)"), "epoch": 0}, step=0)

    optimizer = build_mup_optimizer(model, base_lr=args.lr, use_cuda=(device.type == "cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and amp_dtype == torch.float16))

    # 3. Fast Training Loop (Epoch-based when --epochs > 0, else Step-based)
    model.train()
    ema_loss = None
    t0 = time.time()

    def _train_step(latents, captions, pos_map, log_ars, scale_loss: float = 1.0):
        latents = latents.to(device=device, non_blocking=True)
        pos_map = pos_map.to(device=device, non_blocking=True)
        log_ars = log_ars.to(device=device, non_blocking=True)

        captions_drop = ["" if random.random() < 0.1 else c for c in captions]
        toks = tokenizer(captions_drop, padding="longest", truncation=True, max_length=args.max_ctx_len, return_tensors="pt")
        with torch.no_grad(), torch.amp.autocast(device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
            ctx = text_encoder(input_ids=toks.input_ids.to(device, non_blocking=True), attention_mask=toks.attention_mask.to(device, non_blocking=True)).last_hidden_state

        b = latents.shape[0]
        noise = torch.randn_like(latents, dtype=torch.float32)
        t = torch.sigmoid(torch.randn(b, 1, 1, 1, device=device))
        noisy = t * noise + (1.0 - t) * latents.float()
        target = noise - latents.float()

        with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
            pred = model(noisy.to(amp_dtype), t.view(b).to(amp_dtype), ctx=ctx, pos_map=pos_map, addon_info=log_ars)
        loss = F.mse_loss(pred.float(), target) * scale_loss
        scaler.scale(loss).backward()
        return float(loss.item())

    if args.epochs > 0:
        global_step = 0
        for epoch in range(1, args.epochs + 1):
            ep_t0 = time.time()
            ep_losses = []
            optimizer.zero_grad(set_to_none=True)
            accum_cnt = 0
            step_loss = 0.0
            n_ep_batches = store.epoch_num_batches(args.batch_size)
            pbar = tqdm(
                store.iter_epoch_batches(args.batch_size, seed=args.seed + epoch, tag_dropout=args.tag_dropout),
                total=n_ep_batches,
                desc=f"Epoch {epoch}/{args.epochs}",
            )
            for latents, captions, pos_map, log_ars in pbar:
                step_loss += _train_step(latents, captions, pos_map, log_ars, scale_loss=1.0 / args.grad_accum)
                accum_cnt += 1
                if accum_cnt == args.grad_accum:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    ep_losses.append(step_loss)
                    ema_loss = step_loss if ema_loss is None else 0.98 * ema_loss + 0.02 * step_loss
                    pbar.set_postfix(loss=f"{step_loss:.4f}", ema=f"{ema_loss:.4f}")
                    accum_cnt = 0
                    step_loss = 0.0
                    if wb is not None and global_step % 50 == 0:
                        wb.log(
                            {
                                "train/loss": ep_losses[-1],
                                "train/ema_loss": ema_loss,
                                "train/epoch": epoch,
                                "train/img_s": (global_step * args.batch_size * args.grad_accum) / max(1e-3, time.time() - t0),
                            },
                            step=global_step,
                        )
                    if args.val_every > 0 and global_step % args.val_every == 0:
                        g_path = run_validation(model, text_encoder, tokenizer, vae, mean, std, out_dir, global_step, device, amp_dtype, prompts=val_prompts, num_steps=args.val_steps, cfg_scale=args.val_cfg, seed=args.seed)
                        if wb is not None:
                            import wandb
                            wb.log({"val/grid": wandb.Image(str(g_path), caption=f"Step {global_step}")}, step=global_step)

            ep_dt = time.time() - ep_t0
            mean_ep = sum(ep_losses) / max(1, len(ep_losses))
            print(f"=== Epoch {epoch}/{args.epochs} DONE ({ep_dt:.1f}s) | mean_loss={mean_ep:.4f} | ema={ema_loss:.4f} | step={global_step} ===", flush=True)
            g_path = run_validation(model, text_encoder, tokenizer, vae, mean, std, out_dir, epoch, device, amp_dtype, prompts=val_prompts, num_steps=args.val_steps, cfg_scale=args.val_cfg, seed=args.seed)
            if wb is not None:
                import wandb
                wb.log(
                    {
                        "epoch/mean_loss": mean_ep,
                        "epoch/ema_loss": ema_loss,
                        "epoch/time_sec": ep_dt,
                        "epoch": epoch,
                        "val/grid": wandb.Image(str(g_path), caption=f"Epoch {epoch}"),
                    },
                    step=global_step,
                )
            save_file(model.state_dict(), str(out_dir / "hdm_latest.safetensors"))
            if args.save_every > 0 and (epoch % args.save_every == 0 or epoch == args.epochs):
                save_file(model.state_dict(), str(out_dir / f"hdm_epoch_{epoch:03d}.safetensors"))
    else:
        pbar = tqdm(range(1, args.steps + 1), desc="HDM Anima PoC")
        for step in pbar:
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for _ in range(args.grad_accum):
                latents, captions, pos_map, log_ars = store.sample_batch(args.batch_size, tag_dropout=args.tag_dropout)
                step_loss += _train_step(latents, captions, pos_map, log_ars, scale_loss=1.0 / args.grad_accum)

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            ema_loss = step_loss if ema_loss is None else 0.98 * ema_loss + 0.02 * step_loss
            pbar.set_postfix(loss=f"{step_loss:.4f}", ema=f"{ema_loss:.4f}", img_s=f"{(step * args.batch_size * args.grad_accum) / (time.time() - t0):.1f}")

            if args.val_every > 0 and step % args.val_every == 0:
                run_validation(model, text_encoder, tokenizer, vae, mean, std, out_dir, step, device, amp_dtype, prompts=val_prompts, num_steps=args.val_steps, cfg_scale=args.val_cfg, seed=args.seed)
            if args.save_every > 0 and step % args.save_every == 0:
                save_file(model.state_dict(), str(out_dir / f"hdm_step_{step:05d}.safetensors"))

    save_file(model.state_dict(), str(out_dir / "hdm_final.safetensors"))
    print(f"Training complete in {time.time() - t0:.1f}s! Checkpoints & grids in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
