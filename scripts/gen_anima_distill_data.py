"""
High-throughput aspect-ratio-bucketed Anima synthetic distillation data generator.

Optimizations:
- Aspect-Ratio Bucket Batching (7 clean ~512^2 buckets, batch_size=8..16)
- torch.compile(pipe.transformer, dynamic=True) + TF32 + cudnn.benchmark
- Optional FP8 quantization via torchao (if installed) for Ada/Hopper Tensor Cores
- Skips negative prompt encoding at CFG <= 1.0 & disables inner step tqdm
- Async multi-threaded WebP + txt saving (ThreadPoolExecutor)
- Streams Danbooru2026 prompts via HTTP range requests (no full parquet download)

Usage:
    python gen_anima_distill_data.py \
        --variant turbo --num_images 256 --batch_size 8 \
        --target_res 512 --steps 10 --cfg 1.0 \
        --compile --fp8 \
        --out_dir /content/outputs/anima_turbo_512_opt
"""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import math
import random
import re
import time
from pathlib import Path

import fsspec
import pyarrow.parquet as pq
import torch
from diffusers import DiffusionPipeline
from PIL import Image, ImageOps

COUNT_TAG_RE = re.compile(r"^([1-6]\+?(girl|boy|other)s?|multiple_(girls|boys|others)|no_humans)$")
KAOMOJI_CHARS = set("^><=;:")
SKIP_TAGS = {"comic", "4koma", "2koma", "monochrome", "greyscale", "speech_bubble", "manga"}

NEG_PROMPT_AESTHETIC = (
    "worst quality, low quality, artist name, blurry, jpeg artifacts, chromatic aberration, watermark"
)
NEG_PROMPT_BASE = (
    "worst quality, low quality, score_1, score_2, score_3, artist name, blurry, jpeg artifacts, chromatic aberration, watermark"
)

# 7 clean aspect-ratio buckets around 512^2 (all multiples of 16)
AR_BUCKETS_512 = [
    (384, 688),  # 9:16  (0.558)
    (416, 624),  # 2:3   (0.667)
    (448, 576),  # 3:4   (0.778)
    (512, 512),  # 1:1   (1.000)
    (576, 448),  # 4:3   (1.286)
    (624, 416),  # 3:2   (1.500)
    (688, 384),  # 16:9  (1.792)
]


def clean_tag(tag: str) -> str:
    tag = tag.strip().lower()
    if any(ch in tag for ch in KAOMOJI_CHARS):
        return tag
    return tag.replace("_", " ")


def year_to_period(year: int) -> str:
    if year >= 2024:
        return "newest"
    if year >= 2021:
        return "recent"
    if year >= 2017:
        return "mid"
    if year >= 2013:
        return "early"
    return "old"


def format_anima_prompt(row: dict, aesthetic_mode: bool = True, max_general_tags: int = 40) -> str:
    """Format a Danbooru2026 parquet row strictly per circlestone-labs/Anima README."""
    sections: list[str] = []

    prefix = []
    created_at = row.get("created_at")
    if created_at is not None and hasattr(created_at, "year"):
        y = created_at.year
        prefix.append(f"year {y}")
        prefix.append(year_to_period(y))

    if aesthetic_mode:
        prefix.extend(["masterpiece", "best quality", "highres"])
    else:
        prefix.extend(["masterpiece", "best quality", "score_7", "highres"])

    rating = row.get("rating", "g")
    rating_map = {"g": "safe", "s": "sensitive", "q": "nsfw", "e": "explicit"}
    if rating in rating_map:
        prefix.append(rating_map[rating])
    sections.extend(prefix)

    raw_general = [t for t in (row.get("tag_string_general") or "").split() if t]
    count_tags, general_tags = [], []
    for t in raw_general:
        if COUNT_TAG_RE.match(t):
            count_tags.append(clean_tag(t))
        else:
            general_tags.append(clean_tag(t))

    sections.extend(count_tags)
    sections.extend(clean_tag(t) for t in (row.get("tag_string_character") or "").split() if t)
    sections.extend(clean_tag(t) for t in (row.get("tag_string_copyright") or "").split() if t)
    sections.extend(f"@{clean_tag(t)}" for t in (row.get("tag_string_artist") or "").split() if t)
    sections.extend(general_tags[:max_general_tags])

    return ", ".join(sections)


def snap_to_ar_bucket(orig_w: int, orig_h: int, target_res: int = 512) -> tuple[int, int]:
    """Snap original (w, h) to the closest aspect-ratio bucket scaled to `target_res`."""
    scale = target_res / 512.0
    buckets = [
        (int(round(w * scale / 16)) * 16, int(round(h * scale / 16)) * 16)
        for w, h in AR_BUCKETS_512
    ]
    if not orig_w or not orig_h:
        return (target_res, target_res)
    log_ar = math.log(orig_w / orig_h)
    return min(buckets, key=lambda bh: abs(math.log(bh[0] / bh[1]) - log_ar))


def sample_danbooru2026_prompts(
    num_images: int,
    rating_filter: str = "g",
    min_score: int = 20,
    min_year: int = 2020,
    filter_comics: bool = True,
    seed: int = 42,
) -> list[dict]:
    """Stream random row groups from nyanko-devs/danbooru2026 parquet via HTTP range requests."""
    rng = random.Random(seed)
    url = (
        "https://huggingface.co/datasets/nyanko-devs/danbooru2026/"
        "resolve/main/metadata/posts-snapshot.parquet"
    )
    cols = [
        "id",
        "created_at",
        "score",
        "rating",
        "image_width",
        "image_height",
        "tag_string_artist",
        "tag_string_character",
        "tag_string_copyright",
        "tag_string_general",
    ]
    print("Opening nyanko-devs/danbooru2026 parquet via HTTP range requests...")
    fs = fsspec.filesystem("https")
    samples: list[dict] = []

    with fs.open(url, "rb") as f:
        pf = pq.ParquetFile(f)
        candidate_rgs = list(range(max(0, pf.num_row_groups - 70), pf.num_row_groups))
        rng.shuffle(candidate_rgs)

        for rg in candidate_rgs:
            if len(samples) >= num_images:
                break
            tbl = pf.read_row_group(rg, columns=cols)
            rows = tbl.to_pylist()
            rng.shuffle(rows)
            for row in rows:
                if rating_filter and row.get("rating") != rating_filter:
                    continue
                if (row.get("score") or 0) < min_score:
                    continue
                created = row.get("created_at")
                if min_year and created is not None and hasattr(created, "year") and created.year < min_year:
                    continue
                gen_list = (row.get("tag_string_general") or "").split()
                if len(gen_list) < 8:
                    continue
                if filter_comics and any(t in SKIP_TAGS for t in gen_list):
                    continue
                w, h = row.get("image_width") or 0, row.get("image_height") or 0
                if w < 256 or h < 256:
                    continue
                samples.append(row)
                if len(samples) >= num_images:
                    break

    print(
        f"Sampled {len(samples)} Danbooru2026 prompts "
        f"(rating={rating_filter}, score>={min_score}, year>={min_year}, filter_comics={filter_comics})."
    )
    return samples


def _convert_anima_comfy_transformer_sd(raw_sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert ComfyUI/Cosmos-Predict2 Anima transformer keys to Diffusers CosmosTransformer3DModel."""
    stripped = {
        (k[len("model.diffusion_model."):] if k.startswith("model.diffusion_model.") else k): v
        for k, v in raw_sd.items()
    }
    try:
        from diffusers.loaders.single_file_utils import convert_cosmos_transformer_checkpoint_to_diffusers
        converted = convert_cosmos_transformer_checkpoint_to_diffusers(stripped)
        return {
            (k[len("model.diffusion_model."):] if k.startswith("model.diffusion_model.") else k): v
            for k, v in converted.items()
        }
    except Exception:
        pass

    out = {}
    for k, v in stripped.items():
        if k.startswith("llm_adapter."):
            continue
        k = k.replace("x_embedder.proj.1.", "patch_embed.proj.")
        k = k.replace("t_embedder.1.linear_1.", "time_embed.t_embedder.linear_1.")
        k = k.replace("t_embedder.1.linear_2.", "time_embed.t_embedder.linear_2.")
        k = k.replace("t_embedding_norm.", "time_embed.norm.")
        k = k.replace("final_layer.linear.", "proj_out.")
        k = k.replace("final_layer.adaln_modulation.1.", "norm_out.linear_1.")
        k = k.replace("final_layer.adaln_modulation.2.", "norm_out.linear_2.")
        k = k.replace("blocks.", "transformer_blocks.")
        k = k.replace(".self_attn.q_proj.", ".attn1.to_q.")
        k = k.replace(".self_attn.k_proj.", ".attn1.to_k.")
        k = k.replace(".self_attn.v_proj.", ".attn1.to_v.")
        k = k.replace(".self_attn.output_proj.", ".attn1.to_out.0.")
        k = k.replace(".self_attn.q_norm.", ".attn1.norm_q.")
        k = k.replace(".self_attn.k_norm.", ".attn1.norm_k.")
        k = k.replace(".cross_attn.q_proj.", ".attn2.to_q.")
        k = k.replace(".cross_attn.k_proj.", ".attn2.to_k.")
        k = k.replace(".cross_attn.v_proj.", ".attn2.to_v.")
        k = k.replace(".cross_attn.output_proj.", ".attn2.to_out.0.")
        k = k.replace(".cross_attn.q_norm.", ".attn2.norm_q.")
        k = k.replace(".cross_attn.k_norm.", ".attn2.norm_k.")
        k = k.replace(".adaln_modulation_self_attn.1.", ".norm1.linear_1.")
        k = k.replace(".adaln_modulation_self_attn.2.", ".norm1.linear_2.")
        k = k.replace(".adaln_modulation_cross_attn.1.", ".norm2.linear_1.")
        k = k.replace(".adaln_modulation_cross_attn.2.", ".norm2.linear_2.")
        k = k.replace(".adaln_modulation_mlp.1.", ".norm3.linear_1.")
        k = k.replace(".adaln_modulation_mlp.2.", ".norm3.linear_2.")
        k = k.replace(".mlp.layer1.", ".ff.net.0.proj.")
        k = k.replace(".mlp.layer2.", ".ff.net.2.")
        out[k] = v
    return out


def load_anima_pipeline(variant: str, model_repo: str | None, use_fp8: bool = False, use_compile: bool = False):
    t0 = time.time()
    if variant == "turbo":
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        print("Loading Anima-Turbo v1.1 single-file weights...")
        pipe = DiffusionPipeline.from_pretrained(
            "CalamitousFelicitousness/Anima-1.0-Aesthetic-Diffusers",
            custom_pipeline="CalamitousFelicitousness/Anima-1.0-Aesthetic-Diffusers",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        ckpt_path = hf_hub_download(
            repo_id=model_repo or "circlestone-labs/Anima",
            filename="split_files/diffusion_models/anima-turbo-v1.1.safetensors",
        )
        raw_sd = load_file(ckpt_path, device="cpu")
        adapter_sd, trns_raw = {}, {}
        for k, v in raw_sd.items():
            clean = k[len("model.diffusion_model."):] if k.startswith("model.diffusion_model.") else k
            if clean.startswith("llm_adapter."):
                adapter_sd[clean[len("llm_adapter."):]] = v
            else:
                trns_raw[k] = v

        m_a, u_a = pipe.llm_adapter.load_state_dict(adapter_sd, strict=False)
        trns_sd = _convert_anima_comfy_transformer_sd(trns_raw)
        m_t, u_t = pipe.transformer.load_state_dict(trns_sd, strict=False)
        print(f"  llm_adapter: missing={len(m_a)}, unexpected={len(u_a)} | transformer: missing={len(m_t)}, unexpected={len(u_t)}")
        pipe.to("cuda", dtype=torch.bfloat16)
        is_modular = False
    elif variant == "base":
        from diffusers import ModularPipeline
        repo = model_repo or "circlestone-labs/Anima-Base-v1.0-Diffusers"
        print(f"Loading Anima-Base from {repo}...")
        pipe = ModularPipeline.from_pretrained(repo)
        pipe.load_components(torch_dtype=torch.bfloat16)
        pipe.to("cuda")
        is_modular = True
    else:
        repo = model_repo or "CalamitousFelicitousness/Anima-1.0-Aesthetic-Diffusers"
        print(f"Loading Anima-Aesthetic from {repo}...")
        pipe = DiffusionPipeline.from_pretrained(
            repo,
            custom_pipeline=repo,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        pipe.to("cuda")
        is_modular = False

    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)

    # Optional FP8 quantization (Ada/Hopper FP8 Tensor Cores)
    if use_fp8:
        try:
            from torchao.quantization import quantize_, float8_dynamic_activation_float8_weight
            print("Applying torchao FP8 dynamic activation + weight quantization to transformer...")
            quantize_(pipe.transformer, float8_dynamic_activation_float8_weight())
            print("  FP8 dynamic quantization enabled!")
        except Exception as e:
            try:
                from torchao.quantization import quantize_, float8_weight_only
                print(f"  FP8 dynamic unavailable ({e}); applying FP8 weight-only...")
                quantize_(pipe.transformer, float8_weight_only())
                print("  FP8 weight-only quantization enabled!")
            except Exception as e2:
                print(f"  [info] Skipping FP8 (torchao unavailable or unsupported: {e2})")

    # Optional per-block torch.compile on transformer (fast ~1 min compile, +48% speedup at bs=8)
    if use_compile:
        print("Enabling per-block torch.compile(mode='default', dynamic=True)...")
        for i, blk in enumerate(pipe.transformer.transformer_blocks):
            pipe.transformer.transformer_blocks[i] = torch.compile(blk, mode="default", dynamic=True)

    print(f"Pipeline ready in {time.time() - t0:.1f}s.")
    return pipe, is_modular


def make_contact_sheet(images: list[Image.Image], tile_size: int = 384, cols: int = 4) -> Image.Image:
    rows = math.ceil(len(images) / cols)
    sheet = Image.new("RGB", (cols * tile_size, rows * tile_size), (18, 18, 18))
    for i, im in enumerate(images):
        thumb = ImageOps.contain(im, (tile_size, tile_size), Image.Resampling.LANCZOS)
        x0 = (i % cols) * tile_size + (tile_size - thumb.width) // 2
        y0 = (i // cols) * tile_size + (tile_size - thumb.height) // 2
        sheet.paste(thumb, (x0, y0))
    return sheet


def parse_args():
    p = argparse.ArgumentParser(description="High-throughput AR-bucketed Anima distillation generator")
    p.add_argument("--variant", type=str, default="turbo", choices=["base", "aesthetic", "turbo"])
    p.add_argument("--model_repo", type=str, default=None)
    p.add_argument("--num_images", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--target_res", type=int, default=512)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--cfg", type=float, default=1.0)
    p.add_argument("--rating_filter", type=str, default="g")
    p.add_argument("--min_score", type=int, default=20)
    p.add_argument("--min_year", type=int, default=2020)
    p.add_argument("--no_filter_comics", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--fp8", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="/content/outputs/anima_turbo_512_opt")
    return p.parse_args()


def main():
    args = parse_args()

    # Global CUDA speed flags
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    aesthetic_mode = args.variant in ("aesthetic", "turbo")
    neg_prompt = NEG_PROMPT_AESTHETIC if aesthetic_mode else NEG_PROMPT_BASE

    # 1. Sample & group prompts into AR buckets
    rows = sample_danbooru2026_prompts(
        num_images=args.num_images,
        rating_filter=args.rating_filter,
        min_score=args.min_score,
        min_year=args.min_year,
        filter_comics=not args.no_filter_comics,
        seed=args.seed,
    )

    bucket_groups: dict[tuple[int, int], list[tuple[int, dict]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        bkt = snap_to_ar_bucket(row["image_width"], row["image_height"], target_res=args.target_res)
        bucket_groups[bkt].append((idx, row))

    # Build ordered list of (bucket, batch_items) — largest buckets first so compile warms up fast
    batches: list[tuple[tuple[int, int], list[tuple[int, dict]]]] = []
    for bkt, items in sorted(bucket_groups.items(), key=lambda kv: -len(kv[1])):
        print(f"  Bucket {bkt[0]}x{bkt[1]}: {len(items)} images")
        for i in range(0, len(items), args.batch_size):
            batches.append((bkt, items[i : i + args.batch_size]))

    # 2. Load & optimize pipeline
    pipe, is_modular = load_anima_pipeline(
        args.variant, args.model_repo, use_fp8=args.fp8, use_compile=args.compile
    )

    # 3. Generate with async WebP saving
    meta_path = out_dir / "metadata.jsonl"
    preview_imgs: list[Image.Image] = []
    save_pool = ThreadPoolExecutor(max_workers=4)
    futures = []

    def save_one(img: Image.Image, img_path: Path, txt_path: Path, prompt: str):
        img.save(img_path, "WEBP", quality=92)
        txt_path.write_text(prompt, encoding="utf-8")

    total_imgs = 0
    steady_imgs = 0
    steady_time = 0.0
    t_all = time.time()

    with open(meta_path, "w", encoding="utf-8") as mf, torch.inference_mode():
        for b_idx, ((w, h), items) in enumerate(batches):
            prompts = [format_anima_prompt(row, aesthetic_mode=aesthetic_mode) for _, row in items]
            gens = [torch.Generator(device="cuda").manual_seed(args.seed + idx) for idx, _ in items]

            torch.cuda.synchronize()
            t0 = time.time()
            call_kwargs = dict(
                prompt=prompts,
                width=w,
                height=h,
                num_inference_steps=args.steps,
                guidance_scale=args.cfg,
                generator=gens,
            )
            if args.cfg > 1.0:
                call_kwargs["negative_prompt"] = [neg_prompt] * len(prompts)
            if is_modular:
                call_kwargs["output"] = "images"
                out = pipe(**call_kwargs)
                imgs = out if isinstance(out, list) else out.images
            else:
                imgs = pipe(**call_kwargs).images
            torch.cuda.synchronize()
            dt = time.time() - t0

            bs = len(imgs)
            total_imgs += bs
            # Exclude first batch (compile/cuDNN warmup) from steady-state benchmark
            if b_idx >= 1:
                steady_imgs += bs
                steady_time += dt

            for (idx, row), prompt, img in zip(items, prompts, imgs):
                pid = row["id"]
                img_path = out_dir / f"{pid}.webp"
                txt_path = out_dir / f"{pid}.txt"
                futures.append(save_pool.submit(save_one, img, img_path, txt_path, prompt))
                if len(preview_imgs) < 16:
                    preview_imgs.append(img.copy())
                rec = {
                    "id": pid,
                    "variant": args.variant,
                    "file": img_path.name,
                    "width": w,
                    "height": h,
                    "orig_width": row["image_width"],
                    "orig_height": row["image_height"],
                    "seed": args.seed + idx,
                    "prompt": prompt,
                }
                mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            mf.flush()

            batch_ips = bs / dt
            steady_ips = (steady_imgs / steady_time) if steady_time > 0 else batch_ips
            print(
                f"[batch {b_idx+1:02d}/{len(batches)}] {w}x{h} bs={bs:2d} | "
                f"{dt:.2f}s ({batch_ips:.2f} img/s) | steady={steady_ips:.2f} img/s ({steady_ips*3600:.0f} img/hr) | "
                f"total={total_imgs}/{len(rows)}"
            )

    for fut in futures:
        fut.result()
    save_pool.shutdown(wait=True)

    wall = time.time() - t_all
    overall_ips = total_imgs / wall
    steady_ips = (steady_imgs / steady_time) if steady_time > 0 else overall_ips

    sheet = make_contact_sheet(preview_imgs, tile_size=384, cols=4)
    sheet_path = out_dir / "preview_grid.jpg"
    sheet.save(sheet_path, "JPEG", quality=82, optimize=True)

    print("=" * 72)
    print(f"DONE: {total_imgs} images in {wall:.1f}s")
    print(f"  Overall (incl. warmup/compile): {overall_ips:.2f} img/s ({overall_ips*3600:.0f} img/hr)")
    print(f"  Steady-state (post-warmup)    : {steady_ips:.2f} img/s ({steady_ips*3600:.0f} img/hr)")
    print(f"  Preview grid: {sheet_path} ({sheet_path.stat().st_size // 1024} KB)")
    print("=" * 72)


if __name__ == "__main__":
    main()
