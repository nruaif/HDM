"""
Resumable multi-GPU RunPod WebDataset generator for Anima-Turbo v1.1 distillation datasets.

Matches `modal_gen_anima_wds.py` 1:1 (same Danbooru2026 deterministic sampling, same prompt
formatting, same shortest-side-512 AR buckets, same .webp/.txt/.json tar layout) so it
resumes seamlessly onto the same Hugging Face dataset repo without duplicate Danbooru IDs!

Usage:
    export HF_TOKEN=hf_...
    python3 runpod_gen_anima_wds.py \
        --hf-repo Shio-Koube/anima-turbo-512-wds \
        --num-shards 60 \
        --shard-size 5000 \
        --batch-size 24
"""

import argparse
import io
import json
import math
import multiprocessing as mp
import os
import random
import re
import shutil
import tarfile
import time
from collections import defaultdict
from pathlib import Path


# Shortest side = 512 (all 32-divisible for Anima + Flux.2 VAE + HDM)
AR_BUCKETS_512 = [
    (512, 896),  # ~9:16
    (512, 768),  # 2:3
    (512, 672),  # ~3:4
    (512, 512),  # 1:1
    (672, 512),  # ~4:3
    (768, 512),  # 3:2
    (896, 512),  # ~16:9
]

SKIP_TAGS = {"comic", "4koma", "2koma", "monochrome", "greyscale", "speech_bubble", "manga"}
NEG_PROMPT_AESTHETIC = (
    "worst quality, low quality, artist name, blurry, jpeg artifacts, chromatic aberration, watermark"
)
NEG_PROMPT_BASE = (
    "worst quality, low quality, score_1, score_2, score_3, artist name, blurry, jpeg artifacts, chromatic aberration, watermark"
)

COUNT_RE = re.compile(r"^([1-6]\+?(girl|boy|other)s?|multiple_(girls|boys|others)|no_humans)$")
KAOMOJI = set("^><=;:")


def _format_anima_prompt(row: dict, aesthetic_mode: bool = True, max_general_tags: int = 40) -> str:
    def clean(tag: str) -> str:
        t = tag.strip().lower()
        return t if any(ch in t for ch in KAOMOJI) else t.replace("_", " ")

    sections = []
    year = row.get("year")
    if year:
        sections.append(f"year {year}")
        if year >= 2024:
            sections.append("newest")
        elif year >= 2021:
            sections.append("recent")
        elif year >= 2017:
            sections.append("mid")
        elif year >= 2013:
            sections.append("early")
        else:
            sections.append("old")

    if aesthetic_mode:
        sections.extend(["masterpiece", "best quality", "highres"])
    else:
        sections.extend(["masterpiece", "best quality", "score_7", "highres"])

    rating_map = {"g": "safe", "s": "sensitive", "q": "nsfw", "e": "explicit"}
    if row.get("rating") in rating_map:
        sections.append(rating_map[row["rating"]])

    raw_gen = [t for t in (row.get("tag_string_general") or "").split() if t]
    counts, gens = [], []
    for t in raw_gen:
        (counts if COUNT_RE.match(t) else gens).append(clean(t))

    sections.extend(counts)
    sections.extend(clean(t) for t in (row.get("tag_string_character") or "").split() if t)
    sections.extend(clean(t) for t in (row.get("tag_string_copyright") or "").split() if t)
    sections.extend(f"@{clean(t)}" for t in (row.get("tag_string_artist") or "").split() if t)
    sections.extend(gens[:max_general_tags])
    return ", ".join(sections)


def _snap_ar(orig_w: int, orig_h: int, target_res: int = 512) -> tuple[int, int]:
    scale = target_res / 512.0
    buckets = [
        (int(round(w * scale / 16)) * 16, int(round(h * scale / 16)) * 16)
        for w, h in AR_BUCKETS_512
    ]
    if not orig_w or not orig_h:
        return (target_res, target_res)
    log_ar = math.log(orig_w / orig_h)
    return min(buckets, key=lambda bh: abs(math.log(bh[0] / bh[1]) - log_ar))


def _convert_anima_comfy_transformer_sd(raw_sd):
    stripped = {
        (k[len("model.diffusion_model."):] if k.startswith("model.diffusion_model.") else k): v
        for k, v in raw_sd.items()
    }
    from diffusers.loaders.single_file_utils import convert_cosmos_transformer_checkpoint_to_diffusers

    converted = convert_cosmos_transformer_checkpoint_to_diffusers(stripped)
    return {
        (k[len("model.diffusion_model."):] if k.startswith("model.diffusion_model.") else k): v
        for k, v in converted.items()
    }


def prepare_and_plan(args) -> tuple[str, list[int]]:
    import fsspec
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    hf_repo = args.hf_repo
    if "/" not in hf_repo:
        username = api.whoami()["name"]
        hf_repo = f"{username}/{hf_repo}"
    api.create_repo(repo_id=hf_repo, repo_type="dataset", exist_ok=True, private=True)

    hf_files = set(api.list_repo_files(repo_id=hf_repo, repo_type="dataset"))
    hf_tars = {f for f in hf_files if f.startswith("shards/data-") and f.endswith(".tar")}

    work_dir = Path(args.work_dir)
    manifests_dir = work_dir / "manifests"
    shards_dir = work_dir / "shards"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    shards_dir.mkdir(parents=True, exist_ok=True)

    # Upload any leftover local shards first
    for p in sorted(shards_dir.glob("data-*.tar")):
        rel = f"shards/{p.name}"
        if rel not in hf_tars:
            print(f"[resume] Uploading leftover local shard {p.name}...", flush=True)
            api.upload_file(
                path_or_fileobj=str(p),
                path_in_repo=rel,
                repo_id=hf_repo,
                repo_type="dataset",
                commit_message=f"Add {p.name}",
            )
            hf_tars.add(rel)
        p.unlink(missing_ok=True)

    needed_manifests = [
        sid for sid in range(args.num_shards)
        if not (manifests_dir / f"shard-{sid:04d}.jsonl").exists()
    ]
    if needed_manifests:
        total_needed = args.num_shards * args.shard_size
        print(f"Sampling {total_needed:,} Danbooru2026 prompts for {args.num_shards} shards (seed={args.seed})...", flush=True)
        rng = random.Random(args.seed)
        url = (
            "https://huggingface.co/datasets/nyanko-devs/danbooru2026/"
            "resolve/main/metadata/posts-snapshot.parquet"
        )
        cols = [
            "id", "created_at", "score", "rating", "image_width", "image_height",
            "tag_string_artist", "tag_string_character", "tag_string_copyright", "tag_string_general",
        ]
        fs = fsspec.filesystem("https")
        picked = []
        with fs.open(url, "rb") as f:
            pf = pq.ParquetFile(f)
            rgs = list(range(max(0, pf.num_row_groups - 90), pf.num_row_groups))
            rng.shuffle(rgs)
            for rg in rgs:
                if len(picked) >= total_needed:
                    break
                tbl = pf.read_row_group(rg, columns=cols)
                rows = tbl.to_pylist()
                rng.shuffle(rows)
                for row in rows:
                    if args.rating_filter and row.get("rating") != args.rating_filter:
                        continue
                    if (row.get("score") or 0) < args.min_score:
                        continue
                    created = row.get("created_at")
                    year = created.year if (created is not None and hasattr(created, "year")) else None
                    if args.min_year and (year is None or year < args.min_year):
                        continue
                    gen = (row.get("tag_string_general") or "").split()
                    if len(gen) < 8 or any(t in SKIP_TAGS for t in gen):
                        continue
                    w, h = row.get("image_width") or 0, row.get("image_height") or 0
                    if w < 256 or h < 256:
                        continue
                    row["year"] = year
                    row.pop("created_at", None)
                    picked.append(row)
                    if len(picked) >= total_needed:
                        break
            # If 90 row groups wasn't enough for >300k images, sample earlier row groups deterministically
            if len(picked) < total_needed and pf.num_row_groups > 90:
                extra_rgs = list(range(0, pf.num_row_groups - 90))
                rng.shuffle(extra_rgs)
                for rg in extra_rgs:
                    if len(picked) >= total_needed:
                        break
                    tbl = pf.read_row_group(rg, columns=cols)
                    rows = tbl.to_pylist()
                    rng.shuffle(rows)
                    for row in rows:
                        if args.rating_filter and row.get("rating") != args.rating_filter:
                            continue
                        if (row.get("score") or 0) < args.min_score:
                            continue
                        created = row.get("created_at")
                        year = created.year if (created is not None and hasattr(created, "year")) else None
                        if args.min_year and (year is None or year < args.min_year):
                            continue
                        gen = (row.get("tag_string_general") or "").split()
                        if len(gen) < 8 or any(t in SKIP_TAGS for t in gen):
                            continue
                        w, h = row.get("image_width") or 0, row.get("image_height") or 0
                        if w < 256 or h < 256:
                            continue
                        row["year"] = year
                        row.pop("created_at", None)
                        picked.append(row)
                        if len(picked) >= total_needed:
                            break

        print(f"Sampled {len(picked):,} prompts.", flush=True)
        for sid in range(args.num_shards):
            mpath = manifests_dir / f"shard-{sid:04d}.jsonl"
            if mpath.exists():
                continue
            chunk = picked[sid * args.shard_size : (sid + 1) * args.shard_size]
            if not chunk:
                break
            with open(mpath, "w", encoding="utf-8") as mf:
                for r in chunk:
                    mf.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved manifests up to shard-{args.num_shards - 1:04d}.jsonl.", flush=True)

    todo = [
        sid for sid in range(args.start_shard, args.num_shards)
        if f"shards/data-{sid:04d}.tar" not in hf_tars
        and (manifests_dir / f"shard-{sid:04d}.jsonl").exists()
    ]
    print(f"Target HF repo: {hf_repo}", flush=True)
    print(f"Resume check: {len(hf_tars)} shards already on HF, {len(todo)} shards todo ({todo[0]:04d}..{todo[-1]:04d} if non-empty)." if todo else f"All {args.num_shards} shards already on HF!", flush=True)
    return hf_repo, todo


def gpu_worker(gpu_id: int, shard_queue: mp.Queue, upload_queue: mp.Queue, args_dict: dict):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    from diffusers import DiffusionPipeline
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    variant = args_dict["variant"]
    print(f"[GPU {gpu_id}] Loading Anima ({variant})...", flush=True)
    if variant == "turbo":
        pipe = DiffusionPipeline.from_pretrained(
            "CalamitousFelicitousness/Anima-1.0-Aesthetic-Diffusers",
            custom_pipeline="CalamitousFelicitousness/Anima-1.0-Aesthetic-Diffusers",
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )
        ckpt = hf_hub_download(
            "circlestone-labs/Anima",
            "split_files/diffusion_models/anima-turbo-v1.1.safetensors",
        )
        raw_sd = load_file(ckpt, device="cpu")
        adapter_sd, trns_raw = {}, {}
        for k, v in raw_sd.items():
            clean = k[len("model.diffusion_model."):] if k.startswith("model.diffusion_model.") else k
            if clean.startswith("llm_adapter."):
                adapter_sd[clean[len("llm_adapter."):]] = v
            else:
                trns_raw[k] = v
        pipe.llm_adapter.load_state_dict(adapter_sd, strict=False)
        pipe.transformer.load_state_dict(_convert_anima_comfy_transformer_sd(trns_raw), strict=False)
        pipe.to("cuda", dtype=torch.bfloat16)
    else:
        pipe = DiffusionPipeline.from_pretrained(
            "CalamitousFelicitousness/Anima-1.0-Aesthetic-Diffusers",
            custom_pipeline="CalamitousFelicitousness/Anima-1.0-Aesthetic-Diffusers",
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )
        pipe.to("cuda", dtype=torch.bfloat16)

    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)

    # Chunked VAE decode (chunk=6 fits easily in 32GB VRAM, much faster than slice=1)
    if hasattr(pipe.vae, "disable_slicing"):
        pipe.vae.disable_slicing()
    orig_decode = pipe.vae.decode
    vae_chunk = args_dict.get("vae_chunk", 6)

    def _chunked_decode(z, return_dict=True, **kw):
        if z.shape[0] <= vae_chunk:
            return orig_decode(z, return_dict=return_dict, **kw)
        outs = [orig_decode(z[i : i + vae_chunk], return_dict=False, **kw)[0] for i in range(0, z.shape[0], vae_chunk)]
        cat = torch.cat(outs, dim=0)
        if not return_dict:
            return (cat,)
        from diffusers.models.autoencoders.vae import DecoderOutput
        return DecoderOutput(sample=cat)

    pipe.vae.decode = _chunked_decode

    if args_dict.get("fp8", True):
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, quantize_
        quantize_(pipe.transformer, Float8DynamicActivationFloat8WeightConfig())
        print(f"[GPU {gpu_id}] Enabled FP8 dynamic quantization on transformer.", flush=True)

    # Per-block torch.compile (+70% speedup, fast compile)
    for i, blk in enumerate(pipe.transformer.transformer_blocks):
        pipe.transformer.transformer_blocks[i] = torch.compile(blk, mode="default", dynamic=True)

    print(f"[GPU {gpu_id}] Ready!", flush=True)

    work_dir = Path(args_dict["work_dir"])
    batch_size = args_dict["batch_size"]
    target_res = args_dict["target_res"]
    steps = args_dict["steps"]
    cfg = args_dict["cfg"]
    seed = args_dict["seed"]
    aesthetic_mode = variant in ("aesthetic", "turbo")
    neg_prompt = NEG_PROMPT_AESTHETIC if aesthetic_mode else NEG_PROMPT_BASE

    while True:
        shard_id = shard_queue.get()
        if shard_id is None:
            break

        manifest_path = work_dir / "manifests" / f"shard-{shard_id:04d}.jsonl"
        rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]

        buckets = defaultdict(list)
        for idx, r in enumerate(rows):
            bkt = _snap_ar(r["image_width"], r["image_height"], target_res=target_res)
            buckets[bkt].append((idx, r))

        tmp_tar = work_dir / "shards" / f"data-{shard_id:04d}.tar.tmp"
        final_tar = work_dir / "shards" / f"data-{shard_id:04d}.tar"
        t0 = time.time()
        count = 0

        from concurrent.futures import ThreadPoolExecutor

        def _encode_batch(batch_items, batch_prompts, batch_imgs, bw, bh):
            encoded = []
            for (idx, r), prompt, img in zip(batch_items, batch_prompts, batch_imgs):
                key = f"{r['id']}"
                buf = io.BytesIO()
                img.save(buf, "WEBP", quality=92)
                webp_bytes = buf.getvalue()
                txt_bytes = prompt.encode("utf-8")
                meta_bytes = json.dumps(
                    {
                        "id": r["id"],
                        "width": bw,
                        "height": bh,
                        "orig_width": r["image_width"],
                        "orig_height": r["image_height"],
                        "variant": variant,
                        "steps": steps,
                        "cfg": cfg,
                        "fp8": args_dict.get("fp8", True),
                        "seed": seed + shard_id * 100000 + idx,
                        "rating": r.get("rating"),
                        "score": r.get("score"),
                        "year": r.get("year"),
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                encoded.append((key, webp_bytes, txt_bytes, meta_bytes))
            return encoded

        def _write_encoded(tf_obj, encoded_items):
            nonlocal count
            for key, webp_bytes, txt_bytes, meta_bytes in encoded_items:
                ti = tarfile.TarInfo(name=f"{key}.webp")
                ti.size = len(webp_bytes)
                tf_obj.addfile(ti, io.BytesIO(webp_bytes))

                ti = tarfile.TarInfo(name=f"{key}.txt")
                ti.size = len(txt_bytes)
                tf_obj.addfile(ti, io.BytesIO(txt_bytes))

                ti = tarfile.TarInfo(name=f"{key}.json")
                ti.size = len(meta_bytes)
                tf_obj.addfile(ti, io.BytesIO(meta_bytes))
                count += 1

        pending_futs = []
        with tarfile.open(tmp_tar, "w") as tf, ThreadPoolExecutor(max_workers=4) as pool, torch.inference_mode():
            for (w, h), items in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
                for start in range(0, len(items), batch_size):
                    batch = items[start : start + batch_size]
                    prompts = [_format_anima_prompt(r, aesthetic_mode=aesthetic_mode) for _, r in batch]
                    gens = [
                        torch.Generator(device="cuda").manual_seed(seed + shard_id * 100000 + idx)
                        for idx, _ in batch
                    ]
                    kwargs = dict(
                        prompt=prompts,
                        width=w,
                        height=h,
                        num_inference_steps=steps,
                        guidance_scale=cfg,
                        generator=gens,
                    )
                    if cfg > 1.0:
                        kwargs["negative_prompt"] = [neg_prompt] * len(prompts)
                    imgs = pipe(**kwargs).images

                    pending_futs.append(pool.submit(_encode_batch, batch, prompts, imgs, w, h))
                    # Keep at most 2 batches in flight so RAM stays bounded while overlapping CPU WebP with GPU
                    while len(pending_futs) >= 2:
                        _write_encoded(tf, pending_futs.pop(0).result())
                        if count % 240 < batch_size or count == len(rows):
                            elapsed = time.time() - t0
                            print(
                                f"[GPU {gpu_id} | shard {shard_id:04d}] {count}/{len(rows)} imgs | {elapsed:.1f}s ({count/elapsed:.2f} img/s)",
                                flush=True,
                            )

            while pending_futs:
                _write_encoded(tf, pending_futs.pop(0).result())
            elapsed = time.time() - t0
            print(
                f"[GPU {gpu_id} | shard {shard_id:04d}] {count}/{len(rows)} imgs | {elapsed:.1f}s ({count/elapsed:.2f} img/s)",
                flush=True,
            )

        shutil.move(str(tmp_tar), str(final_tar))
        dt = time.time() - t0
        print(
            f"[GPU {gpu_id} | shard {shard_id:04d}] DONE {count} imgs in {dt:.1f}s ({count/dt:.2f} img/s) -> queued for HF upload",
            flush=True,
        )
        upload_queue.put((shard_id, str(final_tar)))

    print(f"[GPU {gpu_id}] Worker exiting.", flush=True)


def uploader_worker(hf_repo: str, upload_queue: mp.Queue):
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    while True:
        item = upload_queue.get()
        if item is None:
            break
        shard_id, tar_path = item
        p = Path(tar_path)
        if not p.exists():
            continue
        t0 = time.time()
        rel = f"shards/{p.name}"
        for attempt in range(5):
            try:
                size_mb = p.stat().st_size / (1024 * 1024)
                print(f"[uploader] Uploading {p.name} ({size_mb:.1f} MB) -> {hf_repo}...", flush=True)
                api.upload_file(
                    path_or_fileobj=str(p),
                    path_in_repo=rel,
                    repo_id=hf_repo,
                    repo_type="dataset",
                    commit_message=f"Add {p.name}",
                )
                p.unlink(missing_ok=True)
                print(f"[uploader] Uploaded & cleaned {p.name} in {time.time() - t0:.1f}s.", flush=True)
                break
            except Exception as e:
                print(f"[uploader] Attempt {attempt + 1} failed for {p.name}: {e}", flush=True)
                time.sleep(15 * (attempt + 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-repo", type=str, default="anima-turbo-512-wds")
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=60)
    parser.add_argument("--shard-size", type=int, default=5000)
    parser.add_argument("--variant", type=str, default="turbo")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--target-res", type=int, default=512)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--fp8", action="store_true", default=True)
    parser.add_argument("--no-fp8", dest="fp8", action="store_false")
    parser.add_argument("--vae-chunk", type=int, default=6)
    parser.add_argument("--rating-filter", type=str, default="g")
    parser.add_argument("--min-score", type=int, default=20)
    parser.add_argument("--min-year", type=int, default=2020)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--work-dir", type=str, default="/workspace/anima_wds")
    parser.add_argument("--num-gpus", type=int, default=0, help="0 = auto-detect all GPUs")
    args = parser.parse_args()

    import torch

    num_gpus = args.num_gpus or torch.cuda.device_count()
    assert num_gpus >= 1, "No CUDA GPUs found!"
    print(f"Using {num_gpus} GPU(s).", flush=True)

    # Pre-download model weights once before spawning workers so they don't race on HF cache
    from huggingface_hub import hf_hub_download, snapshot_download

    print("Pre-caching Anima-Turbo v1.1 weights...", flush=True)
    snapshot_download("CalamitousFelicitousness/Anima-1.0-Aesthetic-Diffusers")
    if args.variant == "turbo":
        hf_hub_download(
            "circlestone-labs/Anima",
            "split_files/diffusion_models/anima-turbo-v1.1.safetensors",
        )

    hf_repo, todo = prepare_and_plan(args)
    if not todo:
        return

    mp.set_start_method("spawn", force=True)
    shard_queue: mp.Queue = mp.Queue()
    upload_queue: mp.Queue = mp.Queue()

    for sid in todo:
        shard_queue.put(sid)
    for _ in range(num_gpus):
        shard_queue.put(None)

    up_proc = mp.Process(target=uploader_worker, args=(hf_repo, upload_queue))
    up_proc.start()

    args_dict = vars(args)
    workers = [
        mp.Process(target=gpu_worker, args=(gid, shard_queue, upload_queue, args_dict))
        for gid in range(num_gpus)
    ]
    for w in workers:
        w.start()
    for w in workers:
        w.join()

    upload_queue.put(None)
    up_proc.join()
    print(f"ALL DONE! Uploaded to https://huggingface.co/datasets/{hf_repo}", flush=True)


if __name__ == "__main__":
    main()
