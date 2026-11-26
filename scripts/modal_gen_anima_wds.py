"""
Resumable parallel Modal H100 WebDataset generator for Anima distillation datasets.

Architecture:
1. CPU function `prepare_manifests` samples Danbooru2026 prompts once into deterministic
   1,000-image shard manifests on a Modal Volume (`/staging/manifests/shard-XXXX.jsonl`).
2. Checks both Hugging Face Hub (`hf_repo`) and `/staging/shards/` to skip already-completed shards.
3. `@app.cls(gpu="H100")` loads & per-block-compiles Anima ONCE per container (`@modal.enter()`),
   then generates shards at ~10 img/s (bs=24, AR-bucketed, ~100s per 1,000-image shard).
   Writes atomically to `/staging/shards/data-XXXX.tar` (`.webp` + `.txt` + `.json` per sample).
4. Cheap CPU function `upload_staged_shards` pushes completed tars to Hugging Face in batched
   `upload_folder` commits (avoiding HF commit rate limits) and cleans up uploaded tars.

One-time secure HF token setup (Modal Dashboard -> Secrets -> Hugging Face, OR CLI):
    modal secret create huggingface-secret HF_TOKEN=hf_your_write_token_here

Run (example: 10,000 images = 10 shards of 1,000, ~$1.15 on H100):
    modal run HDM/scripts/modal_gen_anima_wds.py \
        --hf-repo your-username/anima-turbo-512-wds \
        --num-shards 10 \
        --shard-size 1000 \
        --variant turbo
"""

import modal

app = modal.App("anima-wds-generator")

hf_cache = modal.Volume.from_name("anima-hf-cache", create_if_missing=True)
staging_vol = modal.Volume.from_name("anima-wds-staging", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "torchvision",
        "diffusers>=0.37.0",
        "transformers>=4.52.0",
        "accelerate",
        "safetensors",
        "huggingface_hub",
        "pillow",
        "sentencepiece",
    )
    .env({"HF_HOME": "/cache/hf"})
)

cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy>=2.0", "huggingface_hub", "pyarrow", "fsspec", "aiohttp", "requests")
    .env({"HF_HOME": "/cache/hf"})
)

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


def _format_anima_prompt(row: dict, aesthetic_mode: bool = True, max_general_tags: int = 40) -> str:
    import re

    count_re = re.compile(r"^([1-6]\+?(girl|boy|other)s?|multiple_(girls|boys|others)|no_humans)$")
    kaomoji = set("^><=;:")

    def clean(tag: str) -> str:
        t = tag.strip().lower()
        return t if any(ch in t for ch in kaomoji) else t.replace("_", " ")

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
        (counts if count_re.match(t) else gens).append(clean(t))

    sections.extend(counts)
    sections.extend(clean(t) for t in (row.get("tag_string_character") or "").split() if t)
    sections.extend(clean(t) for t in (row.get("tag_string_copyright") or "").split() if t)
    sections.extend(f"@{clean(t)}" for t in (row.get("tag_string_artist") or "").split() if t)
    sections.extend(gens[:max_general_tags])
    return ", ".join(sections)


def _snap_ar(orig_w: int, orig_h: int, target_res: int = 512) -> tuple[int, int]:
    import math

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
    try:
        from diffusers.loaders.single_file_utils import convert_cosmos_transformer_checkpoint_to_diffusers
        converted = convert_cosmos_transformer_checkpoint_to_diffusers(stripped)
        return {
            (k[len("model.diffusion_model."):] if k.startswith("model.diffusion_model.") else k): v
            for k, v in converted.items()
        }
    except Exception:
        pass
    raise RuntimeError("diffusers convert_cosmos_transformer_checkpoint_to_diffusers unavailable")


# ---------------------------------------------------------------------------
# 1. CPU: Build deterministic shard manifests once + check resume state
# ---------------------------------------------------------------------------
@app.function(
    image=cpu_image,
    volumes={"/staging": staging_vol},
    secrets=[hf_secret],
    timeout=1800,
)
def prepare_and_plan(
    hf_repo: str,
    num_shards: int,
    shard_size: int = 1000,
    rating_filter: str = "g",
    min_score: int = 20,
    min_year: int = 2020,
    seed: int = 42,
    delete_shards: str = "",
) -> tuple[str, list[int]]:
    import json
    import os
    import random
    from pathlib import Path
    import fsspec
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    if "/" not in hf_repo:
        username = api.whoami()["name"]
        hf_repo = f"{username}/{hf_repo}"
    api.create_repo(repo_id=hf_repo, repo_type="dataset", exist_ok=True, private=True)

    # Optionally delete specific shards from HF Hub & staging before resume check
    if delete_shards.strip():
        sids = [int(x.strip()) for x in delete_shards.split(",") if x.strip()]
        existing_hf = set(api.list_repo_files(repo_id=hf_repo, repo_type="dataset"))
        for sid in sids:
            rel = f"shards/data-{sid:04d}.tar"
            if rel in existing_hf:
                api.delete_file(
                    path_in_repo=rel,
                    repo_id=hf_repo,
                    repo_type="dataset",
                    commit_message=f"Delete {rel} before regeneration",
                )
                print(f"Deleted {rel} from {hf_repo}")
            p = Path(f"/staging/{rel}")
            if p.exists():
                p.unlink()
                staging_vol.commit()

    # Check which shards already exist on HF Hub or in /staging/shards
    hf_files = set(api.list_repo_files(repo_id=hf_repo, repo_type="dataset"))
    shards_dir = Path("/staging/shards")
    shards_dir.mkdir(parents=True, exist_ok=True)
    staged_files = {f"shards/{p.name}" for p in shards_dir.glob("data-*.tar")}
    done_files = hf_files | staged_files

    manifests_dir = Path("/staging/manifests")
    manifests_dir.mkdir(parents=True, exist_ok=True)
    needed_manifests = [
        sid for sid in range(num_shards)
        if not (manifests_dir / f"shard-{sid:04d}.jsonl").exists()
    ]

    if needed_manifests:
        total_needed = num_shards * shard_size
        print(f"Sampling {total_needed:,} Danbooru2026 prompts for {num_shards} shards...")
        rng = random.Random(seed)
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
                    if rating_filter and row.get("rating") != rating_filter:
                        continue
                    if (row.get("score") or 0) < min_score:
                        continue
                    created = row.get("created_at")
                    year = created.year if (created is not None and hasattr(created, "year")) else None
                    if min_year and (year is None or year < min_year):
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

        for sid in range(num_shards):
            mpath = manifests_dir / f"shard-{sid:04d}.jsonl"
            if mpath.exists():
                continue
            chunk = picked[sid * shard_size : (sid + 1) * shard_size]
            if not chunk:
                break
            with open(mpath, "w", encoding="utf-8") as mf:
                for r in chunk:
                    mf.write(json.dumps(r, ensure_ascii=False) + "\n")
        staging_vol.commit()
        print(f"Saved manifests up to shard-{num_shards - 1:04d}.jsonl.")

    hf_tars = {f for f in hf_files if f.startswith("shards/data-") and f.endswith(".tar")}
    done_tars = hf_tars | staged_files
    todo = [sid for sid in range(num_shards) if f"shards/data-{sid:04d}.tar" not in done_tars]
    print(f"Target HF repo: {hf_repo}")
    print(f"Resume check: {len(done_tars)} shards done (HF={len(hf_tars)}, staged={len(staged_files)}), {len(todo)} todo.")
    return hf_repo, todo


# ---------------------------------------------------------------------------
# 2. H100 Worker Class (loads & compiles ONCE per container, generates shards)
# ---------------------------------------------------------------------------
@app.cls(
    image=gpu_image,
    gpu="H100",
    volumes={"/cache/hf": hf_cache, "/staging": staging_vol},
    secrets=[hf_secret],
    max_containers=4,
    timeout=3600,
)
class AnimaShardWorker:
    variant: str = modal.parameter(default="turbo")

    @modal.enter()
    def setup(self):
        import torch
        from diffusers import DiffusionPipeline
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

        if self.variant == "turbo":
            pipe = DiffusionPipeline.from_pretrained(
                "CalamitousFelicitousness/Anima-1.0-Aesthetic-Diffusers",
                custom_pipeline="CalamitousFelicitousness/Anima-1.0-Aesthetic-Diffusers",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
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
            self.is_modular = False
        elif self.variant == "base":
            from diffusers import ModularPipeline
            pipe = ModularPipeline.from_pretrained("circlestone-labs/Anima-Base-v1.0-Diffusers")
            pipe.load_components(torch_dtype=torch.bfloat16)
            pipe.to("cuda")
            self.is_modular = True
        else:
            pipe = DiffusionPipeline.from_pretrained(
                "CalamitousFelicitousness/Anima-1.0-Aesthetic-Diffusers",
                custom_pipeline="CalamitousFelicitousness/Anima-1.0-Aesthetic-Diffusers",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
            pipe.to("cuda", dtype=torch.bfloat16)
            self.is_modular = False

        if hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)

        # Per-block torch.compile (+70% speedup on H100!)
        for i, blk in enumerate(pipe.transformer.transformer_blocks):
            pipe.transformer.transformer_blocks[i] = torch.compile(blk, mode="default", dynamic=True)

        hf_cache.commit()
        self.pipe = pipe

    @modal.method()
    def generate_shard(
        self,
        shard_id: int,
        batch_size: int = 24,
        target_res: int = 512,
        steps: int = 10,
        cfg: float = 1.0,
        seed: int = 42,
    ) -> dict:
        from collections import defaultdict
        import io
        import json
        import shutil
        import tarfile
        import time
        from pathlib import Path
        import torch

        staging_vol.reload()
        final_tar = Path(f"/staging/shards/data-{shard_id:04d}.tar")
        if final_tar.exists():
            return {"shard_id": shard_id, "skipped": True}

        manifest_path = Path(f"/staging/manifests/shard-{shard_id:04d}.jsonl")
        rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        aesthetic_mode = self.variant in ("aesthetic", "turbo")
        neg_prompt = NEG_PROMPT_AESTHETIC if aesthetic_mode else NEG_PROMPT_BASE

        # Group by AR bucket
        buckets = defaultdict(list)
        for idx, r in enumerate(rows):
            bkt = _snap_ar(r["image_width"], r["image_height"], target_res=target_res)
            buckets[bkt].append((idx, r))

        tmp_tar = Path(f"/tmp/data-{shard_id:04d}.tar.tmp")
        t0 = time.time()
        count = 0

        with tarfile.open(tmp_tar, "w") as tf, torch.inference_mode():
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
                    if self.is_modular:
                        kwargs["output"] = "images"
                        out = self.pipe(**kwargs)
                        imgs = out if isinstance(out, list) else out.images
                    else:
                        imgs = self.pipe(**kwargs).images

                    for (idx, r), prompt, img in zip(batch, prompts, imgs):
                        key = f"{r['id']}"
                        # 1. .webp
                        buf = io.BytesIO()
                        img.save(buf, "WEBP", quality=92)
                        webp_bytes = buf.getvalue()
                        ti = tarfile.TarInfo(name=f"{key}.webp")
                        ti.size = len(webp_bytes)
                        tf.addfile(ti, io.BytesIO(webp_bytes))

                        # 2. .txt
                        txt_bytes = prompt.encode("utf-8")
                        ti = tarfile.TarInfo(name=f"{key}.txt")
                        ti.size = len(txt_bytes)
                        tf.addfile(ti, io.BytesIO(txt_bytes))

                        # 3. .json
                        meta_bytes = json.dumps(
                            {
                                "id": r["id"],
                                "width": w,
                                "height": h,
                                "orig_width": r["image_width"],
                                "orig_height": r["image_height"],
                                "variant": self.variant,
                                "steps": steps,
                                "cfg": cfg,
                                "seed": seed + shard_id * 100000 + idx,
                                "rating": r.get("rating"),
                                "score": r.get("score"),
                                "year": r.get("year"),
                            },
                            ensure_ascii=False,
                        ).encode("utf-8")
                        ti = tarfile.TarInfo(name=f"{key}.json")
                        ti.size = len(meta_bytes)
                        tf.addfile(ti, io.BytesIO(meta_bytes))
                        count += 1

                    if count % 240 < batch_size or count == len(rows):
                        elapsed = time.time() - t0
                        print(f"[shard {shard_id:04d}] {count}/{len(rows)} imgs | {elapsed:.1f}s ({count/elapsed:.2f} img/s)")

        final_tar.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp_tar), str(final_tar))
        staging_vol.commit()
        dt = time.time() - t0
        print(f"[shard {shard_id:04d}] Generated {count} images in {dt:.1f}s ({count/dt:.2f} img/s) -> {final_tar.name}")
        return {"shard_id": shard_id, "count": count, "sec": round(dt, 1), "ips": round(count / dt, 2)}


# ---------------------------------------------------------------------------
# 3. CPU: Batched HF Hub Uploader (1 commit per batch of shards)
# ---------------------------------------------------------------------------
@app.function(
    image=cpu_image,
    volumes={"/staging": staging_vol},
    secrets=[hf_secret],
    timeout=1800,
)
def upload_staged_shards(hf_repo: str, delete_after_upload: bool = True) -> int:
    import os
    from pathlib import Path
    from huggingface_hub import HfApi

    staging_vol.reload()
    shards_dir = Path("/staging/shards")
    tars = sorted(shards_dir.glob("data-*.tar"))
    if not tars:
        return 0

    api = HfApi(token=os.environ["HF_TOKEN"])
    print(f"Uploading {len(tars)} shards to {hf_repo} in 1 commit...")
    api.upload_folder(
        folder_path=str(shards_dir),
        path_in_repo="shards",
        repo_id=hf_repo,
        repo_type="dataset",
        allow_patterns="data-*.tar",
        commit_message=f"Add {len(tars)} shards ({tars[0].name}..{tars[-1].name})",
    )
    if delete_after_upload:
        for p in tars:
            p.unlink(missing_ok=True)
        staging_vol.commit()
    print(f"Uploaded & cleaned {len(tars)} shards.")
    return len(tars)


# ---------------------------------------------------------------------------
# 4. CPU: Cloud-side Orchestrator (safe to turn off local PC after .spawn())
# ---------------------------------------------------------------------------
@app.function(
    image=cpu_image,
    timeout=86400,
)
def orchestrate(
    hf_repo: str = "anima-turbo-512-wds",
    num_shards: int = 10,
    shard_size: int = 1000,
    variant: str = "turbo",
    batch_size: int = 24,
    target_res: int = 512,
    steps: int = 10,
    cfg: float = 1.0,
    upload_every: int = 10,
    rating_filter: str = "g",
    min_score: int = 20,
    min_year: int = 2020,
    seed: int = 42,
    delete_shards: str = "",
):
    hf_repo, todo = prepare_and_plan.remote(
        hf_repo=hf_repo,
        num_shards=num_shards,
        shard_size=shard_size,
        rating_filter=rating_filter,
        min_score=min_score,
        min_year=min_year,
        seed=seed,
        delete_shards=delete_shards,
    )
    if not todo:
        print("All shards already exist on HF Hub! Uploading any leftover staged shards...")
        upload_staged_shards.remote(hf_repo)
        return

    worker = AnimaShardWorker(variant=variant)
    # Process in waves of `upload_every` shards so completed tars flush to HF in 1 commit per wave
    for i in range(0, len(todo), upload_every):
        wave = todo[i : i + upload_every]
        print(f"=== Launching wave {i//upload_every + 1}: shards {wave[0]:04d}..{wave[-1]:04d} ({len(wave)} shards) ===")
        results = list(
            worker.generate_shard.map(
                wave,
                kwargs=dict(
                    batch_size=batch_size,
                    target_res=target_res,
                    steps=steps,
                    cfg=cfg,
                    seed=seed,
                ),
            )
        )
        for r in results:
            print(" ", r)
        upload_staged_shards.remote(hf_repo)

    print(f"All {num_shards} shards complete and uploaded to https://huggingface.co/datasets/{hf_repo} !")


@app.local_entrypoint()
def main(
    hf_repo: str = "anima-turbo-512-wds",
    num_shards: int = 10,
    shard_size: int = 1000,
    variant: str = "turbo",
    batch_size: int = 24,
    target_res: int = 512,
    steps: int = 10,
    cfg: float = 1.0,
    upload_every: int = 10,
    rating_filter: str = "g",
    min_score: int = 20,
    min_year: int = 2020,
    seed: int = 42,
    delete_shards: str = "",
    background: bool = True,
):
    kwargs = dict(
        hf_repo=hf_repo,
        num_shards=num_shards,
        shard_size=shard_size,
        variant=variant,
        batch_size=batch_size,
        target_res=target_res,
        steps=steps,
        cfg=cfg,
        upload_every=upload_every,
        rating_filter=rating_filter,
        min_score=min_score,
        min_year=min_year,
        seed=seed,
        delete_shards=delete_shards,
    )
    if background:
        fc = orchestrate.spawn(**kwargs)
        print(f"Spawned cloud orchestrator! Call ID: {fc.object_id}")
        print("Safe to close terminal or turn off your PC now (ensure you ran with `modal run --detach`).")
    else:
        orchestrate.remote(**kwargs)
