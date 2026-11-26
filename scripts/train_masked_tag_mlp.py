"""
Dual-GPU (DDP) / Kaggle 2xT4-ready Masked Tag Completion MLP on Danbooru 2026
(`general + character + copyright`, ignoring `artist` and `meta`, `df >= 50`)
with 75% random tag masking and Asymmetric Loss (ASL).

Highlights:
  - Auto Dual-GPU DDP: Just run `python3 train_masked_tag_mlp.py ...` — if >=2 GPUs are detected
    (e.g. Kaggle 2x T4), it builds a low-RAM (<2GB) memory-mapped cache once and spawns NCCL DDP
    across all GPUs automatically! Also supports `torchrun` or single-GPU.
  - Hardware-aware AMP: Automatically uses `FP16 + GradScaler` on Turing GPUs (Kaggle T4 `sm_75`,
    which lack native BF16 Tensor Cores) and `BF16` on Ampere/Ada/Hopper/Blackwell (`sm_80+`).
  - Compact `20.5M`-param weight-tied `d_model=256` architecture + marginal log-odds bias init.
  - Pure vectorized NumPy 75%-masking (~0.5ms/batch) + background prefetch thread + fused ASL.
"""

import argparse
import json
import os
import queue
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP


class ResMLPBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.05):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc2(self.drop(F.gelu(self.fc1(self.norm(x)))))
        return x + self.drop(h)


class MaskedTagMLP(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        d_ff: int = 1024,
        num_blocks: int = 3,
        dropout: float = 0.05,
        tie_weights: bool = True,
        init_log_odds: np.ndarray | None = None,
    ):
        super().__init__()
        self.emb = nn.EmbeddingBag(vocab_size, d_model, mode="sum", sparse=False)
        nn.init.normal_(self.emb.weight, std=d_model ** -0.5)
        self.in_norm = nn.LayerNorm(d_model)
        self.blocks = nn.ModuleList([ResMLPBlock(d_model, d_ff, dropout) for _ in range(num_blocks)])
        self.out_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=True)
        if tie_weights:
            self.head.weight = self.emb.weight
        # Small initial scale so step-0 logits stay centered on empirical marginal log-odds
        self.logit_scale = nn.Parameter(torch.tensor(0.25))

        if init_log_odds is not None:
            with torch.no_grad():
                self.head.bias.copy_(torch.from_numpy(init_log_odds).float())
        else:
            nn.init.constant_(self.head.bias, -6.0)

    def forward(self, flat_ids: torch.Tensor, bag_offsets: torch.Tensor, vis_counts: torch.Tensor) -> torch.Tensor:
        x = self.emb(flat_ids, bag_offsets) / vis_counts.clamp(min=1.0).sqrt().unsqueeze(1)
        x = self.in_norm(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.out_norm(x) * self.logit_scale
        return self.head(x)


def _neg_asl_elem(logits: torch.Tensor, gamma_neg: float, clip: float) -> torch.Tensor:
    x = logits.float()
    probs = torch.sigmoid(x)
    xs_neg = (1.0 - probs + clip).clamp(max=1.0)
    neg_log = torch.log(xs_neg.clamp(min=1e-8))
    if gamma_neg > 0.0:
        return -((1.0 - xs_neg).pow(gamma_neg)) * neg_log
    return -neg_log


def _dense_neg_asl_sum(logits: torch.Tensor, gamma_neg: float, clip: float) -> torch.Tensor:
    return _neg_asl_elem(logits, gamma_neg, clip).sum(dtype=torch.float64)


_compiled_dense_neg_asl_sum = None
_compile_failed = False


def asymmetric_loss_masked_fast(
    logits: torch.Tensor,
    target_rows: torch.Tensor,
    target_cols: torch.Tensor,
    vis_rows: torch.Tensor,
    vis_cols: torch.Tensor,
    gamma_neg: float = 4.0,
    gamma_pos: float = 0.0,
    clip: float = 0.05,
    use_compile: bool = True,
) -> torch.Tensor:
    global _compiled_dense_neg_asl_sum, _compile_failed
    B = logits.shape[0]
    if use_compile and logits.is_cuda and not _compile_failed:
        try:
            if _compiled_dense_neg_asl_sum is None:
                _compiled_dense_neg_asl_sum = torch.compile(_dense_neg_asl_sum)
            all_neg = _compiled_dense_neg_asl_sum(logits, gamma_neg, clip)
        except Exception as e:
            print(f"[warn] torch.compile fallback to eager ASL: {e}", flush=True)
            _compile_failed = True
            all_neg = _dense_neg_asl_sum(logits, gamma_neg, clip)
    else:
        all_neg = _dense_neg_asl_sum(logits, gamma_neg, clip)

    vis_neg = _neg_asl_elem(logits[vis_rows, vis_cols], gamma_neg, clip).sum(dtype=torch.float64)
    z_tgt = logits[target_rows, target_cols].float()
    tgt_neg = _neg_asl_elem(z_tgt, gamma_neg, clip).sum(dtype=torch.float64)

    pos_probs = torch.sigmoid(z_tgt)
    pos_log = torch.log(pos_probs.clamp(min=1e-8))
    if gamma_pos > 0.0:
        tgt_pos = (-((1.0 - pos_probs).pow(gamma_pos)) * pos_log).sum(dtype=torch.float64)
    else:
        tgt_pos = (-pos_log).sum(dtype=torch.float64)

    return ((all_neg - vis_neg - tgt_neg + tgt_pos) / B).float()


def prepare_cached_dataset(
    data_dir: Path,
    cache_dir: Path,
    min_df: int = 50,
    max_posts: int = 0,
    general_only: bool = False,
    seed: int = 42,
) -> Path:
    """
    Build filtered CSR files on disk in low-RAM chunks (<2GB peak RAM, safe for Kaggle 29GB RAM)
    so multiple DDP worker processes can mmap them with zero memory duplication.
    """
    tag_key = f"df{min_df}_gen{int(general_only)}_max{max_posts}_s{seed}"
    sub = cache_dir / f"_cache_{tag_key}"
    meta_file = sub / "meta.json"
    if meta_file.exists() and (sub / "compact_tags.npy").exists() and (sub / "compact_off.npy").exists():
        return sub

    t0 = time.time()
    sub.mkdir(parents=True, exist_ok=True)
    offsets = np.load(data_dir / "offsets.npy", mmap_mode="r")
    tag_ids = np.load(data_dir / "tag_ids.npy", mmap_mode="r")

    kept_tags, kept_cats, kept_cnts = [], [], []
    old_to_new = {}
    with open(data_dir / "vocab.tsv", "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            tid_s, tag, cat, cnt_s = line.rstrip("\n").split("\t")
            cnt = int(cnt_s)
            if cat in ("artist", "meta") or cnt < min_df:
                continue
            if general_only and cat != "general":
                continue
            new_id = len(kept_tags)
            old_to_new[int(tid_s)] = new_id
            kept_tags.append(tag)
            kept_cats.append(cat)
            kept_cnts.append(cnt)

    V = len(kept_tags)
    max_old_id = max(old_to_new.keys()) + 1
    lut = np.full(max_old_id, -1, dtype=np.int32)
    for o, n in old_to_new.items():
        lut[o] = n

    N_total = len(offsets) - 1
    if max_posts and max_posts < N_total:
        rng = np.random.default_rng(seed)
        chosen = np.sort(rng.choice(N_total, size=max_posts, replace=False))
        import array
        out_tags = array.array("I")
        out_off = array.array("Q", [0])
        for p_idx in chosen:
            s, e = int(offsets[p_idx]), int(offsets[p_idx + 1])
            raw = tag_ids[s:e]
            raw = raw[raw < max_old_id]
            mapped = lut[raw]
            mapped = mapped[mapped >= 0]
            if len(mapped) >= 4:
                out_tags.extend(mapped.tolist())
                out_off.append(len(out_tags))
        compact_tags = np.frombuffer(out_tags, dtype=np.uint32)
        compact_off = np.frombuffer(out_off, dtype=np.uint64)
    else:
        # Chunked vectorized filtering (processes 1M posts at a time -> <1.5 GB peak RAM!)
        tag_chunks = []
        count_chunks = []
        post_chunk = 1_000_000
        for p_start in range(0, N_total, post_chunk):
            p_end = min(N_total, p_start + post_chunk)
            off_slice = np.asarray(offsets[p_start : p_end + 1], dtype=np.int64)
            t_base = int(off_slice[0])
            t_stop = int(off_slice[-1])
            raw_slice = np.asarray(tag_ids[t_base:t_stop])

            in_range = raw_slice < max_old_id
            mapped = np.full(len(raw_slice), -1, dtype=np.int32)
            mapped[in_range] = lut[raw_slice[in_range]]
            valid_mask = mapped >= 0

            csum = np.empty(len(raw_slice) + 1, dtype=np.int64)
            csum[0] = 0
            np.cumsum(valid_mask, out=csum[1:])
            rel_off = off_slice - t_base
            p_counts = csum[rel_off[1:]] - csum[rel_off[:-1]]
            keep_p = p_counts >= 4

            valid_tags = mapped[valid_mask].astype(np.uint32)
            if not keep_p.all():
                valid_tags = valid_tags[np.repeat(keep_p, p_counts)]
                p_counts = p_counts[keep_p]

            tag_chunks.append(valid_tags)
            count_chunks.append(p_counts)

        compact_tags = np.concatenate(tag_chunks)
        all_counts = np.concatenate(count_chunks)
        compact_off = np.empty(len(all_counts) + 1, dtype=np.uint64)
        compact_off[0] = 0
        np.cumsum(all_counts, out=compact_off[1:])

    n_posts = len(compact_off) - 1
    p_marg = np.clip(np.array(kept_cnts, dtype=np.float64) * 0.75 / max(1, N_total), 1e-6, 0.5)
    init_log_odds = np.clip(np.log(p_marg / (1.0 - p_marg)), -9.0, -1.5).astype(np.float32)

    np.save(sub / "compact_tags.npy", compact_tags)
    np.save(sub / "compact_off.npy", compact_off)
    np.save(sub / "init_log_odds.npy", init_log_odds)
    with open(meta_file, "w", encoding="utf-8") as mf:
        json.dump({"kept_tags": kept_tags, "kept_cats": kept_cats, "n_posts": n_posts, "V": V}, mf)

    print(
        f"Prepared mmap cache in {time.time()-t0:.1f}s: "
        f"V={V:,} tags (df>={min_df}, general_only={general_only}) | {n_posts:,} posts (>=4 tags) | "
        f"{len(compact_tags):,} tag occurrences (avg {len(compact_tags)/n_posts:.1f} tags/post)",
        flush=True,
    )
    return sub


def load_cached_dataset(cache_sub: Path):
    compact_tags = np.load(cache_sub / "compact_tags.npy", mmap_mode="r")
    compact_off = np.load(cache_sub / "compact_off.npy", mmap_mode="r")
    init_log_odds = np.load(cache_sub / "init_log_odds.npy")
    with open(cache_sub / "meta.json", "r", encoding="utf-8") as mf:
        meta = json.load(mf)
    return compact_tags, compact_off, meta["kept_tags"], meta["kept_cats"], init_log_odds


def make_batch_vectorized(
    compact_tags,
    compact_off,
    post_indices,
    mask_ratio: float,
    rng: np.random.Generator,
    pin: bool = False,
    mask_std: float = 0.0,
    mask_min: float = 0.55,
    mask_max: float = 0.95,
):
    B = len(post_indices)
    starts = compact_off[post_indices].astype(np.int64)
    ends = compact_off[post_indices + 1].astype(np.int64)
    lengths = ends - starts

    row_off = np.empty(B + 1, dtype=np.int64)
    row_off[0] = 0
    np.cumsum(lengths, out=row_off[1:])
    M = int(row_off[-1])

    row_ids = np.repeat(np.arange(B, dtype=np.int64), lengths)
    intra = np.arange(M, dtype=np.int64) - np.repeat(row_off[:-1], lengths)
    flat_idx = np.repeat(starts, lengths) + intra
    batch_tags = compact_tags[flat_idx].astype(np.int64)

    if mask_std > 0.0:
        # Per-post truncated normal N(mask_ratio, mask_std^2) on [mask_min, mask_max]
        post_ratios = rng.normal(loc=mask_ratio, scale=mask_std, size=B).astype(np.float32)
        oob = (post_ratios < mask_min) | (post_ratios > mask_max)
        if oob.any():
            post_ratios[oob] = rng.normal(loc=mask_ratio, scale=mask_std, size=int(oob.sum())).astype(np.float32)
            post_ratios = np.clip(post_ratios, mask_min, mask_max)
        thresholds = post_ratios[row_ids]
    else:
        thresholds = mask_ratio

    keep_mask = rng.random(M, dtype=np.float32) >= thresholds
    rand_int = rng.integers(0, 1_000_000_000, size=B)
    vis_pick = row_off[:-1] + (rand_int % lengths)
    msk_pick = row_off[:-1] + ((rand_int % lengths + 1) % lengths)
    keep_mask[vis_pick] = True
    keep_mask[msk_pick] = False

    vis_ids = batch_tags[keep_mask]
    vis_rows = row_ids[keep_mask]
    tgt_cols = batch_tags[~keep_mask]
    tgt_rows = row_ids[~keep_mask]

    vis_counts = np.bincount(vis_rows, minlength=B)
    vis_bag_off = np.empty(B, dtype=np.int64)
    vis_bag_off[0] = 0
    np.cumsum(vis_counts[:-1], out=vis_bag_off[1:])

    tensors = (
        torch.from_numpy(vis_ids),
        torch.from_numpy(vis_bag_off),
        torch.from_numpy(vis_counts.astype(np.float32)),
        torch.from_numpy(vis_rows),
        torch.from_numpy(vis_ids),
        torch.from_numpy(tgt_rows),
        torch.from_numpy(tgt_cols),
    )
    if pin:
        tensors = tuple(t.pin_memory() for t in tensors)
    return tensors


def iter_prefetched_batches(
    compact_tags,
    compact_off,
    indices,
    batch_size: int,
    mask_ratio: float,
    mask_std: float,
    mask_min: float,
    mask_max: float,
    seed: int,
    use_cuda: bool,
):
    q: queue.Queue = queue.Queue(maxsize=8)

    def _producer():
        rng = np.random.default_rng(seed)
        for start in range(0, len(indices) - batch_size + 1, batch_size):
            batch_idx = indices[start : start + batch_size]
            q.put(
                make_batch_vectorized(
                    compact_tags,
                    compact_off,
                    batch_idx,
                    mask_ratio,
                    rng,
                    pin=use_cuda,
                    mask_std=mask_std,
                    mask_min=mask_min,
                    mask_max=mask_max,
                )
            )
        q.put(None)

    th = threading.Thread(target=_producer, daemon=True)
    th.start()
    while True:
        item = q.get()
        if item is None:
            break
        yield item


@torch.no_grad()
def evaluate(raw_model, compact_tags, compact_off, val_indices, mask_ratio, device, amp_dtype, gamma_neg, gamma_pos, clip):
    raw_model.eval()
    rng = np.random.default_rng(12345)
    bs = 512
    total_loss, n_batches = 0.0, 0
    rec10, rec25, prec10 = [], [], []
    examples = []

    for start in range(0, len(val_indices), bs):
        batch_idx = val_indices[start : start + bs]
        vis_ids, bag_off, vis_cnt, vr, vc, tr, tc = make_batch_vectorized(
            compact_tags, compact_off, batch_idx, mask_ratio, rng, pin=False
        )
        vis_ids = vis_ids.to(device, non_blocking=True)
        bag_off = bag_off.to(device, non_blocking=True)
        vis_cnt = vis_cnt.to(device, non_blocking=True)
        vr, vc = vr.to(device, non_blocking=True), vc.to(device, non_blocking=True)
        tr, tc = tr.to(device, non_blocking=True), tc.to(device, non_blocking=True)

        with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
            logits = raw_model(vis_ids, bag_off, vis_cnt)
        loss = asymmetric_loss_masked_fast(logits, tr, tc, vr, vc, gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=clip)
        total_loss += float(loss.item())
        n_batches += 1

        logits_eval = logits.float()
        logits_eval[vr, vc] = -1e9
        top25 = torch.topk(logits_eval, k=25, dim=1).indices.cpu().numpy()

        tr_np, tc_np = tr.cpu().numpy(), tc.cpu().numpy()
        vr_np, vc_np = vr.cpu().numpy(), vc.cpu().numpy()
        tr_splits = np.searchsorted(tr_np, np.arange(len(batch_idx) + 1))
        vr_splits = np.searchsorted(vr_np, np.arange(len(batch_idx) + 1))
        for b in range(len(batch_idx)):
            gt_arr = tc_np[tr_splits[b] : tr_splits[b + 1]]
            if len(gt_arr) == 0:
                continue
            gt = set(gt_arr.tolist())
            preds25 = top25[b].tolist()
            preds10 = preds25[:10]
            h10 = sum(1 for p in preds10 if p in gt)
            h25 = sum(1 for p in preds25 if p in gt)
            rec10.append(h10 / len(gt))
            rec25.append(h25 / len(gt))
            prec10.append(h10 / 10.0)
            if len(examples) < 5 and len(gt) >= 12:
                vis_list = vc_np[vr_splits[b] : vr_splits[b + 1]].tolist()
                examples.append((vis_list, preds10, gt))

    raw_model.train()
    return total_loss / max(1, n_batches), float(np.mean(rec10)), float(np.mean(rec25)), float(np.mean(prec10)), examples


def save_ckpt(path: str, args, kept_tags, kept_cats, raw_model, step: int, val_metrics: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": vars(args),
            "step": step,
            "val_metrics": val_metrics,
            "vocab_tags": kept_tags,
            "vocab_cats": kept_cats,
            "model_state_dict": raw_model.state_dict(),
        },
        path,
    )


def train_worker(rank: int, world_size: int, cache_sub: Path, args):
    is_ddp = world_size > 1
    use_cuda = torch.cuda.is_available() and not args.cpu

    if is_ddp:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(args.master_port))
        backend = "nccl" if use_cuda else "gloo"
        if use_cuda:
            torch.cuda.set_device(rank)
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    device = torch.device(f"cuda:{rank}" if use_cuda else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # Hardware-aware AMP: FP16 + GradScaler on Kaggle T4 (sm_75), BF16 on Ampere+ (sm_80+)
    if use_cuda:
        bf16_ok = torch.cuda.is_bf16_supported() and torch.cuda.get_device_capability(device)[0] >= 8
        amp_dtype = torch.bfloat16 if bf16_ok else torch.float16
        use_scaler = (amp_dtype == torch.float16)
    else:
        amp_dtype = torch.float32
        use_scaler = False
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    compact_tags, compact_off, kept_tags, kept_cats, init_log_odds = load_cached_dataset(cache_sub)
    V = len(kept_tags)
    n_posts = len(compact_off) - 1

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_posts)
    n_val = min(10_000, max(1000, n_posts // 20))
    val_idx = perm[:n_val]
    train_idx_all = perm[n_val:]

    torch.manual_seed(args.seed)
    raw_model = MaskedTagMLP(
        V,
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_blocks=args.num_blocks,
        tie_weights=args.tie_weights,
        init_log_odds=init_log_odds,
    ).to(device)

    if rank == 0:
        n_params = sum(p.numel() for p in raw_model.parameters())
        gpu_name = torch.cuda.get_device_name(device) if use_cuda else "CPU"
        mask_desc = (
            f"N({args.mask_ratio:.2f}, {args.mask_std:.2f}) in [{args.mask_min:.2f}, {args.mask_max:.2f}]"
            if args.mask_std > 0
            else f"fixed {args.mask_ratio:.2f}"
        )
        print(
            f"Model: MaskedTagMLP(V={V:,}, d={args.d_model}, ff={args.d_ff}, blocks={args.num_blocks}, tied={args.tie_weights}) "
            f"| {n_params/1e6:.2f}M params | {world_size}x {gpu_name} | AMP={str(amp_dtype).split('.')[-1]} "
            f"| per-GPU bs={args.batch_size} (global bs={args.batch_size * world_size}) | drop={mask_desc}",
            flush=True,
        )

    model = DDP(raw_model, device_ids=[rank] if use_cuda else None) if is_ddp else raw_model

    decay_params, no_decay_params = [], []
    seen = set()
    for name, p in raw_model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if p.ndim >= 2 and "emb" not in name and "head" not in name:
            decay_params.append(p)
        else:
            no_decay_params.append(p)

    # Scale LR gently by sqrt(world_size) when doubling global batch size on 2 GPUs
    eff_lr = args.lr * (world_size ** 0.5) if is_ddp else args.lr
    opt = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": 1e-2},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=eff_lr,
        fused=use_cuda,
    )

    # Trim train_idx_all so all ranks run the exact same number of steps per epoch
    per_rank_len = (len(train_idx_all) // (args.batch_size * world_size)) * args.batch_size
    steps_per_epoch = max(1, per_rank_len // args.batch_size)
    total_steps = args.epochs * steps_per_epoch

    def lr_lambda(cur_step: int) -> float:
        if cur_step < args.warmup_steps:
            return float(cur_step + 1) / float(max(1, args.warmup_steps))
        prog = (cur_step - args.warmup_steps) / float(max(1, total_steps - args.warmup_steps))
        return 0.05 + 0.95 * 0.5 * (1.0 + np.cos(np.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    t0 = time.time()
    step = 0
    best_val = float("inf")
    examples = []

    for ep in range(1, args.epochs + 1):
        ep_t0 = time.time()
        # Deterministic epoch shuffle across ranks, then shard cleanly across world_size
        ep_rng = np.random.default_rng(args.seed + ep * 9973)
        ep_perm = ep_rng.permutation(train_idx_all)[: per_rank_len * world_size]
        rank_idx = ep_perm[rank::world_size]

        running_loss, running_n = 0.0, 0
        for batch in iter_prefetched_batches(
            compact_tags, compact_off, rank_idx, args.batch_size,
            args.mask_ratio, args.mask_std, args.mask_min, args.mask_max,
            seed=args.seed + ep * 1000 + rank, use_cuda=use_cuda,
        ):
            vis_ids, bag_off, vis_cnt, vr, vc, tr, tc = [t.to(device, non_blocking=True) for t in batch]

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_cuda):
                logits = model(vis_ids, bag_off, vis_cnt)
            loss = asymmetric_loss_masked_fast(
                logits, tr, tc, vr, vc,
                gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos, clip=args.clip,
            )

            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
                opt.step()
            sched.step()

            running_loss += float(loss.item())
            running_n += 1
            step += 1

            if rank == 0 and step % 250 == 0:
                print(
                    f"[ep {ep}/{args.epochs} | step {step}/{total_steps}] "
                    f"train_asl={running_loss/running_n:.3f} | lr={sched.get_last_lr()[0]:.2e} | {time.time()-ep_t0:.1f}s",
                    flush=True,
                )
                running_loss, running_n = 0.0, 0

            if args.eval_every > 0 and step % args.eval_every == 0:
                if is_ddp:
                    dist.barrier()
                if rank == 0:
                    val_loss, r10, r25, p10, examples = evaluate(
                        raw_model, compact_tags, compact_off, val_idx, args.mask_ratio, device, amp_dtype,
                        args.gamma_neg, args.gamma_pos, args.clip,
                    )
                    metrics = {"val_asl": val_loss, "p10": p10, "r10": r10, "r25": r25}
                    print(
                        f"  --> [EVAL step {step}] val_asl={val_loss:.3f} | "
                        f"P@10={p10:.3f} | R@10={r10:.3f} | R@25={r25:.3f}",
                        flush=True,
                    )
                    if val_loss < best_val:
                        best_val = val_loss
                        save_ckpt(args.out_ckpt, args, kept_tags, kept_cats, raw_model, step, metrics)
                if is_ddp:
                    dist.barrier()

        if is_ddp:
            dist.barrier()
        if rank == 0:
            val_loss, r10, r25, p10, examples = evaluate(
                raw_model, compact_tags, compact_off, val_idx, args.mask_ratio, device, amp_dtype,
                args.gamma_neg, args.gamma_pos, args.clip,
            )
            metrics = {"val_asl": val_loss, "p10": p10, "r10": r10, "r25": r25}
            print(
                f"=== Epoch {ep}/{args.epochs} DONE ({time.time()-ep_t0:.1f}s) | "
                f"val_asl={val_loss:.3f} | P@10={p10:.3f} | R@10={r10:.3f} | R@25={r25:.3f} ===",
                flush=True,
            )
            save_ckpt(args.out_ckpt, args, kept_tags, kept_cats, raw_model, step, metrics)
        if is_ddp:
            dist.barrier()

    if rank == 0:
        print("\n--- Sample Validation Predictions (25% visible -> Top-10 predicted masked tags) ---")
        for i, (vis_list, preds10, gt) in enumerate(examples):
            vis_str = ", ".join(kept_tags[x] for x in vis_list)
            pred_str = ", ".join(f"{kept_tags[x]}{'✓' if x in gt else ''}" for x in preds10)
            gt_str = ", ".join(kept_tags[x] for x in list(gt)[:12])
            print(f"Example {i+1}:")
            print(f"  VISIBLE (25%) : {vis_str}")
            print(f"  TOP-10 PRED   : {pred_str}")
            print(f"  TRUE MASKED   : {gt_str} ({len(gt)} total)")
        print(f"\nSaved checkpoint to {args.out_ckpt} (total {time.time()-t0:.1f}s).", flush=True)

    if is_ddp:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="/workspace/danbooru2026_tags_uint32")
    parser.add_argument("--cache-dir", type=str, default="", help="Defaults to --data-dir")
    parser.add_argument("--out-ckpt", type=str, default="./masked_tag_mlp.pt")
    parser.add_argument("--min-df", type=int, default=50)
    parser.add_argument("--max-posts", type=int, default=0, help="0 = all 11.74M posts")
    parser.add_argument("--general-only", action="store_true")
    parser.add_argument("--mask-ratio", type=float, default=0.75, help="Mean tag drop ratio")
    parser.add_argument("--mask-std", type=float, default=0.07, help="Std of normal drop ratio (0 = fixed)")
    parser.add_argument("--mask-min", type=float, default=0.55, help="Min tag drop ratio")
    parser.add_argument("--mask-max", type=float, default=0.95, help="Max tag drop ratio")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--num-blocks", type=int, default=3)
    parser.add_argument("--tie-weights", action="store_true", default=True)
    parser.add_argument("--no-tie-weights", dest="tie_weights", action="store_false")
    parser.add_argument("--batch-size", type=int, default=1024, help="Per-GPU batch size")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--warmup-steps", type=int, default=400)
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--gamma-neg", type=float, default=4.0)
    parser.add_argument("--gamma-pos", type=float, default=0.0)
    parser.add_argument("--clip", type=float, default=0.05)
    parser.add_argument("--num-gpus", type=int, default=0, help="0 = auto-detect all GPUs")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--master-port", type=int, default=29517)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
    elif os.access(data_dir, os.W_OK):
        cache_dir = data_dir
    else:
        cache_dir = Path("./tag_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

    # Support both `torchrun` and plain `python3 train_masked_tag_mlp.py` (auto mp.spawn!)
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        if rank == 0:
            prepare_cached_dataset(data_dir, cache_dir, args.min_df, args.max_posts, args.general_only, args.seed)
        dist_tmp = cache_dir / f"_cache_df{args.min_df}_gen{int(args.general_only)}_max{args.max_posts}_s{args.seed}"
        while not (dist_tmp / "meta.json").exists():
            time.sleep(0.5)
        train_worker(rank, world_size, dist_tmp, args)
    else:
        cache_sub = prepare_cached_dataset(data_dir, cache_dir, args.min_df, args.max_posts, args.general_only, args.seed)
        if args.cpu:
            world_size = args.num_gpus if args.num_gpus > 0 else 1
        else:
            avail = torch.cuda.device_count()
            world_size = args.num_gpus if args.num_gpus > 0 else max(1, avail)

        if world_size > 1:
            print(f"Auto-spawning DDP across {world_size} processes...", flush=True)
            mp.spawn(train_worker, args=(world_size, cache_sub, args), nprocs=world_size, join=True)
        else:
            train_worker(0, 1, cache_sub, args)


if __name__ == "__main__":
    main()
