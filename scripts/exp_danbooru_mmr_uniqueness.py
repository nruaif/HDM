"""
Experiment: Danbooru 2026 Tag Pruning (Implication + MMR) & Combinatorial Image Uniqueness
(Bayesian IDF vs k-NN Embedding Density + Length-Stratified Z-Scoring).

Uses `/workspace/danbooru2026_tags_uint32/` (11.74M posts encoded as uint32 CSR).
Runs purely on CPU (`nice -n 15`) so GPU generation is unaffected.
"""

import argparse
import time
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.decomposition import TruncatedSVD


DATA_DIR = Path("/workspace/danbooru2026_tags_uint32")


def load_data():
    image_ids = np.load(DATA_DIR / "image_ids.npy", mmap_mode="r")
    offsets = np.load(DATA_DIR / "offsets.npy", mmap_mode="r")
    tag_ids = np.load(DATA_DIR / "tag_ids.npy", mmap_mode="r")
    id_to_tag, id_to_cat, id_to_cnt = [], [], []
    with open(DATA_DIR / "vocab.tsv", "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            _, tag, cat, cnt = line.rstrip("\n").split("\t")
            id_to_tag.append(tag)
            id_to_cat.append(cat)
            id_to_cnt.append(int(cnt))
    return image_ids, offsets, tag_ids, id_to_tag, id_to_cat, np.array(id_to_cnt, dtype=np.int64)


def build_csr_subset(offsets, tag_ids, post_indices, max_vocab):
    """Build scipy CSR matrix (len(post_indices) x max_vocab) for a subset of posts."""
    starts = offsets[post_indices]
    ends = offsets[post_indices + 1]
    lengths = ends - starts
    total = int(lengths.sum())

    # Gather slices
    rows_out = np.empty(total, dtype=np.int32)
    cols_out = np.empty(total, dtype=np.int32)
    pos = 0
    for i, (s, e) in enumerate(zip(starts, ends)):
        n = int(e - s)
        if n == 0:
            continue
        rows_out[pos : pos + n] = i
        cols_out[pos : pos + n] = tag_ids[s:e]
        pos += n

    mask = cols_out < max_vocab
    mat = sparse.csr_matrix(
        (np.ones(int(mask.sum()), dtype=np.float32), (rows_out[mask], cols_out[mask])),
        shape=(len(post_indices), max_vocab),
    )
    mat.sum_duplicates()
    mat.data[:] = 1.0
    return mat


def mmr_prune(tids, emb, idf, keep_k, lam=0.55):
    """Order/specificity-aware MMR over a post's tag IDs."""
    if len(tids) <= keep_k:
        return list(tids)
    # Normalize priority weights to [0, 1] using IDF (since Danbooru tags are alphabetical, not ranked)
    w = idf[tids]
    w_norm = (w - w.min()) / (w.max() - w.min() + 1e-8)
    E = emb[tids]  # already L2-normalized: (m, d)
    sim = E @ E.T  # (m, m)

    selected = [int(np.argmax(w_norm))]
    selected_mask = np.zeros(len(tids), dtype=bool)
    selected_mask[selected[0]] = True
    max_sim = sim[:, selected[0]].copy()

    while len(selected) < keep_k:
        scores = lam * w_norm - (1.0 - lam) * max_sim
        scores[selected_mask] = -1e9
        best = int(np.argmax(scores))
        selected.append(best)
        selected_mask[best] = True
        np.maximum(max_sim, sim[:, best], out=max_sim)

    return [int(tids[i]) for i in selected]


def pooled_vec(tids, emb, idf):
    if len(tids) == 0:
        return np.zeros(emb.shape[1], dtype=np.float32)
    w = idf[tids].astype(np.float32)
    v = (emb[tids] * w[:, None]).sum(axis=0)
    n = np.linalg.norm(v)
    return v / (n + 1e-8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--general-only", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    image_ids, offsets, tag_ids, id_to_tag, id_to_cat, id_to_cnt = load_data()
    N_total = len(image_ids)
    print(f"Loaded {N_total:,} posts, {len(id_to_tag):,} vocab tags. (general_only={args.general_only})", flush=True)

    # Global IDF (with min df >= 5 cap for edge-case typo protection)
    valid_vocab = int((id_to_cnt >= 5).sum())
    MAX_VOCAB = min(30000, valid_vocab)  # top 30k tags cover >97% of all tag occurrences
    idf_full = np.log((N_total + 1.0) / (np.maximum(id_to_cnt, 5) + 1.0)).astype(np.float32)
    idf = idf_full[:MAX_VOCAB]

    # Stop-tags: meta tags + ultra-common tags (df > 40% of dataset) + non-general if --general-only
    stop_tags = {
        i for i in range(MAX_VOCAB)
        if id_to_cat[i] == "meta"
        or (args.general_only and id_to_cat[i] != "general")
        or (id_to_cnt[i] / N_total) > 0.40
    }
    print(
        f"Using top {MAX_VOCAB:,} vocab tags (df>=5 vocab={valid_vocab:,}). "
        f"Stop-tags (meta or df>40%): {len(stop_tags)} "
        f"(e.g. {[id_to_tag[i] for i in sorted(stop_tags)[:10]]})",
        flush=True,
    )

    # 1. Sample 400,000 posts to build co-occurrence graph, implication rules, & 128-d PPMI-SVD embeddings
    rng = np.random.default_rng(42)
    sample_idx = np.sort(rng.choice(N_total, size=400_000, replace=False))
    t1 = time.time()
    X = build_csr_subset(offsets, tag_ids, sample_idx, MAX_VOCAB)
    if args.general_only:
        gen_mask = np.array([1.0 if id_to_cat[i] == "general" else 0.0 for i in range(MAX_VOCAB)], dtype=np.float32)
        X = X.multiply(gen_mask).tocsr()
        X.eliminate_zeros()
    print(f"Built 400k x {MAX_VOCAB} CSR matrix ({X.nnz:,} nnz) in {time.time()-t1:.1f}s.", flush=True)

    # Co-occurrence matrix C = X^T X
    t2 = time.time()
    C = (X.T @ X).tocsr()
    diag = C.diagonal().astype(np.float64)
    print(f"Computed co-occurrence matrix ({C.nnz:,} nnz) in {time.time()-t2:.1f}s.", flush=True)

    # Directed implication rules: P(B | A) = C[A, B] / C[A, A] > 0.92 (for tags with count >= 100 in 400k sample)
    implies: dict[int, set[int]] = {}
    C_coo = C.tocoo()
    mask_imp = (C_coo.row != C_coo.col) & (diag[C_coo.row] >= 100) & (C_coo.data / np.maximum(diag[C_coo.row], 1.0) >= 0.92)
    imp_rows = C_coo.row[mask_imp]
    imp_cols = C_coo.col[mask_imp]
    imp_probs = C_coo.data[mask_imp] / diag[imp_rows]
    for a, b in zip(imp_rows, imp_cols):
        if id_to_cnt[b] > id_to_cnt[a]:  # drop broader parent B when narrower child A is present
            implies.setdefault(int(a), set()).add(int(b))

    print(f"Discovered {sum(len(v) for v in implies.values()):,} directed implication rules P(B|A)>=0.92.", flush=True)
    # Print 12 interesting implication examples
    order_imp = np.argsort(-diag[imp_rows])
    shown = 0
    print("Sample discovered implications (child -> parent):")
    for idx in order_imp:
        a, b, p = int(imp_rows[idx]), int(imp_cols[idx]), float(imp_probs[idx])
        if id_to_cnt[b] > id_to_cnt[a] and a not in stop_tags and b not in stop_tags:
            print(f"  {id_to_tag[a]} ({id_to_cat[a]}) -> {id_to_tag[b]} ({id_to_cat[b]})  [P={p:.3f}]")
            shown += 1
            if shown >= 12:
                break

    # PPMI + TruncatedSVD -> 128-d L2-normalized tag embeddings
    t3 = time.time()
    total_co = float(C.sum())
    row_sum = np.asarray(C.sum(axis=1)).ravel() + 1e-8
    col_sum = np.asarray(C.sum(axis=0)).ravel() + 1e-8
    C_ppmi = C.tocoo().astype(np.float32)
    pmi = np.log((C_ppmi.data * total_co) / (row_sum[C_ppmi.row] * col_sum[C_ppmi.col]) + 1e-12)
    pos_mask = pmi > 0
    PPMI = sparse.csr_matrix(
        (pmi[pos_mask].astype(np.float32), (C_ppmi.row[pos_mask], C_ppmi.col[pos_mask])),
        shape=C.shape,
    )
    svd = TruncatedSVD(n_components=128, random_state=42)
    emb = svd.fit_transform(PPMI).astype(np.float32)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    print(f"Trained 128-d PPMI-SVD tag embeddings in {time.time()-t3:.1f}s.", flush=True)

    # =========================================================================
    # PART 1: Tag Pruning Benchmark (50% & 70% reduction) on 10,000 posts
    # =========================================================================
    print("\n=== PART 1: TAG PRUNING BENCHMARK (10,000 posts, >=15 tags) ===", flush=True)
    eval_posts = []
    for p_idx in rng.choice(N_total, size=30_000, replace=False):
        s, e = int(offsets[p_idx]), int(offsets[p_idx + 1])
        tids = tag_ids[s:e]
        tids = tids[tids < MAX_VOCAB]
        if len(tids) >= 15:
            eval_posts.append((int(p_idx), tids))
        if len(eval_posts) >= 10_000:
            break

    for keep_frac in [0.50, 0.30]:
        cos_naive, cos_topidf, cos_mmr = [], [], []
        avg_imp_dropped = []
        for p_idx, tids in eval_posts:
            v_full = pooled_vec(tids, emb, idf)
            k = max(3, int(round(len(tids) * keep_frac)))

            # 1. Naive truncation (first k)
            v_naive = pooled_vec(tids[:k], emb, idf)
            cos_naive.append(float(v_full @ v_naive))

            # 2. Top-IDF only (no diversity)
            top_k = tids[np.argsort(-idf[tids])[:k]]
            v_topidf = pooled_vec(top_k, emb, idf)
            cos_topidf.append(float(v_full @ v_topidf))

            # 3. Structural (stop-tag + implication) + MMR
            tset = set(int(x) for x in tids)
            to_drop = set(stop_tags) & tset
            for a in tset:
                if a in implies:
                    to_drop |= (implies[a] & tset)
            filtered = np.array([x for x in tids if int(x) not in to_drop], dtype=np.int32)
            if len(filtered) < k:
                filtered = tids
            avg_imp_dropped.append(len(tids) - len(filtered))
            pruned = mmr_prune(filtered, emb, idf, keep_k=k, lam=0.55)
            v_mmr = pooled_vec(np.array(pruned, dtype=np.int32), emb, idf)
            cos_mmr.append(float(v_full @ v_mmr))

        print(
            f"Keep {int(keep_frac*100)}% (prune {int((1-keep_frac)*100)}%): "
            f"Naive={np.mean(cos_naive):.4f} | Top-IDF={np.mean(cos_topidf):.4f} | "
            f"Implication+MMR={np.mean(cos_mmr):.4f}  (structurally dropped {np.mean(avg_imp_dropped):.1f} tags/post avg)",
            flush=True,
        )

    # Print 3 concrete pruning examples (keep 35%)
    print("\n--- Concrete Pruning Examples (Keep 35%) ---")
    for p_idx, tids in eval_posts[:3]:
        tset = set(int(x) for x in tids)
        to_drop = set(stop_tags) & tset
        for a in tset:
            if a in implies:
                to_drop |= (implies[a] & tset)
        filtered = np.array([x for x in tids if int(x) not in to_drop], dtype=np.int32)
        k = max(4, int(round(len(tids) * 0.35)))
        pruned = mmr_prune(filtered if len(filtered) >= k else tids, emb, idf, keep_k=k, lam=0.55)
        sim_score = float(pooled_vec(tids, emb, idf) @ pooled_vec(np.array(pruned, dtype=np.int32), emb, idf))
        print(f"Post id={int(image_ids[p_idx])} ({len(tids)} -> {len(pruned)} tags, cos={sim_score:.3f}):")
        print(f"  FULL   : {', '.join(id_to_tag[int(x)] for x in tids)}")
        print(f"  PRUNED : {', '.join(id_to_tag[int(x)] for x in pruned)}")

    # =========================================================================
    # PART 2: Image Uniqueness & Length-Bias Audit on 50,000 posts
    # =========================================================================
    print("\n=== PART 2: UNIQUENESS & LENGTH-BIAS AUDIT (50,000 posts) ===", flush=True)
    t4 = time.time()
    u_posts = rng.choice(N_total, size=50_000, replace=False)
    lengths = np.zeros(len(u_posts), dtype=np.int32)
    raw_sum = np.zeros(len(u_posts), dtype=np.float32)
    raw_mean = np.zeros(len(u_posts), dtype=np.float32)
    vecs = np.zeros((len(u_posts), 128), dtype=np.float32)

    for i, p_idx in enumerate(u_posts):
        s, e = int(offsets[p_idx]), int(offsets[p_idx + 1])
        tids = tag_ids[s:e]
        tids = tids[tids < MAX_VOCAB]
        # Filter out stop-tags for uniqueness
        tids = np.array([x for x in tids if int(x) not in stop_tags], dtype=np.int32)
        if len(tids) == 0:
            tids = tag_ids[s:e]
            tids = tids[tids < MAX_VOCAB]
        lengths[i] = len(tids)
        if len(tids) > 0:
            vals = idf[tids]
            raw_sum[i] = float(vals.sum())
            raw_mean[i] = float(vals.mean())
            vecs[i] = pooled_vec(tids, emb, idf)

    valid = lengths > 0
    u_posts, lengths, raw_sum, raw_mean, vecs = u_posts[valid], lengths[valid], raw_sum[valid], raw_mean[valid], vecs[valid]

    # Approach A: Bayesian-smoothed IDF
    m_prior = float(np.median(lengths))
    mu_global = float(raw_sum.sum() / lengths.sum())
    bayes_u = (raw_sum + m_prior * mu_global) / (lengths + m_prior)

    # Approach B: Exact k-NN (k=20) cosine distance via chunked BLAS matmul (50k x 128 takes ~4s on 48 cores!)
    k_nn = 20
    knn_dist = np.zeros(len(u_posts), dtype=np.float32)
    chunk = 2000
    for start in range(0, len(u_posts), chunk):
        end = min(start + chunk, len(u_posts))
        sims = vecs[start:end] @ vecs.T  # (chunk, 50k)
        # Exclude self
        for r in range(end - start):
            sims[r, start + r] = -2.0
        topk = np.partition(sims, -k_nn, axis=1)[:, -k_nn:]
        knn_dist[start:end] = 1.0 - topk.mean(axis=1)

    # Length-stratified Z-scoring on Approach B (bucket = clip(len, 1, 60))
    knn_z = np.zeros_like(knn_dist)
    buckets = np.minimum(lengths // 5, 16)  # 5-tag buckets (0-4, 5-9, ..., 80+)
    for b in np.unique(buckets):
        idx_b = np.where(buckets == b)[0]
        mu_b = knn_dist[idx_b].mean()
        sd_b = knn_dist[idx_b].std() + 1e-6
        knn_z[idx_b] = (knn_dist[idx_b] - mu_b) / sd_b

    print(f"Computed all 5 uniqueness metrics on {len(u_posts):,} posts in {time.time()-t4:.1f}s.", flush=True)
    print("\nLength-bias audit (Spearman rho with tag count |T|; closer to 0.00 = unbiased):")
    for name, arr in [
        ("Raw IDF Sum", raw_sum),
        ("Raw IDF Mean", raw_mean),
        ("Approach A: Bayesian-Smoothed IDF", bayes_u),
        ("Approach B: k-NN Cosine Dist (raw)", knn_dist),
        ("Approach B + Length-Stratified Z-Score", knn_z),
    ]:
        rho, _ = spearmanr(lengths, arr)
        print(f"  {name:42s} | rho(|T|, score) = {rho:+.4f}")

    # Show Top-10 Most Unique vs Top-5 Most Generic by Approach B + Z-Score
    def show_posts(title, idxs):
        print(f"\n--- {title} ---")
        for i in idxs:
            p_idx = int(u_posts[i])
            s, e = int(offsets[p_idx]), int(offsets[p_idx + 1])
            raw_tids = [int(x) for x in tag_ids[s:e] if int(x) < MAX_VOCAB and (not args.general_only or id_to_cat[int(x)] == "general")]
            tags = [id_to_tag[x] for x in raw_tids[:16]]
            print(
                f"  id={int(image_ids[p_idx]):8d} | len={lengths[i]:2d} | z={knn_z[i]:+.2f} | "
                f"knn_d={knn_dist[i]:.3f} | bayes={bayes_u[i]:.2f} | tags: {', '.join(tags)}"
            )

    order_z = np.argsort(-knn_z)
    show_posts("TOP 10 MOST UNIQUE VISUAL SCENES (Approach B + Length-Stratified Z)", order_z[:10])
    show_posts("TOP 5 MOST GENERIC VISUAL SCENES (Approach B + Length-Stratified Z)", order_z[-5:][::-1])
    print(f"\nTotal experiment runtime: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
