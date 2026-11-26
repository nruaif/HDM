"""
Encode ALL Danbooru 2026 post tags into compact uint32 CSR arrays + vocab decoder.

Outputs in `--out-dir`:
  - image_ids.npy   : uint32 [N]     Danbooru post IDs
  - offsets.npy     : uint32 [N + 1] CSR offsets (tags for post i = tag_ids[offsets[i]:offsets[i+1]])
  - tag_ids.npy     : uint32 [M]     Flat uint32 tag IDs (frequency-sorted: 0 = most frequent tag)
  - vocab.tsv       : TSV mapping uint32 tag_id -> (tag, category, count)
  - decode_tags.py  : Standalone zero-dependency (numpy only) helper to decode tags by index or image_id

Also packs everything into a single high-compression archive `danbooru2026_tags_uint32.tar.zst`
(or `.tar.gz` if zstd CLI is unavailable) for fast download.
"""

import argparse
import array
import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


TAG_COLS = [
    ("tag_string_artist", "artist"),
    ("tag_string_character", "character"),
    ("tag_string_copyright", "copyright"),
    ("tag_string_general", "general"),
    ("tag_string_meta", "meta"),
]

DECODER_CODE = '''"""Tiny helper to decode Danbooru 2026 uint32 tags (numpy only)."""
from pathlib import Path
import numpy as np


class DanbooruTags:
    def __init__(self, data_dir: str | Path, mmap: bool = True):
        d = Path(data_dir)
        mode = "r" if mmap else None
        self.image_ids = np.load(d / "image_ids.npy", mmap_mode=mode)
        self.offsets = np.load(d / "offsets.npy", mmap_mode=mode)
        self.tag_ids = np.load(d / "tag_ids.npy", mmap_mode=mode)
        self.id_to_tag: list[str] = []
        self.id_to_cat: list[str] = []
        with open(d / "vocab.tsv", "r", encoding="utf-8") as f:
            header = f.readline()
            for line in f:
                _, tag, cat, _ = line.rstrip("\\n").split("\\t")
                self.id_to_tag.append(tag)
                self.id_to_cat.append(cat)
        self.tag_to_id = {t: i for i, t in enumerate(self.id_to_tag)}
        self._sorted_idx = None

    def __len__(self) -> int:
        return len(self.image_ids)

    def get_tag_ids(self, idx: int) -> np.ndarray:
        return self.tag_ids[self.offsets[idx] : self.offsets[idx + 1]]

    def decode(self, tids) -> list[str]:
        return [self.id_to_tag[int(t)] for t in tids]

    def __getitem__(self, idx: int) -> tuple[int, list[str]]:
        return int(self.image_ids[idx]), self.decode(self.get_tag_ids(idx))

    def get_by_image_id(self, image_id: int) -> list[str]:
        if self._sorted_idx is None:
            self._sorted_idx = np.argsort(self.image_ids)
            self._sorted_ids = self.image_ids[self._sorted_idx]
        pos = int(np.searchsorted(self._sorted_ids, image_id))
        if pos >= len(self._sorted_ids) or int(self._sorted_ids[pos]) != image_id:
            raise KeyError(f"image_id {image_id} not found")
        return self.decode(self.get_tag_ids(int(self._sorted_idx[pos])))


if __name__ == "__main__":
    import sys
    ds = DanbooruTags(sys.argv[1] if len(sys.argv) > 1 else ".")
    print(f"Loaded {len(ds):,} posts, {len(ds.id_to_tag):,} unique tags, {len(ds.tag_ids):,} tag occurrences.")
    img_id, tags = ds[0]
    print(f"Sample[0] id={img_id}: {tags[:15]}")
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=str, default="/workspace/posts-snapshot.parquet")
    parser.add_argument("--out-dir", type=str, default="/workspace/danbooru2026_tags_uint32")
    parser.add_argument("--archive", type=str, default="/workspace/danbooru2026_tags_uint32.tar.zst")
    args = parser.parse_args()

    t0 = time.time()
    pq_path = Path(args.parquet)
    if not pq_path.exists():
        from huggingface_hub import hf_hub_download
        print("Downloading posts-snapshot.parquet from nyanko-devs/danbooru2026...", flush=True)
        dl = hf_hub_download(
            repo_id="nyanko-devs/danbooru2026",
            filename="metadata/posts-snapshot.parquet",
            repo_type="dataset",
            local_dir=str(pq_path.parent / "_dl_danbooru"),
        )
        pq_path = Path(dl)
        print(f"Downloaded in {time.time() - t0:.1f}s -> {pq_path}", flush=True)

    pf = pq.ParquetFile(pq_path)
    avail_cols = set(pf.schema_arrow.names)
    active_tag_cols = [(c, cat) for c, cat in TAG_COLS if c in avail_cols]
    read_cols = ["id"] + [c for c, _ in active_tag_cols]
    print(f"Parquet rows: {pf.metadata.num_rows:,} across {pf.num_row_groups} row groups. Columns: {read_cols}", flush=True)

    # Pass 1: build frequency-sorted vocab + tag categories
    t1 = time.time()
    counts = Counter()
    tag_cat: dict[str, str] = {}
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg, columns=[c for c, _ in active_tag_cols])
        for col_name, cat in active_tag_cols:
            col = tbl[col_name].to_pylist()
            for s in col:
                if not s:
                    continue
                for tok in s.split():
                    counts[tok] += 1
                    if tok not in tag_cat:
                        tag_cat[tok] = cat
        if (rg + 1) % 25 == 0 or rg == pf.num_row_groups - 1:
            print(f"[Pass 1] RG {rg+1}/{pf.num_row_groups} | unique tags: {len(counts):,} | {time.time()-t1:.1f}s", flush=True)

    # Sort by frequency descending so small uint32 IDs repeat most -> best compression
    vocab_sorted = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    tag_to_id = {tag: idx for idx, (tag, _) in enumerate(vocab_sorted)}
    print(f"Vocab built: {len(vocab_sorted):,} unique tags in {time.time()-t1:.1f}s.", flush=True)

    # Pass 2: encode all posts into flat uint32 arrays
    t2 = time.time()
    image_ids = array.array("I")
    offsets = array.array("Q", [0])
    tag_ids = array.array("I")

    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg, columns=read_cols)
        ids_col = tbl["id"].to_pylist()
        tag_lists = [tbl[c].to_pylist() for c, _ in active_tag_cols]
        n = len(ids_col)
        for i in range(n):
            image_ids.append(int(ids_col[i]))
            for tl in tag_lists:
                s = tl[i]
                if s:
                    for tok in s.split():
                        tag_ids.append(tag_to_id[tok])
            offsets.append(len(tag_ids))
        if (rg + 1) % 25 == 0 or rg == pf.num_row_groups - 1:
            print(f"[Pass 2] RG {rg+1}/{pf.num_row_groups} | posts={len(image_ids):,} | total_tags={len(tag_ids):,} | {time.time()-t2:.1f}s", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "image_ids.npy", np.frombuffer(image_ids, dtype=np.uint32))
    # Use uint32 offsets if total_tags < 2^32 (saves 50% size)
    off_arr = np.frombuffer(offsets, dtype=np.uint64)
    if int(off_arr[-1]) < (1 << 32):
        np.save(out_dir / "offsets.npy", off_arr.astype(np.uint32))
    else:
        np.save(out_dir / "offsets.npy", off_arr)
    np.save(out_dir / "tag_ids.npy", np.frombuffer(tag_ids, dtype=np.uint32))

    with open(out_dir / "vocab.tsv", "w", encoding="utf-8") as vf:
        vf.write("tag_id\ttag\tcategory\tcount\n")
        for tid, (tag, cnt) in enumerate(vocab_sorted):
            vf.write(f"{tid}\t{tag}\t{tag_cat[tag]}\t{cnt}\n")

    (out_dir / "decode_tags.py").write_text(DECODER_CODE, encoding="utf-8")

    # Quick self-test
    import sys
    sys.path.insert(0, str(out_dir))
    from decode_tags import DanbooruTags
    ds = DanbooruTags(out_dir)
    sample_id, sample_tags = ds[0]
    print(f"Self-test OK! posts={len(ds):,}, vocab={len(ds.id_to_tag):,}, total_tags={len(ds.tag_ids):,}", flush=True)
    print(f"Sample[0] id={sample_id}: {sample_tags[:12]}", flush=True)

    # Compress with zstd if available, else gzip
    archive = Path(args.archive)
    if subprocess.run(["which", "zstd"], capture_output=True).returncode == 0:
        cmd = ["tar", "-C", str(out_dir.parent), "-I", "zstd -T4 -9", "-cf", str(archive), out_dir.name]
    else:
        archive = archive.with_suffix(".gz")
        cmd = ["tar", "-C", str(out_dir.parent), "-czf", str(archive), out_dir.name]
    print(f"Compressing to {archive}...", flush=True)
    subprocess.run(cmd, check=True)
    size_mb = archive.stat().st_size / (1024 * 1024)
    print(f"ALL DONE in {time.time()-t0:.1f}s! Archive: {archive} ({size_mb:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
