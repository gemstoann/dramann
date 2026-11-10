"""
TapeANN v3 indexer (schema 3).

Files written to out_dir:
  centroids.f32 / centroids.f16   list centroids
  hnsw.bin                        HNSW graph over the centroids (search router)
  pq.f32                          residual PQ codebook, M sub-quantizers x 16 x dsub
  lists.bin                       per list: u64 first_block, u32 n_entries, u32 n_primary
  codes.bin                       per list, consecutive 32-entry blocks:
                                    M x 16 bytes of 4-bit codes, then 32 x u32 positions
  vectors.bin                     4 KB pages: 30 raw uint8 vectors, 30 x u32 ids, padding
  v3_meta.json

Each vector is stored once in vectors.bin, grouped by its primary list.
Its code is replicated into up to `replicas` lists (SPANN closure rule with
the RNG rule). Codes encode x - centroid of the list that holds them.

    python -u python/build_v3.py --config configs/v3/sift10m.json
"""

import argparse
import concurrent.futures
import datetime
import json
import os
import shutil
import struct
import time

import faiss
import numba
import numpy as np

DIM = 128
PAGE = 4096
PER_PAGE = 30
BLOCK = 32
NO_POS = 0xFFFFFFFF
HNSW_MAGIC = 0x57534E4845504154  # "TAPEHNSW"

_raw_print = print


def print(*args, **kwargs):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _raw_print(f"[{ts}]", *args, flush=True, **kwargs)


class Cfg:
    def __init__(self, cfg: dict):
        paths = cfg["paths"]
        index = cfg["index"]
        self.base = os.path.expanduser(paths["base"])
        self.query = os.path.expanduser(paths["query"])
        self.gt = os.path.expanduser(paths.get("gt") or "")
        self.out = os.path.expanduser(paths["out_dir"])
        self.n = int(index["n_vectors"])
        self.k = int(index["n_clusters"])
        self.coarse = int(index.get("coarse_clusters", 256))
        self.train_per_centroid = int(index.get("train_per_centroid", 64))
        self.niter = int(index.get("niter", 20))
        self.split_factor = float(index.get("split_factor", 3.0))
        self.pq_m = int(index.get("pq_m", 32))
        self.pq_train = int(index.get("pq_train", 1_000_000))
        self.replicas = int(index.get("replicas", 4))
        self.closure_eps = float(index.get("closure_eps", 0.15))
        self.assign_k = int(index.get("assign_k", 8))
        self.assign_ef = int(index.get("assign_ef", 64))
        self.hnsw_m = int(index.get("hnsw_m", 32))
        self.hnsw_efc = int(index.get("hnsw_ef_construction", 100))
        self.chunk = int(index.get("chunk", 500_000))
        self.parts = int(index.get("spill_parts", 16))
        self.threads = int(index.get("threads", os.cpu_count() or 1))

    def path(self, name: str) -> str:
        return os.path.join(self.out, name)


def open_base(path: str, n: int) -> np.ndarray:
    with open(path, "rb") as f:
        n_file, d = struct.unpack("<II", f.read(8))
    if d != DIM:
        raise SystemExit(f"[!] {path}: dim {d} != {DIM}")
    if n > n_file:
        raise SystemExit(f"[!] n_vectors {n:,} > {n_file:,} in {path}")
    return np.memmap(path, dtype=np.uint8, mode="r", offset=8, shape=(n, DIM))


def sample_runs(base: np.ndarray, count: int, seed: int, run: int = 1024) -> np.ndarray:
    """Random runs of consecutive rows: sequential reads, many images."""
    n = base.shape[0]
    if count >= n:
        return np.ascontiguousarray(base[:])
    n_runs = (count + run - 1) // run
    rng = np.random.default_rng(seed)
    starts = np.sort(rng.choice(n // run, size=n_runs, replace=False)) * run
    out = np.empty((n_runs * run, DIM), dtype=np.uint8)
    for i, s in enumerate(starts):
        out[i * run:(i + 1) * run] = base[s:s + run]
    return out[:count]


def as_f32(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.float32)


def kmeans(x: np.ndarray, k: int, niter: int, seed: int) -> np.ndarray:
    if x.shape[0] <= k:
        return as_f32(x)
    km = faiss.Kmeans(DIM, k, niter=niter, seed=seed, verbose=False,
                      max_points_per_centroid=256)
    km.train(as_f32(x))
    return as_f32(km.centroids)


def flat_assign(centroids: np.ndarray, x: np.ndarray, chunk: int = 1_000_000) -> np.ndarray:
    index = faiss.IndexFlatL2(DIM)
    index.add(centroids)
    out = np.empty(x.shape[0], dtype=np.int64)
    for s in range(0, x.shape[0], chunk):
        _, lab = index.search(as_f32(x[s:s + chunk]), 1)
        out[s:s + chunk] = lab[:, 0]
    return out


def build_hnsw(cfg: Cfg, centroids: np.ndarray) -> faiss.IndexHNSWFlat:
    t0 = time.time()
    index = faiss.IndexHNSWFlat(DIM, cfg.hnsw_m)
    index.hnsw.efConstruction = cfg.hnsw_efc
    index.add(centroids)
    print(f"    HNSW M={cfg.hnsw_m} over {centroids.shape[0]:,} centroids "
          f"in {time.time() - t0:.0f}s")
    return index


def hnsw_assign(index: faiss.IndexHNSWFlat, x: np.ndarray, k: int, ef: int,
                chunk: int = 1_000_000):
    index.hnsw.efSearch = max(ef, k)
    dist = np.empty((x.shape[0], k), dtype=np.float32)
    lab = np.empty((x.shape[0], k), dtype=np.int64)
    for s in range(0, x.shape[0], chunk):
        dist[s:s + chunk], lab[s:s + chunk] = index.search(as_f32(x[s:s + chunk]), k)
    return dist, lab


def train_fine(cfg: Cfg, sample: np.ndarray) -> np.ndarray:
    """Coarse k-means, then per coarse bucket k-means sized by bucket share."""
    t0 = time.time()
    coarse_k = min(cfg.coarse, cfg.k)
    sub = sample[: min(sample.shape[0], coarse_k * 256)]
    print(f"    coarse k-means k={coarse_k:,} on {sub.shape[0]:,} rows")
    coarse = kmeans(sub, coarse_k, cfg.niter, seed=1)
    cl = flat_assign(coarse, sample)
    order = np.argsort(cl, kind="stable")
    bounds = np.searchsorted(cl[order], np.arange(coarse_k + 1))
    ratio = cfg.k / sample.shape[0]
    jobs = []
    for b in range(coarse_k):
        rows = order[bounds[b]:bounds[b + 1]]
        if rows.size == 0:
            continue
        jobs.append((b, rows, max(1, int(round(rows.size * ratio)))))
    print(f"    fine k-means in {len(jobs):,} buckets "
          f"(~{1 / ratio:.0f} train rows per centroid, niter={cfg.niter})")

    faiss.omp_set_num_threads(1)
    blocks = [None] * len(jobs)

    def run(i):
        faiss.omp_set_num_threads(1)
        b, rows, kb = jobs[i]
        blocks[i] = kmeans(sample[rows], kb, cfg.niter, seed=100 + b)

    with concurrent.futures.ThreadPoolExecutor(cfg.threads) as pool:
        for done, _ in enumerate(pool.map(run, range(len(jobs)))):
            if done % 500 == 0:
                print(f"      bucket {done:,} / {len(jobs):,}")
    faiss.omp_set_num_threads(cfg.threads)
    cents = np.vstack(blocks)
    print(f"    {cents.shape[0]:,} fine centroids in {time.time() - t0:.0f}s")
    return cents


def split_big(cfg: Cfg, cents: np.ndarray, sample: np.ndarray) -> np.ndarray:
    """Split lists whose sampled size is over split_factor x mean."""
    index = build_hnsw(cfg, cents)
    _, lab = hnsw_assign(index, sample, 1, cfg.assign_ef)
    lab = lab[:, 0]
    counts = np.bincount(lab, minlength=cents.shape[0])
    mean = sample.shape[0] / cents.shape[0]
    big = np.flatnonzero(counts > cfg.split_factor * mean)
    print(f"    sampled list size: mean {mean:.1f}  p99 {np.percentile(counts, 99):.0f}  "
          f"max {counts.max()}  splitting {big.size:,} lists > {cfg.split_factor}x mean")
    if big.size == 0:
        return cents
    keep = np.ones(cents.shape[0], dtype=bool)
    keep[big] = False
    extra = []
    order = np.argsort(lab, kind="stable")
    bounds = np.searchsorted(lab[order], np.arange(cents.shape[0] + 1))
    for c in big:
        rows = order[bounds[c]:bounds[c + 1]]
        extra.append(kmeans(sample[rows], int(np.ceil(rows.size / mean)), cfg.niter, seed=int(c)))
    return np.vstack([cents[keep]] + extra).astype(np.float32)


def write_hnsw(path: str, index: faiss.IndexHNSWFlat):
    h = index.hnsw
    cum = faiss.vector_to_array(h.cum_nneighbor_per_level).astype(np.int32)
    levels = faiss.vector_to_array(h.levels).astype(np.int32)
    offsets = faiss.vector_to_array(h.offsets).astype(np.uint64)
    neighbors = faiss.vector_to_array(h.neighbors).astype(np.int32)
    n = levels.size
    with open(path, "wb") as f:
        f.write(struct.pack("<QQiiii", HNSW_MAGIC, n, int(h.max_level), int(h.entry_point),
                            cum.size, 0))
        cum.tofile(f)
        levels.tofile(f)
        offsets.tofile(f)
        neighbors.tofile(f)
    print(f"[+] hnsw.bin  ({os.path.getsize(path) / 1e6:.1f} MB, max_level {h.max_level})")


def phase_centroids(cfg: Cfg, base: np.ndarray):
    if os.path.exists(cfg.path("hnsw.faiss")) and os.path.exists(cfg.path("pq.f32")):
        print("[*] Phase 1: reusing centroids, HNSW and PQ")
        index = faiss.read_index(cfg.path("hnsw.faiss"))
        cents = np.fromfile(cfg.path("centroids.f32"), dtype=np.float32).reshape(-1, DIM)
        pq = faiss.ProductQuantizer(DIM, cfg.pq_m, 4)
        faiss.copy_array_to_vector(np.fromfile(cfg.path("pq.f32"), dtype=np.float32).ravel(),
                                   pq.centroids)
        return cents, index, pq

    print(f"\n[*] Phase 1: centroids  K={cfg.k:,}")
    t0 = time.time()
    count = min(cfg.n, cfg.k * cfg.train_per_centroid)
    sample = sample_runs(base, count, seed=42)
    print(f"    sample {sample.shape[0]:,} rows ({time.time() - t0:.0f}s)")
    cents = train_fine(cfg, sample)
    cents = split_big(cfg, cents, sample)
    index = build_hnsw(cfg, cents)

    rng = np.random.default_rng(3)
    rows = rng.choice(sample.shape[0], size=min(cfg.pq_train, sample.shape[0]), replace=False)
    x = as_f32(sample[rows])
    _, lab = hnsw_assign(index, x, 1, cfg.assign_ef)
    resid = x - cents[lab[:, 0]]
    pq = faiss.ProductQuantizer(DIM, cfg.pq_m, 4)
    pq.train(resid)
    del sample

    cents.tofile(cfg.path("centroids.f32"))
    cents.astype(np.float16).tofile(cfg.path("centroids.f16"))
    faiss.vector_to_array(pq.centroids).astype(np.float32).tofile(cfg.path("pq.f32"))
    write_hnsw(cfg.path("hnsw.bin"), index)
    faiss.write_index(index, cfg.path("hnsw.faiss"))
    print(f"[+] {cents.shape[0]:,} centroids, PQ M={cfg.pq_m}x4bit  ({time.time() - t0:.0f}s)")
    return cents, index, pq


@numba.njit(parallel=True, cache=True)
def _closure(cents, dist, lab, replicas, factor):
    n, k = lab.shape
    acc = np.full((n, replicas), -1, dtype=np.int64)
    for i in numba.prange(n):
        acc[i, 0] = lab[i, 0]
        nacc = 1
        limit = dist[i, 0] * factor
        for j in range(1, k):
            if nacc >= replicas or lab[i, j] < 0 or dist[i, j] > limit:
                break
            cj = lab[i, j]
            ok = True
            for a in range(nacc):
                ca = acc[i, a]
                s = 0.0
                for t in range(cents.shape[1]):
                    d = cents[cj, t] - cents[ca, t]
                    s += d * d
                if s < dist[i, j]:
                    ok = False
                    break
            if ok:
                acc[i, nacc] = cj
                nacc += 1
    return acc


def closure(cfg: Cfg, cents: np.ndarray, dist: np.ndarray, lab: np.ndarray) -> np.ndarray:
    """Lists that hold each row: nearest, plus close lists that pass the RNG rule."""
    return _closure(cents, dist, lab, cfg.replicas, (1.0 + cfg.closure_eps) ** 2)


def list_parts(cfg: Cfg, counts: np.ndarray) -> np.ndarray:
    cum = np.cumsum(counts, dtype=np.float64)
    part = np.floor(cum / max(cum[-1], 1) * cfg.parts - 1e-9).astype(np.int64)
    return np.clip(part, 0, cfg.parts - 1)


def _write_assign_prog(spill: str, nxt: int, n_entries: int):
    tmp = os.path.join(spill, "assign.prog.tmp")
    with open(tmp, "w") as f:
        json.dump({"next": int(nxt), "entries": int(n_entries)}, f)
    os.replace(tmp, os.path.join(spill, "assign.prog"))


def phase_assign(cfg: Cfg, base: np.ndarray, cents: np.ndarray, index, pq):
    """One pass over the base: assign, encode, spill by list part."""
    spill = cfg.path("spill")
    if os.path.exists(os.path.join(spill, "assign.done")):
        print("[*] Phase 2: reusing assignment spill")
        return np.load(os.path.join(spill, "part_of_list.npy"))

    print(f"\n[*] Phase 2: assign {cfg.n:,} vectors  k={cfg.assign_k} ef={cfg.assign_ef} "
          f"replicas<={cfg.replicas} eps={cfg.closure_eps}")
    cb = cfg.pq_m // 2
    code_dt = np.dtype([("list", "<u4"), ("row", "<u4"), ("code", "u1", (cb,))])
    vec_dt = np.dtype([("row", "<u4"), ("vec", "u1", (DIM,))])
    prog_path = os.path.join(spill, "assign.prog")
    resume_from = 0
    n_entries = 0
    if (os.path.exists(prog_path)
            and os.path.exists(os.path.join(spill, "part_of_list.npy"))
            and os.path.exists(cfg.path("primary.npy"))):
        with open(prog_path) as f:
            prog = json.load(f)
        resume_from = int(prog.get("next", 0))
        n_entries = int(prog.get("entries", 0))
        if not (0 < resume_from < cfg.n):
            resume_from = 0
            n_entries = 0

    if resume_from:
        print(f"    resuming at {resume_from:,} / {cfg.n:,}  entries so far {n_entries:,}")
        part_of_list = np.load(os.path.join(spill, "part_of_list.npy"))
        primary = np.load(cfg.path("primary.npy"), mmap_mode="r+")
        mode = "ab"
    else:
        shutil.rmtree(spill, ignore_errors=True)
        os.makedirs(spill)
        est = sample_runs(base, min(cfg.n, 2_000_000), seed=9)
        _, lab = hnsw_assign(index, est, 1, cfg.assign_ef)
        part_of_list = list_parts(cfg, np.bincount(lab[:, 0], minlength=cents.shape[0]))
        np.save(os.path.join(spill, "part_of_list.npy"), part_of_list)
        primary = np.lib.format.open_memmap(cfg.path("primary.npy"), mode="w+",
                                            dtype=np.int32, shape=(cfg.n,))
        mode = "wb"

    vfiles = [open(os.path.join(spill, f"vec_{p:03d}.bin"), mode) for p in range(cfg.parts)]
    cfiles = [open(os.path.join(spill, f"code_{p:03d}.bin"), mode) for p in range(cfg.parts)]
    t0 = time.time()
    for s in range(resume_from, cfg.n, cfg.chunk):
        e = min(s + cfg.chunk, cfg.n)
        raw = np.ascontiguousarray(base[s:e])
        x = as_f32(raw)
        dist, lab = hnsw_assign(index, x, cfg.assign_k, cfg.assign_ef)
        acc = closure(cfg, cents, dist, lab)
        primary[s:e] = acc[:, 0]
        rows = np.arange(s, e, dtype=np.uint32)

        vparts = part_of_list[acc[:, 0]]
        order = np.argsort(vparts, kind="stable")
        rec = np.empty(e - s, dtype=vec_dt)
        rec["row"] = rows[order]
        rec["vec"] = raw[order]
        bounds = np.searchsorted(vparts[order], np.arange(cfg.parts + 1))
        for p in range(cfg.parts):
            if bounds[p + 1] > bounds[p]:
                rec[bounds[p]:bounds[p + 1]].tofile(vfiles[p])

        r_idx, a_idx = np.nonzero(acc >= 0)
        lists = acc[r_idx, a_idx]
        codes = pq.compute_codes(np.ascontiguousarray(x[r_idx] - cents[lists]))
        cparts = part_of_list[lists]
        order = np.argsort(cparts, kind="stable")
        crec = np.empty(r_idx.size, dtype=code_dt)
        crec["list"] = lists[order]
        crec["row"] = rows[r_idx[order]]
        crec["code"] = codes[order]
        bounds = np.searchsorted(cparts[order], np.arange(cfg.parts + 1))
        for p in range(cfg.parts):
            if bounds[p + 1] > bounds[p]:
                crec[bounds[p]:bounds[p + 1]].tofile(cfiles[p])
        n_entries += r_idx.size
        for fh in vfiles + cfiles:
            fh.flush()
        primary.flush()
        _write_assign_prog(spill, e, n_entries)

        done = e - resume_from
        if (s // cfg.chunk) % 5 == 0 or e == cfg.n:
            rate = done / max(time.time() - t0, 1e-9)
            print(f"    {e:,} / {cfg.n:,}  replication {n_entries / e:.2f}  "
                  f"ETA {(cfg.n - e) / max(rate, 1e-9) / 60:.1f} min")
    for f in vfiles + cfiles:
        f.close()
    primary.flush()
    with open(os.path.join(spill, "assign.done"), "w") as f:
        json.dump({"entries": int(n_entries)}, f)
    print(f"[+] assignment done in {(time.time() - t0) / 60:.1f} min, "
          f"replication {n_entries / cfg.n:.3f}")
    return part_of_list


def phase_vectors(cfg: Cfg, n_lists: int):
    """vectors.bin pages, grouped by primary list; row -> position map."""
    spill = cfg.path("spill")
    done = os.path.join(spill, "vectors.done")
    if os.path.exists(done):
        print("[*] Phase 3: reusing vectors.bin")
        return
    print("\n[*] Phase 3: vectors.bin")
    t0 = time.time()
    primary = np.load(cfg.path("primary.npy"))
    row2pos = np.lib.format.open_memmap(cfg.path("row2pos.npy"), mode="w+",
                                        dtype=np.uint32, shape=(cfg.n,))
    vec_dt = np.dtype([("row", "<u4"), ("vec", "u1", (DIM,))])
    page_base = 0
    with open(cfg.path("vectors.bin"), "wb") as out:
        for p in range(cfg.parts):
            path = os.path.join(spill, f"vec_{p:03d}.bin")
            rec = np.fromfile(path, dtype=vec_dt)
            if rec.size == 0:
                continue
            rows = rec["row"].astype(np.int64)
            order = np.lexsort((rows, primary[rows]))
            rows = rows[order]
            n_loc = rows.size
            n_pages = (n_loc + PER_PAGE - 1) // PER_PAGE
            row2pos[rows] = (page_base * PER_PAGE + np.arange(n_loc)).astype(np.uint32)
            vecs = np.zeros((n_pages * PER_PAGE, DIM), dtype=np.uint8)
            vecs[:n_loc] = rec["vec"][order]
            ids = np.full(n_pages * PER_PAGE, NO_POS, dtype=np.uint32)
            ids[:n_loc] = rows.astype(np.uint32)
            pages = np.zeros((n_pages, PAGE), dtype=np.uint8)
            pages[:, :PER_PAGE * DIM] = vecs.reshape(n_pages, -1)
            pages[:, PER_PAGE * DIM:PER_PAGE * (DIM + 4)] = ids.view(np.uint8).reshape(n_pages, -1)
            pages.tofile(out)
            page_base += n_pages
            del rec, vecs, pages
            os.remove(path)
            print(f"    part {p + 1}/{cfg.parts}  pages {page_base:,}")
    row2pos.flush()
    with open(done, "w") as f:
        json.dump({"pages": int(page_base)}, f)
    print(f"[+] vectors.bin {page_base * PAGE / 1e9:.2f} GB in {time.time() - t0:.0f}s")


def phase_codes(cfg: Cfg, n_lists: int, part_of_list: np.ndarray):
    """codes.bin: per list, 32-entry blocks of 4-bit codes plus positions."""
    spill = cfg.path("spill")
    done = os.path.join(spill, "codes.done")
    if os.path.exists(done):
        print("[*] Phase 4: reusing codes.bin")
        return
    print("\n[*] Phase 4: codes.bin")
    t0 = time.time()
    m = cfg.pq_m
    cb = m // 2
    bb = m * 16 + BLOCK * 4
    code_dt = np.dtype([("list", "<u4"), ("row", "<u4"), ("code", "u1", (cb,))])
    row2pos = np.load(cfg.path("row2pos.npy"))
    primary = np.load(cfg.path("primary.npy"), mmap_mode="r")
    first_block = np.zeros(n_lists, dtype=np.uint64)
    n_entries = np.zeros(n_lists, dtype=np.uint32)
    n_primary = np.bincount(np.asarray(primary), minlength=n_lists).astype(np.uint32)
    block_base = 0
    with open(cfg.path("codes.bin"), "wb") as out:
        for p in range(cfg.parts):
            path = os.path.join(spill, f"code_{p:03d}.bin")
            rec = np.fromfile(path, dtype=code_dt)
            if rec.size == 0:
                continue
            order = np.argsort(rec["list"], kind="stable")
            rec = rec[order]
            lists = rec["list"].astype(np.int64)
            lo = int(lists[0])
            hi = int(lists[-1]) + 1
            cnt = np.bincount(lists - lo, minlength=hi - lo)
            slots = (cnt + BLOCK - 1) // BLOCK * BLOCK
            slot_base = np.zeros(hi - lo, dtype=np.int64)
            np.cumsum(slots[:-1], out=slot_base[1:])
            start = np.zeros(hi - lo, dtype=np.int64)
            np.cumsum(cnt[:-1], out=start[1:])
            local = np.arange(rec.size) - start[lists - lo]
            slot = slot_base[lists - lo] + local
            total = int(slots.sum())

            raw = rec["code"]
            nib = np.zeros((total, m), dtype=np.uint8)
            nib[slot, 0::2] = raw & 15
            nib[slot, 1::2] = raw >> 4
            pos = np.full(total, NO_POS, dtype=np.uint32)
            pos[slot] = row2pos[rec["row"].astype(np.int64)]

            n_blocks = total // BLOCK
            nib = nib.reshape(n_blocks, 2, 16, m)
            packed = (nib[:, 0] | (nib[:, 1] << 4)).transpose(0, 2, 1)
            blocks = np.empty((n_blocks, bb), dtype=np.uint8)
            blocks[:, :m * 16] = packed.reshape(n_blocks, -1)
            blocks[:, m * 16:] = pos.view(np.uint8).reshape(n_blocks, -1)
            blocks.tofile(out)

            ids = np.arange(lo, hi)
            has = cnt > 0
            first_block[ids[has]] = (block_base + slot_base[has] // BLOCK).astype(np.uint64)
            n_entries[ids[has]] = cnt[has].astype(np.uint32)
            block_base += n_blocks
            del rec, nib, packed, blocks
            os.remove(path)
            print(f"    part {p + 1}/{cfg.parts}  blocks {block_base:,}")
        tail = (-out.tell()) % PAGE
        if tail:
            out.write(b"\x00" * tail)

    table = np.zeros((n_lists, 4), dtype=np.uint32)
    table[:, 0] = (first_block & 0xFFFFFFFF).astype(np.uint32)
    table[:, 1] = (first_block >> np.uint64(32)).astype(np.uint32)
    table[:, 2] = n_entries
    table[:, 3] = n_primary
    table.tofile(cfg.path("lists.bin"))
    with open(done, "w") as f:
        json.dump({"blocks": int(block_base)}, f)
    print(f"[+] codes.bin {block_base * bb / 1e9:.2f} GB, entries {int(n_entries.sum()):,}, "
          f"max list {int(n_entries.max()):,} in {time.time() - t0:.0f}s")


def load_ivecs(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        k = struct.unpack("<i", f.read(4))[0]
    raw = np.memmap(path, dtype=np.int32, mode="r").reshape(-1, 1 + k)
    return np.array(raw[:, 1:], dtype=np.int32)


def phase_queries(cfg: Cfg):
    with open(cfg.query, "rb") as f:
        nq, d = struct.unpack("<II", f.read(8))
    q = np.memmap(cfg.query, dtype=np.uint8, mode="r", offset=8, shape=(nq, d))
    as_f32(q).tofile(cfg.path("test_queries.bin"))
    if cfg.gt:
        gt = load_ivecs(cfg.gt)[:, :100]
        if gt.shape[0] != nq or int(gt[:, :10].max()) >= cfg.n:
            raise SystemExit(f"[!] ground truth {cfg.gt} does not match {nq} queries / {cfg.n:,} base")
        gt.astype(np.uint32).tofile(cfg.path("ground_truth.bin"))
    print(f"[+] test_queries.bin ({nq:,}) and ground_truth.bin")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        raw_cfg = json.load(f)
    cfg = Cfg(raw_cfg)
    os.makedirs(cfg.out, exist_ok=True)
    faiss.omp_set_num_threads(cfg.threads)
    t0 = time.time()
    base = open_base(cfg.base, cfg.n)
    cents, index, pq = phase_centroids(cfg, base)
    part_of_list = phase_assign(cfg, base, cents, index, pq)
    del index
    phase_vectors(cfg, cents.shape[0])
    phase_codes(cfg, cents.shape[0], part_of_list)
    phase_queries(cfg)
    meta = {
        "schema": 3,
        "dim": DIM,
        "n_vectors": cfg.n,
        "n_lists": int(cents.shape[0]),
        "pq_m": cfg.pq_m,
        "pq_bits": 4,
        "block": BLOCK,
        "block_bytes": cfg.pq_m * 16 + BLOCK * 4,
        "page": PAGE,
        "per_page": PER_PAGE,
        "replicas": cfg.replicas,
        "closure_eps": cfg.closure_eps,
        "hnsw_m": cfg.hnsw_m,
        "config": raw_cfg,
    }
    with open(cfg.path("v3_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    shutil.rmtree(cfg.path("spill"), ignore_errors=True)
    print(f"ALL DONE in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
