# DRAMANN

Supplemental code for the ICDE 2027 paper *DRAMANN: Billion-Scale Vector Search with Under One Byte of DRAM per Vector*.

This branch is the artifact: the searcher, the builder, the configs used in the paper, a sweep script, and the CSV/JSONL files that generate the figures. It is not a dump of the lab machine.

Source comments may still say TapeANN. The paper name is DRAMANN.

## What is here

| Path | Role |
| --- | --- |
| `engine/cpp/search_v3.cpp` | Searcher (~800 lines, `io_uring`, `O_DIRECT`) |
| `engine/python/build_v3.py` | Index builder (Faiss + NumPy) |
| `engine/configs/v3/*.json` | Build knobs for BIGANN 10M / 100M / 1B and two 10M variants |
| `engine/Makefile` | `g++ -O3 -march=native -luring` |
| `engine/bench/sweep_v3.sh` | One-thread `t_max` / `eps` / `R` sweep |
| `engine/requirements.txt` | Python deps for the builder |
| `figures/src/` | Figure script and the inputs used in the paper |

## What is not here (do not add)

- Indexes (`indexes/v3_sift1b` is about 200 GB; DiskANN 1B is about 442 GB)
- BIGANN vectors (`base1b.u8bin` is about 128 GB) and official ground-truth `.ivecs`
- Compiled binaries, `.venv`, logs, `results/`
- The Overleaf / LaTeX paper, internal notes, revision notes
- Old SIFT1M/SIFT10M trees, schema-2 covering search, DiskANN/Starling/SPANN binaries
- Machine-specific drivers that hard-code `/home/...` paths (`run_native.py`, `run_v3_session.sh`)

Edit the four `paths` fields in a config. Do not commit your local absolute paths.

## Dependencies

- Linux, `g++` with AVX2, `liburing`
- Python 3 with `numpy`, `faiss-cpu`, `numba` (see `engine/requirements.txt`)
- BIGANN from the [Texmex corpus](http://corpus-texmex.irisa.fr/) (`base*.u8bin`, `query.u8bin`, `idx_*M.ivecs`)

## Build the searcher

```bash
cd engine
make
```

## Build an index (10M example)

```bash
cd engine
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# point configs/v3/sift10m.json at your BIGANN files, then:
python -u python/build_v3.py --config configs/v3/sift10m.json
```

1B uses `configs/v3/sift1b.json` and needs hundreds of GB of disk. The paper's 1B index is not hosted with the code.

## Search

```bash
./engine/cpp/search_v3 --index-dir engine/indexes/v3_sift10m \
  --n-queries 2000 --t-max 128 --t-min 8 --eps 0.4 --rerank 200 --threads 1
```

The 0.95 Recall@10 operating point in the paper at BIGANN-1B is `t_max=128`, `eps=0.4`, `rerank=200`, eight threads.

```bash
engine/bench/sweep_v3.sh engine/indexes/v3_sift10m /tmp/sweep_10m.txt 2000
```

## Figures

```bash
python3 figures/src/make_figures.py   # writes figures/fig_*.pdf
```

The numbers in the paper come from `figures/src/data/`. Re-running the binaries needs the indexes, which you build or request from the authors.

## Hardware in the paper

One machine: Intel Core Ultra 9 285K, 62 GiB DRAM, WD Blue SN5100 2 TB QLC NVMe (no on-board DRAM). Search threads 0–7. That is a measurement setting, not a requirement to compile the searcher.
