#!/bin/bash
# Quick v3 parameter sweep (O_DIRECT, 1 thread). Usage: bench/sweep_v3.sh INDEX_DIR OUT_FILE [NQ]
set -u
dir=$1
out=$2
nq=${3:-2000}
bin=$(dirname "$0")/../cpp/search_v3
: > "$out"
for tmax in 32 64 128 256; do
  for eps in 0.2 0.4 1.0; do
    for r in 100 200 400; do
      line=$("$bin" --index-dir "$dir" --n-queries "$nq" --t-max "$tmax" --t-min 8 --eps "$eps" --rerank "$r" \
        | awk -F': ' '/Avg lists/{l=$2} /Avg bytes read/{b=$2} /Avg pages/{p=$2} /Avg latency/{t=$2} /p50/{pp=$2} /Recall@10/{rc=$2} END{printf "lists=%s pages=%s MB=%.2f lat=%s pcts=%s recall=%s", l, p, b/1e6, t, pp, rc}')
      echo "tmax=$tmax eps=$eps R=$r $line" | tee -a "$out"
    done
  done
done
