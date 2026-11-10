"""Figures for the DRAMANN paper. Run: uv run --with pandas --with matplotlib python make_figures.py
Inputs: data/*.csv. Outputs: ../fig_*.pdf
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE.parent
SYS = "DRAMANN"
# Categorical slots 1-4 (validated light-mode palette); identity also carried by marker shape.
STYLE = {
    SYS:        dict(color="#2a78d6", marker="o"),
    "DiskANN":  dict(color="#eb6834", marker="s"),
    "Starling": dict(color="#1baf7a", marker="^"),
    "SPANN":    dict(color="#eda100", marker="D"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL, FULL = 3.5, 7.16
plt.rcParams.update({
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5, "legend.fontsize": 6.8,
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "font.family": "serif",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.5,
    "lines.markersize": 4.5, "legend.frameon": False, "pdf.fonttype": 42,
})
GIB_TO_GB = 2**30 / 1e9
ALPHA_MS, BETA_MS_PER_MB = 2.63e-3, 0.185  # fitted below; asserted


def rename(s):
    return SYS if str(s).startswith("TapeANN") else str(s).split(" ")[0]


def mva(Z, D, N):
    Q, xs, rs = 0.0, [], []
    for n in range(1, N + 1):
        R = D * (1 + Q); X = n / (Z + R); Q = X * R
        xs.append(X * 1000); rs.append(Z + R)
    return np.array(xs), np.array(rs)


def envelope(df, x, y):
    """Upper-right frontier: points not dominated in (x higher, y higher)."""
    d = df.sort_values(x, ascending=False)
    best, keep = -1, []
    for i, r in d.iterrows():
        if r[y] > best:
            keep.append(i); best = r[y]
    return df.loc[keep].sort_values(x)


sweep = pd.read_csv(HERE / "data/bigann1b_8thread_protocol_c.csv")
sweep["sys"] = sweep.system.map(rename)
t = sweep[sweep.sys == SYS].copy()
dk = sweep[sweep.sys == "DiskANN"].copy()

# ---- model fit (no intercept): 1/QPS [ms] = alpha*reads + beta*MB
X = np.c_[t.reads_per_query, t.bytes_per_query / 1e6]
y = 1000 / t.qps.values
coef, *_ = np.linalg.lstsq(X, y, rcond=None)
pred = X @ coef
r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
print(f"alpha={coef[0]*1e3:.2f}us beta={coef[1]:.3f}ms/MB R2={r2:.4f} maxrel={np.max(np.abs(pred-y)/y):.3f}")
assert abs(coef[0] - ALPHA_MS) < 5e-5 and abs(coef[1] - BETA_MS_PER_MB) < 5e-4

# ---- Fig: 1B recall vs QPS and vs MB/query
fig, axs = plt.subplots(1, 2, figsize=(COL, 1.75), constrained_layout=True)
for name, d in [(SYS, t), ("DiskANN", dk)]:
    st = STYLE[name]
    if name == SYS:
        axs[0].scatter(d.qps, d.recall10, s=9, color=st["color"], alpha=0.35, linewidths=0, marker=st["marker"])
        e = envelope(d, "qps", "recall10")
        axs[0].plot(e.qps, e.recall10, color=st["color"], marker=st["marker"], label=name)
        e2 = envelope(d.assign(neg=-d.bytes_per_query), "neg", "recall10")
        axs[1].scatter(d.bytes_per_query / 1e6, d.recall10, s=9, color=st["color"], alpha=0.35, linewidths=0)
        axs[1].plot(e2.bytes_per_query / 1e6, e2.recall10, color=st["color"], marker=st["marker"])
    else:
        d = d.sort_values("qps")
        axs[0].plot(d.qps, d.recall10, color=st["color"], marker=st["marker"], label=name)
        axs[1].plot(d.bytes_per_query / 1e6, d.recall10, color=st["color"], marker=st["marker"])
for a in axs:
    a.set_ylim(0.80, 1.0); a.axhline(0.95, color=MUTED, lw=0.7, ls=":")
axs[0].set_xlabel("QPS (8 threads)"); axs[0].set_ylabel("Recall@10")
axs[1].set_xscale("log"); axs[1].set_xlabel("MB read per query")
axs[0].set_xlim(0, 2200)
h, l = axs[0].get_legend_handles_labels(); axs[1].legend(h, l, loc="lower right")
fig.savefig(OUT / "fig_1b_recall_qps.pdf"); plt.close(fig)

# ---- Fig: model validation (parity, QPS vs threads 1..24, peak RSS vs QPS)
ts1 = pd.read_csv(HERE / "data/thread_scale_1b_20261107.csv")
g1 = ts1.groupby(["system", "config", "threads"]).agg(qps=("qps", "mean"), qmin=("qps", "min"),
                                                       qmax=("qps", "max")).reset_index()
D_S, Z_S = 1000 / 373.5, 4.11 - 1000 / 373.5     # first SSD state (Section VI-C)
cc = np.arange(1, 25)
Xm, Rm = mva(Z_S, D_S, 24)
DISK_LIMIT = 1e6 / (145.28 * 6.29)                # fio: 159k 4 KB reads/s on the DiskANN file

fig, axs = plt.subplots(1, 3, figsize=(FULL, 2.15), constrained_layout=True)
a = axs[0]
a.plot([0, 1600], [0, 1600], color=MUTED, lw=0.8, ls="--")
a.scatter(1000 / pred, t.qps, s=12, **{k: v for k, v in STYLE[SYS].items()}, linewidths=0)
a.set_xlim(0, 1600); a.set_ylim(0, 1600)
a.set_xlabel(r"predicted QPS  $1/(\alpha n_r+\beta B)$"); a.set_ylabel("measured QPS")
a.text(80, 1450, f"36 operating points\n$R^2$ = {r2:.3f}", color=INK, va="top")

a = axs[1]
d = g1[(g1.system == "DRAMANN")]
a.plot(cc, Xm, color=STYLE[SYS]["color"], lw=1.2, label=f"{SYS} model")
a.errorbar(d.threads, d.qps, yerr=[d.qps - d.qmin, d.qmax - d.qps], ls="none", capsize=2, **STYLE[SYS],
           label=f"{SYS} measured")
d = g1[(g1.system == "DiskANN") & (g1.config == "W4")]
a.plot(d.threads, d.qps, ls="none", **STYLE["DiskANN"], label="DiskANN $w$=4")
d = g1[(g1.system == "DiskANN") & (g1.config == "W16_cache1M")]
a.plot(d.threads, d.qps, ls="none", marker="s", mfc="none", color=STYLE["DiskANN"]["color"],
       label="DiskANN $w$=16, cache")
a.axhline(DISK_LIMIT, color=STYLE["DiskANN"]["color"], lw=1.0, ls="--")
a.text(24, DISK_LIMIT + 25, "DiskANN limit (fio)", ha="right", color=MUTED, fontsize=6.3)
a.set_ylim(0, 1250); a.set_xlim(0, 25); a.set_xlabel("threads $c$"); a.set_ylabel("QPS")
a.set_xticks([1, 4, 8, 12, 16, 24])
h, l = a.get_legend_handles_labels()
fig.legend(h, l, loc="outside upper center", ncol=5)

a = axs[2]
fr = pd.read_csv(HERE / "data/frontier_1b.csv")
for _, r in fr.iterrows():
    st = STYLE[SYS] if r.system == SYS else STYLE["DiskANN"]
    filled = r.system == SYS or r.label.startswith("PQ32")
    a.plot(r.rss_gib, r.qps, ls="none", marker=st["marker"], color=st["color"],
           mfc=st["color"] if filled else "none")
    if r.system == SYS:
        a.errorbar(r.rss_gib, r.qps, yerr=[[r.qps - 371], [438 - r.qps]], color=st["color"], capsize=2, lw=1)
    a.annotate(r.label, (r.rss_gib, r.qps), textcoords="offset points", xytext=(4, 3), fontsize=6.0, color=INK)
a.set_xscale("log"); a.set_xlim(0.4, 80); a.set_ylim(-40, 1000)
a.set_xlabel("peak RSS (GiB, log)"); a.set_ylabel("QPS, $c$=8, Recall@10 $\\geq$ 0.95")
fig.savefig(OUT / "fig_model.pdf"); plt.close(fig)

# ---- Fig: 10M four systems
ts = pd.read_csv(HERE / "data/thread_scale.csv")
ts["sys"] = ts.system.map(rename)
ts["gbs"] = ts.ssd_gb_s * GIB_TO_GB
g = ts.groupby(["sys", "threads"]).agg(qps=("qps", "mean"), qsd=("qps", "std"), lat=("lat_mean_ms", "mean"),
                                       gbs=("gbs", "mean"), gsd=("gbs", "std")).reset_index()
fig, axs = plt.subplots(1, 3, figsize=(FULL, 1.95), constrained_layout=True)
for name in [SYS, "DiskANN", "Starling", "SPANN"]:
    d = g[g.sys == name].sort_values("threads"); st = STYLE[name]
    axs[0].errorbar(d.threads, d.qps, yerr=d.qsd.fillna(0), capsize=2, label=name, **st)
    axs[1].plot(d.threads, d.lat, **st)
    axs[2].errorbar(d.threads, d.gbs, yerr=d.gsd.fillna(0), capsize=2, **st)
axs[0].set_ylabel("QPS"); axs[1].set_ylabel("mean latency (ms)"); axs[2].set_ylabel("SSD read (GB/s, iostat)")
for a in axs:
    a.set_xlabel("threads $c$"); a.set_xticks([1, 2, 4, 8]); a.set_ylim(bottom=0)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="outside upper center", ncol=4)
fig.savefig(OUT / "fig_10m_fourway.pdf"); plt.close(fig)

# ---- Fig: 10M code variants, recall vs MB/query
v = pd.read_csv(HERE / "data/variants_10m_summary.csv")
lab = {"baseline": ("Baseline ($M$=64, ≤8 copies)", "o"),
       "tight_closure": ("Fewer copies ($M$=64, ≤4)", "s"),
       "narrow_codes": ("Narrow codes ($M$=32, ≤8)", "^")}
cols = ["#2a78d6", "#eb6834", "#1baf7a"]
fig, a = plt.subplots(figsize=(COL, 1.9), constrained_layout=True)
for (key, (name, mk)), colr in zip(lab.items(), cols):
    d = v[v.variant == key].sort_values("t_max")
    a.plot(d.bytes_per_query_mean / 1e6, d.recall_at_10_mean, marker=mk, color=colr, label=name)
    for _, r in d.iterrows():
        a.annotate(f"{int(r.t_max)}", (r.bytes_per_query_mean / 1e6, r.recall_at_10_mean),
                   textcoords="offset points", xytext=(3, -7), fontsize=5.8, color=MUTED)
a.axhline(0.95, color=MUTED, lw=0.7, ls=":")
a.set_xlabel("MB read per query"); a.set_ylabel("Recall@10"); a.legend(loc="lower right")
fig.savefig(OUT / "fig_10m_variants.pdf"); plt.close(fig)
print("ok")
