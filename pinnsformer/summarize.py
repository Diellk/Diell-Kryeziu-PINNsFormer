"""Print results/results.csv as report-ready tables (and a LaTeX version).

    python summarize.py            # markdown tables to stdout
    python summarize.py --latex    # LaTeX tabular rows
"""
import argparse
import os

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument("--csv", default=os.path.join(ROOT, "results", "results.csv"))
p.add_argument("--latex", action="store_true")
args = p.parse_args()

df = pd.read_csv(args.csv)
if "activation" not in df.columns:  # rows written before the activation column existed
    df["activation"] = "wavelet"
df["activation"] = df["activation"].fillna("wavelet")


def label(r):
    if r.model == "pinnsformer":
        s = f"pinnsformer (grid {r.train_grid}, k={r.k}"
        if r.activation not in ("wavelet", "default"):
            s += f", act={r.activation}"
        return s + ")"
    s = f"{r.model} (grid {r.train_grid})"
    if "nbconfig" in str(r.run):
        s += " [notebook config]"
    return s


df["config"] = df.apply(label, axis=1)

paper = {  # Table 1 / Table 2 of the paper, for side-by-side comparison
    ("convection", "pinn"): (0.016, 0.778, 0.840), ("convection", "qres"): (0.015, 0.746, 0.816),
    ("convection", "fls"): (0.012, 0.674, 0.771), ("convection", "pinnsformer"): (3.7e-5, 0.023, 0.027),
    ("1d_reaction", "pinn"): (0.199, 0.982, 0.981), ("1d_reaction", "qres"): (0.199, 0.979, 0.977),
    ("1d_reaction", "fls"): (0.2, 0.984, 0.985), ("1d_reaction", "pinnsformer"): (3.0e-6, 0.015, 0.030),
    ("1d_wave", "pinn"): (1.93e-2, 0.326, 0.335), ("1d_wave", "pinnsformer"): (1.38e-2, 0.270, 0.283),
}

for pde, g in df.groupby("pde", sort=False):
    print(f"\n## {pde}\n")
    agg = g.groupby("config").agg(seeds=("seed", "count"), loss=("loss", "mean"),
                                  rMAE=("rMAE", "mean"), rMAE_std=("rMAE", "std"),
                                  rRMSE=("rRMSE", "mean"), rRMSE_std=("rRMSE", "std"),
                                  s_per_step=("s_per_step", "mean"), mem_MiB=("peak_mem_MiB", "mean"),
                                  params=("n_params", "first"), model=("model", "first"),
                                  grid=("train_grid", "first"), k=("k", "first"), act=("activation", "first"))
    rows = []
    for cfg, r in agg.iterrows():
        default_grid = (r.grid == 51) if r.model == "pinnsformer" else (r.grid == 101)
        std_pf = r.model != "pinnsformer" or (r.k == 5 and r.act in ("wavelet", "default"))
        ref = paper.get((pde, r.model)) if (default_grid and std_pf and "nbconfig" not in cfg) else None
        if r.model == "pinnsformer" and r.act in ("sin", "relu") and default_grid:  # paper Table 6
            ref = {("convection", "sin"): (0.3159, 1.074, 1.141), ("convection", "relu"): (0.5256, 1.001, 1.001),
                   ("1d_reaction", "sin"): (4.9e-6, 0.017, 0.032), ("1d_reaction", "relu"): (0.2083, 0.994, 0.996)}.get((pde, r.act))
        rows.append(dict(config=cfg, seeds=int(r.seeds), params=int(r.params), loss=f"{r.loss:.2e}",
                         rMAE=f"{r.rMAE:.3f}" + (f" ± {r.rMAE_std:.3f}" if r.seeds > 1 else ""),
                         rRMSE=f"{r.rRMSE:.3f}" + (f" ± {r.rRMSE_std:.3f}" if r.seeds > 1 else ""),
                         paper_rMAE=f"{ref[1]:.3f}" if ref else "-", paper_rRMSE=f"{ref[2]:.3f}" if ref else "-",
                         s_per_step=f"{r.s_per_step:.2f}", mem_MiB=f"{r.mem_MiB:.0f}"))
    out = pd.DataFrame(rows)
    if args.latex:
        for _, r in out.iterrows():
            print(f"{r.config} & {r.loss} & {r.rMAE} & {r.rRMSE} & {r.paper_rMAE} & {r.paper_rRMSE} & {r.s_per_step} & {r.mem_MiB} \\\\")
    else:
        print(out.to_string(index=False))
