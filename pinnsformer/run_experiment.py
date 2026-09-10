"""
Unified experiment runner for the PINNsFormer reproducibility study.

Re-implements the training/evaluation loop of the authors' demo notebooks
(demo/convection, demo/1d_reaction, demo/1d_wave) as one script so that every
model / PDE / seed combination is run identically and logged to a CSV.

Examples
--------
  # Reproduce Table 1, convection, PINNsFormer, exactly as in the notebook
  python run_experiment.py --pde convection --model pinnsformer

  # Plain PINN baseline
  python run_experiment.py --pde convection --model pinn

  # Beyond the paper: PINN with Wavelet activation instead of Tanh
  python run_experiment.py --pde convection --model pinn_wavelet

  # Beyond the paper: PINNsFormer with k=1 (no pseudo-sequence)
  python run_experiment.py --pde 1d_reaction --model pinnsformer --k 1

  # Fairness: PINN on the same 51x51 training grid PINNsFormer uses
  python run_experiment.py --pde convection --model pinn --train_grid 51

Every run appends one row to results/results.csv and writes
results/<run_name>/{pred.npy,loss.json,figure.png,config.json}.
"""

import argparse
import csv
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import LBFGS

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from util import get_data, make_time_sequence, get_n_params  # noqa: E402
from model.pinn import PINNs  # noqa: E402
from model.qres import QRes  # noqa: E402
from model.fls import FLS  # noqa: E402
from model.pinnsformer import PINNsformer, WaveAct  # noqa: E402


# --------------------------------------------------------------------------- #
# PDE definitions (mirroring Appendix B of the paper and the demo notebooks)
# --------------------------------------------------------------------------- #

class PDE:
    """Holds domain, residual, IC/BC losses and ground truth for one PDE."""

    name = ""
    x_range = (0.0, 1.0)
    t_range = (0.0, 1.0)

    def residual(self, model, x, t):
        raise NotImplementedError

    def ic_bc_loss(self, model, x_left, t_left, x_upper, t_upper, x_lower, t_lower):
        """Returns (loss_ic, loss_bc). 'left' is t=0, 'upper'/'lower' are x=max/min."""
        raise NotImplementedError

    def ground_truth(self, xt):
        """xt: (N,2) numpy array of test points -> (N,) exact solution."""
        raise NotImplementedError


def grad(outputs, inputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs),
                               retain_graph=True, create_graph=True)[0]


class Convection(PDE):
    """u_t + beta u_x = 0, x in [0,2pi], t in [0,1]; IC sin(x); periodic BC; beta=50."""
    name = "convection"
    x_range = (0.0, 2 * np.pi)
    beta = 50.0

    def residual(self, model, x, t):
        u = model(x, t)
        u_x, u_t = grad(u, x), grad(u, t)
        return u_t + self.beta * u_x, u

    def ic_bc_loss(self, model, x_left, t_left, x_upper, t_upper, x_lower, t_lower):
        pred_left = model(x_left, t_left)
        pred_upper = model(x_upper, t_upper)
        pred_lower = model(x_lower, t_lower)
        loss_ic = torch.mean((pred_left[:, 0] - torch.sin(x_left[:, 0])) ** 2)
        loss_bc = torch.mean((pred_upper - pred_lower) ** 2)
        return loss_ic, loss_bc

    def ground_truth(self, xt):
        import scipy.io
        mat = scipy.io.loadmat(os.path.join(ROOT, "demo", "convection", "convection.mat"))
        return mat["u"].reshape(-1)


class Reaction1D(PDE):
    """u_t - rho u (1-u) = 0, x in [0,2pi]; IC gaussian bump; periodic BC; rho=5."""
    name = "1d_reaction"
    x_range = (0.0, 2 * np.pi)
    rho = 5.0

    def residual(self, model, x, t):
        u = model(x, t)
        u_t = grad(u, t)
        return u_t - self.rho * u * (1 - u), u

    def ic_bc_loss(self, model, x_left, t_left, x_upper, t_upper, x_lower, t_lower):
        pred_left = model(x_left, t_left)
        pred_upper = model(x_upper, t_upper)
        pred_lower = model(x_lower, t_lower)
        h = torch.exp(-(x_left[:, 0] - torch.pi) ** 2 / (2 * (torch.pi / 4) ** 2))
        loss_ic = torch.mean((pred_left[:, 0] - h) ** 2)
        loss_bc = torch.mean((pred_upper - pred_lower) ** 2)
        return loss_ic, loss_bc

    def ground_truth(self, xt):
        x, t = xt[:, 0], xt[:, 1]
        h = np.exp(-(x - np.pi) ** 2 / (2 * (np.pi / 4) ** 2))
        return h * np.exp(self.rho * t) / (h * np.exp(self.rho * t) + 1 - h)


class Wave1D(PDE):
    """u_tt - c u_xx = 0 on [0,1]^2. NOTE: paper text says beta=3 but the analytic
    solution and the authors' code both correspond to c = 4."""
    name = "1d_wave"
    x_range = (0.0, 1.0)
    c = 4.0

    def residual(self, model, x, t):
        u = model(x, t)
        u_x = grad(u, x)
        u_xx = grad(u_x, x)
        u_t = grad(u, t)
        u_tt = grad(u_t, t)
        return u_tt - self.c * u_xx, u

    def ic_bc_loss(self, model, x_left, t_left, x_upper, t_upper, x_lower, t_lower):
        pred_left = model(x_left, t_left)
        pred_upper = model(x_upper, t_upper)
        pred_lower = model(x_lower, t_lower)
        pi = torch.pi
        ui_t = grad(pred_left, t_left)
        loss_ic = torch.mean((pred_left[:, 0] - torch.sin(pi * x_left[:, 0])
                              - 0.5 * torch.sin(3 * pi * x_left[:, 0])) ** 2) \
            + torch.mean(ui_t ** 2)
        loss_bc = torch.mean(pred_upper ** 2) + torch.mean(pred_lower ** 2)
        return loss_ic, loss_bc

    def ground_truth(self, xt):
        x, t = xt[:, 0], xt[:, 1]
        return np.sin(np.pi * x) * np.cos(2 * np.pi * t) + 0.5 * np.sin(3 * np.pi * x) * np.cos(6 * np.pi * t)


PDES = {"convection": Convection, "1d_reaction": Reaction1D, "1d_wave": Wave1D}


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

def replace_activation(module, old_cls, new_factory):
    for name, child in module.named_children():
        if isinstance(child, old_cls):
            setattr(module, name, new_factory())
        else:
            replace_activation(child, old_cls, new_factory)


class SinAct(nn.Module):
    def forward(self, x):
        return torch.sin(x)


ACTIVATIONS = {"wavelet": WaveAct, "sin": SinAct, "relu": nn.ReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh}


def build_model(args):
    """Returns (model, is_sequence_model)."""
    if args.model == "pinn":
        return PINNs(in_dim=2, hidden_dim=512, out_dim=1, num_layer=4), False
    if args.model == "pinn_wavelet":
        m = PINNs(in_dim=2, hidden_dim=512, out_dim=1, num_layer=4)
        replace_activation(m, nn.Tanh, WaveAct)
        return m, False
    if args.model == "fls":
        return FLS(in_dim=2, hidden_dim=512, out_dim=1, num_layer=4), False
    if args.model == "qres":
        return QRes(in_dim=2, hidden_dim=args.qres_hidden, out_dim=1, num_layer=args.qres_layers), False
    if args.model == "pinnsformer":
        m = PINNsformer(d_out=1, d_hidden=512, d_model=args.d_model, N=args.n_layers, heads=args.heads)
        if args.activation != "wavelet":  # Table 6 ablation: swap every Wavelet activation
            replace_activation(m, WaveAct, ACTIVATIONS[args.activation])
        return m, True
    raise ValueError(args.model)


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pde", choices=PDES.keys(), required=True)
    p.add_argument("--model", choices=["pinn", "pinn_wavelet", "fls", "qres", "pinnsformer"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=500,
                   help="L-BFGS outer steps. Authors' notebooks use 500 (paper text says 1000).")
    p.add_argument("--train_grid", type=int, default=None,
                   help="Training mesh size per axis. Default: 51 for pinnsformer, 101 otherwise (as in the notebooks).")
    p.add_argument("--test_grid", type=int, default=101)
    p.add_argument("--k", type=int, default=5, help="pseudo-sequence length (pinnsformer only)")
    p.add_argument("--dt", type=float, default=1e-4,
                   help="pseudo-sequence step. All non-NTK notebooks use 1e-4 (the 1d_wave NTK notebook uses 1e-3).")
    p.add_argument("--activation", choices=ACTIVATIONS.keys(), default="wavelet",
                   help="pinnsformer only: replace the Wavelet activation everywhere (paper Table 6 ablation).")
    p.add_argument("--qres_hidden", type=int, default=256,
                   help="QRes width. Paper Table 4 and the 1d_reaction notebook use 256; the convection notebook uses 512.")
    p.add_argument("--qres_layers", type=int, default=4,
                   help="QRes num_layer. Paper and 1d_reaction notebook use 4; the convection notebook uses 2.")
    p.add_argument("--d_model", type=int, default=32)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--n_layers", type=int, default=1)
    p.add_argument("--device", default=None, help="cuda / cpu / mps. Default: cuda if available else cpu.")
    p.add_argument("--out", default=os.path.join(ROOT, "results"))
    p.add_argument("--tag", default="", help="extra string appended to the run name")
    p.add_argument("--save_model", action="store_true")
    args = p.parse_args()

    # ---- defaults that follow the authors' notebooks
    is_seq = args.model == "pinnsformer"
    if args.train_grid is None:
        args.train_grid = 51 if is_seq else 101
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ---- seeding
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    pde = PDES[args.pde]()
    run_name = f"{args.pde}_{args.model}_g{args.train_grid}_it{args.iters}_s{args.seed}"
    if is_seq:
        run_name += f"_k{args.k}_dt{args.dt:g}"
        if args.activation != "wavelet":
            run_name += f"_{args.activation}"
    if args.tag:
        run_name += f"_{args.tag}"
    run_dir = os.path.join(args.out, run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"[run] {run_name}  device={device}")

    # ---- data
    res, b_left, b_right, b_upper, b_lower = get_data(pde.x_range, pde.t_range, args.train_grid, args.train_grid)
    res_test, _, _, _, _ = get_data(pde.x_range, pde.t_range, args.test_grid, args.test_grid)
    n_res, n_ic, n_bc = len(res), len(b_left), len(b_upper)

    if is_seq:
        seq = lambda a: make_time_sequence(a, num_step=args.k, step=args.dt)  # noqa: E731
        res, b_left, b_upper, b_lower = map(seq, (res, b_left, b_upper, b_lower))
        res_test_in = seq(res_test)
    else:
        res_test_in = res_test

    def to_t(a):
        return torch.tensor(a, dtype=torch.float32, requires_grad=True).to(device)

    res, b_left, b_upper, b_lower = map(to_t, (res, b_left, b_upper, b_lower))
    x_res, t_res = res[..., 0:1], res[..., 1:2]
    x_left, t_left = b_left[..., 0:1], b_left[..., 1:2]
    x_upper, t_upper = b_upper[..., 0:1], b_upper[..., 1:2]
    x_lower, t_lower = b_lower[..., 0:1], b_lower[..., 1:2]

    # ---- model
    model, _ = build_model(args)
    model = model.to(device)
    model.apply(init_weights)
    n_params = get_n_params(model)
    print(f"[model] {args.model}  params={n_params:,}")
    optim = LBFGS(model.parameters(), line_search_fn="strong_wolfe")

    # ---- train
    loss_track = []
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for i in range(args.iters):
        def closure():
            r, _ = pde.residual(model, x_res, t_res)
            loss_res = torch.mean(r ** 2)
            loss_ic, loss_bc = pde.ic_bc_loss(model, x_left, t_left, x_upper, t_upper, x_lower, t_lower)
            loss = loss_res + loss_ic + loss_bc
            loss_track.append([loss_res.item(), loss_ic.item(), loss_bc.item()])
            optim.zero_grad()
            loss.backward()
            return loss
        optim.step(closure)
        if (i + 1) % 50 == 0 or i == 0:
            lr_, li_, lb_ = loss_track[-1]
            print(f"  step {i+1:5d}/{args.iters}  res={lr_:.3e} ic={li_:.3e} bc={lb_:.3e}  "
                  f"total={lr_+li_+lb_:.3e}  {time.time()-t0:6.1f}s", flush=True)
    train_time = time.time() - t0
    peak_mem_mib = torch.cuda.max_memory_allocated() / 2**20 if device.startswith("cuda") else float("nan")
    final_loss = float(np.sum(loss_track[-1]))

    # ---- evaluate (rMAE / rRMSE = relative L1 / L2, eq. 7 in the paper)
    xt_test = to_t(res_test_in)
    with torch.no_grad():
        pred = model(xt_test[..., 0:1], xt_test[..., 1:2])
        if is_seq:
            pred = pred[:, 0]  # first element of the pseudo-sequence is u(x,t)
        pred = pred.reshape(-1).cpu().numpy()
    u = pde.ground_truth(res_test).reshape(-1)
    rl1 = float(np.sum(np.abs(u - pred)) / np.sum(np.abs(u)))
    rl2 = float(np.sqrt(np.sum((u - pred) ** 2) / np.sum(u ** 2)))
    print(f"[result] loss={final_loss:.3e}  rMAE={rl1:.4f}  rRMSE={rl2:.4f}  "
          f"time={train_time:.1f}s ({train_time/args.iters:.2f}s/step)  peak_mem={peak_mem_mib:.0f}MiB")

    # ---- save
    np.save(os.path.join(run_dir, "pred.npy"), pred.reshape(args.test_grid, args.test_grid))
    with open(os.path.join(run_dir, "loss.json"), "w") as f:
        json.dump(loss_track, f)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    if args.save_model:
        torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        P = pred.reshape(args.test_grid, args.test_grid)
        U = u.reshape(args.test_grid, args.test_grid)
        ext = [pde.x_range[0], pde.x_range[1], pde.t_range[1], pde.t_range[0]]
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
        for a, img, title in zip(ax, [U, P, np.abs(P - U)], ["Exact u(x,t)", f"{args.model} prediction", "Absolute error"]):
            im = a.imshow(img, extent=ext, aspect="auto")
            a.set_title(title); a.set_xlabel("x"); a.set_ylabel("t")
            fig.colorbar(im, ax=a)
        fig.suptitle(f"{args.pde}: rMAE={rl1:.3f} rRMSE={rl2:.3f}")
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, "figure.png"), dpi=130)
        plt.close(fig)
    except Exception as e:  # plotting is optional
        print("[warn] plotting failed:", e)

    row = dict(run=run_name, pde=args.pde, model=args.model, activation=args.activation if is_seq else
               ("wavelet" if args.model == "pinn_wavelet" else "default"), seed=args.seed, iters=args.iters,
               train_grid=args.train_grid, n_res=n_res, n_ic=n_ic, n_bc=n_bc,
               k=args.k if is_seq else 1, dt=args.dt if is_seq else 0.0,
               n_params=n_params, loss=final_loss, loss_res=loss_track[-1][0], loss_ic=loss_track[-1][1],
               loss_bc=loss_track[-1][2], rMAE=rl1, rRMSE=rl2, train_time_s=round(train_time, 1),
               s_per_step=round(train_time / args.iters, 3), peak_mem_MiB=round(peak_mem_mib, 1),
               device=device, gpu=torch.cuda.get_device_name(0) if device.startswith("cuda") else "cpu")
    csv_path = os.path.join(args.out, "results.csv")
    existing, fields = [], list(row.keys())
    if os.path.exists(csv_path):  # merge columns so older rows (without e.g. 'activation') stay valid
        with open(csv_path, newline="") as f:
            r = csv.DictReader(f)
            existing = list(r)
            fields = list(r.fieldnames or []) + [k for k in row if k not in (r.fieldnames or [])]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        for old in existing:
            w.writerow(old)
        w.writerow(row)
    print(f"[saved] {run_dir}  ->  {csv_path}")


if __name__ == "__main__":
    main()
