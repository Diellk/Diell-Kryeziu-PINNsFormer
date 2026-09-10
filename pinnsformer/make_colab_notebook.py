"""Generate PINNsFormer_Reproduction.ipynb, a self-contained Colab notebook.

The notebook clones the authors' repo and then writes run_experiment.py,
run_all.sh and summarize.py into it (their contents are embedded from this
folder), so nothing needs to be uploaded. Re-run this script after editing
any of those files:

    python make_colab_notebook.py
"""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))


def read(name):
    with open(os.path.join(ROOT, name)) as f:
        return f.read()


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


cells = [
    md("# PINNsFormer reproducibility study\n\n"
       "Reproduces Table 1 (and parts of Table 2) of *PINNsFormer: A Transformer-Based Framework for "
       "Physics-Informed Neural Networks* (Zhao, Ding, Prakash, ICLR 2024) plus a few experiments beyond the paper.\n\n"
       "**Runtime -> Change runtime type -> GPU (T4 is enough).** Then run the cells top to bottom. "
       "Results accumulate in `pinnsformer/results/results.csv`; the last cells copy them to Google Drive so "
       "nothing is lost when the session dies."),
    code("!nvidia-smi\n"
         "!git clone -q https://github.com/AdityaLab/pinnsformer.git\n"
         "%cd pinnsformer\n"
         "!pip -q install scipy"),
    md("## Experiment code\nThe three cells below write our runner into the cloned repo."),
    code("%%writefile run_experiment.py\n" + read("run_experiment.py")),
    code("%%writefile run_all.sh\n" + read("run_all.sh")),
    code("%%writefile summarize.py\n" + read("summarize.py")),
    md("## (Optional) mount Google Drive so results survive a disconnect\n"
       "Skip if you prefer to download `results.zip` manually at the end."),
    code("from google.colab import drive\n"
         "drive.mount('/content/drive')\n"
         "!mkdir -p /content/drive/MyDrive/pinnsformer_results\n"
         "# If you have results from a previous session, restore them so finished runs are skipped:\n"
         "!cp -rn /content/drive/MyDrive/pinnsformer_results/results . 2>/dev/null || true"),
    md("## Sanity check (about 1 minute)\nA short run to make sure everything works before the long sweep."),
    code("!python run_experiment.py --pde convection --model pinnsformer --iters 5 --train_grid 21 --out results_smoke\n"
         "!rm -rf results_smoke"),
    md("## Option A: run the whole study in one cell\n"
       "This runs Tier 1, 2 and 3 in priority order (about 5-7 GPU hours on a T4), copies results to Drive after "
       "every run, and skips runs that already exist. If Colab disconnects, just re-run the setup cells above and this cell "
       "again; it continues where it stopped. Skip to Option B if you prefer to run tier by tier."),
    code("import subprocess, sys\n"
         "# stream output live and back up after each run\n"
         "proc = subprocess.Popen(['bash', 'run_all.sh', '500'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)\n"
         "for line in proc.stdout:\n"
         "    print(line, end='')\n"
         "    if line.startswith('[saved]'):\n"
         "        subprocess.run('cp -r results /content/drive/MyDrive/pinnsformer_results/ 2>/dev/null', shell=True)\n"
         "proc.wait()"),
    md("## Option B, Tier 1: Table 1 (convection + 1D-reaction, four models)\n"
       "About 8 runs. Expect roughly 5-10 min per MLP baseline and 20-40 min per PINNsFormer run on a T4. "
       "`ITERS=500` matches the authors' notebooks; the paper text says 1000. "
       "You can run the whole sweep with `bash run_all.sh` instead (it also does Tier 2 and 3 and skips finished runs)."),
    code("ITERS = 500\n"
         "for pde in ['convection', '1d_reaction']:\n"
         "    for model in ['pinn', 'qres', 'fls', 'pinnsformer']:\n"
         "        !python run_experiment.py --pde {pde} --model {model} --iters {ITERS} --seed 0\n"
         "        !cp -r results /content/drive/MyDrive/pinnsformer_results/ 2>/dev/null || true\n"
         "# QRes with the configuration the convection notebook actually uses (512 wide, 2 layers; the paper says 256 x 4)\n"
         "!python run_experiment.py --pde convection --model qres --iters {ITERS} --seed 0 --qres_hidden 512 --qres_layers 2 --tag nbconfig"),
    code("!python summarize.py"),
    md("## Option B, Tier 2: beyond the paper\n"
       "* **Seeds**: the paper reports single runs. Two more seeds for PINN and PINNsFormer.\n"
       "* **PINN + Wavelet**: is the gain from the Transformer or from the activation?\n"
       "* **PINNsFormer with k=1**: removes the pseudo-sequence, keeps the architecture.\n"
       "* **Grid fairness**: the paper trains baselines on 101x101 but PINNsFormer on 51x51."),
    code("for pde in ['convection', '1d_reaction']:\n"
         "    for seed in [1, 2]:\n"
         "        !python run_experiment.py --pde {pde} --model pinn        --iters {ITERS} --seed {seed}\n"
         "        !python run_experiment.py --pde {pde} --model pinnsformer --iters {ITERS} --seed {seed}\n"
         "    !python run_experiment.py --pde {pde} --model pinn_wavelet --iters {ITERS} --seed 0\n"
         "    !python run_experiment.py --pde {pde} --model pinnsformer  --iters {ITERS} --seed 0 --k 1\n"
         "    !python run_experiment.py --pde {pde} --model pinn         --iters {ITERS} --seed 0 --train_grid 51\n"
         "    !python run_experiment.py --pde {pde} --model pinnsformer  --iters {ITERS} --seed 0 --train_grid 101\n"
         "    !cp -r results /content/drive/MyDrive/pinnsformer_results/ 2>/dev/null || true"),
    code("!python summarize.py"),
    md("## Option B, Tier 2e: is Wavelet necessary? (subset of the paper's Table 6)\n"
       "PINNsFormer with every Wavelet activation replaced by Sin or ReLU. The paper reports both fall back into the failure regime."),
    code("for pde in ['convection', '1d_reaction']:\n"
         "    for act in ['sin', 'relu']:\n"
         "        !python run_experiment.py --pde {pde} --model pinnsformer --iters {ITERS} --seed 0 --activation {act}\n"
         "    !cp -r results /content/drive/MyDrive/pinnsformer_results/ 2>/dev/null || true"),
    md("## Option B, Tier 3 (if time): 1D-wave without NTK (first two rows of Table 2)\n"
       "The NTK variants are in the authors' notebooks `demo/1d_wave/*_ntk.ipynb`; they need 1000 iterations."),
    code("!python run_experiment.py --pde 1d_wave --model pinn        --iters 1000 --seed 0\n"
         "!python run_experiment.py --pde 1d_wave --model pinnsformer --iters 1000 --seed 0\n"
         "!cp -r results /content/drive/MyDrive/pinnsformer_results/ 2>/dev/null || true"),
    md("## Demo: the authors' pretrained checkpoint (no training needed)\n"
       "Useful for the live demo in the presentation. Loads `checkpoint/convection_pinnsformer.pt` and plots it "
       "against the ground truth."),
    code("import numpy as np, torch, scipy.io, matplotlib.pyplot as plt\n"
         "from util import get_data, make_time_sequence\n"
         "from model.pinnsformer import PINNsformer\n"
         "dev = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
         "model = PINNsformer(d_out=1, d_hidden=512, d_model=32, N=1, heads=2).to(dev)\n"
         "model.load_state_dict(torch.load('checkpoint/convection_pinnsformer.pt', map_location=dev))\n"
         "model.eval()\n"
         "res_test, *_ = get_data([0, 2*np.pi], [0, 1], 101, 101)\n"
         "xt = torch.tensor(make_time_sequence(res_test, 5, 1e-4), dtype=torch.float32).to(dev)\n"
         "with torch.no_grad():\n"
         "    pred = model(xt[..., 0:1], xt[..., 1:2])[:, 0].cpu().numpy().reshape(101, 101)\n"
         "u = scipy.io.loadmat('demo/convection/convection.mat')['u'].reshape(101, 101)\n"
         "print('rMAE', np.abs(u-pred).sum()/np.abs(u).sum(), ' rRMSE', np.sqrt(((u-pred)**2).sum()/(u**2).sum()))\n"
         "fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))\n"
         "for a, img, t in zip(ax, [u, pred, np.abs(u-pred)], ['Exact', 'Pretrained PINNsFormer', 'Abs. error']):\n"
         "    im = a.imshow(img, extent=[0, 2*np.pi, 1, 0], aspect='auto'); a.set_title(t); fig.colorbar(im, ax=a)\n"
         "plt.tight_layout(); plt.show()"),
    md("## Collect everything\nDownload `results.zip` (CSV, per-run predictions, loss curves, figures) for the report."),
    code("!python summarize.py --latex\n"
         "!zip -qr results.zip results\n"
         "!cp results.zip /content/drive/MyDrive/pinnsformer_results/ 2>/dev/null || true\n"
         "from google.colab import files; files.download('results.zip')"),
]

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = os.path.join(ROOT, "PINNsFormer_Reproduction.ipynb")
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", out)
