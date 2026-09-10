# [Re] PINNsFormer: a reproducibility study

Diell Kryeziu, Advanced Topics in Machine Learning, USI.

Reproducibility study of *PINNsFormer: A Transformer-Based Framework for Physics-Informed Neural
Networks* (Zhao, Ding, Prakash, ICLR 2024, https://openreview.net/forum?id=DO2WFXU1Be).

## Contents

| Folder / file | What it is |
|---|---|
| `ATML_Report/` | The report. `main.pdf` is the compiled version; `main.tex`, `main.bib`, `appendix.tex` and the `fig_*.png` figures are the source. |
| `ATML_Presentation/` | The slides.
| `pinnsformer/` | The authors' code (cloned from https://github.com/AdityaLab/pinnsformer) plus our additions, listed below, and all experiment outputs in `pinnsformer/results/`. |
| `1048_PINNsFormer_A_Transformer.pdf` | The paper. |
| `NEXT_STEPS.md` | Working notes kept during the project (plan, discrepancies found, verification of the runner). |

## Our changes inside `pinnsformer/`

The authors' files (`model/`, `util.py`, `demo/`, `checkpoint/`, `pyhessian.py`, `vis_landscape.py`,
`README.md`) are unchanged. Everything we added is at the top level of the folder:

| File | Purpose |
|---|---|
| `run_experiment.py` | The experiment runner. Trains and evaluates one model on one PDE with the same meshes, loss terms, optimiser, initialisation and metrics as the authors' notebooks, and logs loss, rMAE, rRMSE, time per step and peak GPU memory. Imports the authors' `model/` and `util.py` unchanged. Adds the configurations the paper does not have: `--model pinn_wavelet` (PINN with the Wavelet activation), `--k` (pseudo-sequence length, e.g. `--k 1`), `--train_grid` (mesh size), `--activation {sin,relu,...}` (Table 6 ablation), `--qres_hidden/--qres_layers` (the two QRes configurations found in the notebooks), `--seed`, `--iters`. |
| `run_all.sh` | The full sweep in priority order (Table 1, seeds, PINN+Wavelet, k=1, matched meshes, Sin/ReLU, 1D-wave). Skips runs whose output already exists. |
| `summarize.py` | Turns `results/results.csv` into tables with the paper's numbers alongside ours (`--latex` prints rows for the report). |
| `PINNsFormer_Reproduction.ipynb` | Self-contained Colab notebook: clones the repo, writes the three scripts above into it, runs the sweep, evaluates the pretrained checkpoint (the live demo), and collects `results.zip`. This is how the experiments were actually run. |
| `make_colab_notebook.py` | Regenerates the notebook from the three scripts. |
| `results/` | One folder per run (34 runs) with `pred.npy` (prediction on the 101x101 test mesh), `loss.json` (loss per L-BFGS closure evaluation), `figure.png` and `config.json`, plus `results.csv` with one row per run. |

## How to reproduce

On Colab: upload `pinnsformer/PINNsFormer_Reproduction.ipynb`, choose a GPU runtime, run top to bottom.

Locally (GPU recommended; Python 3.10+, PyTorch 2.x, NumPy, SciPy, Matplotlib, pandas):

```bash
cd pinnsformer
python run_experiment.py --pde convection --model pinnsformer          # one run, authors' settings
python run_experiment.py --pde convection --model pinnsformer --k 1    # one of our additional experiments
bash run_all.sh                                                        # the whole study (about 7 GPU hours on a T4)
python summarize.py                                                    # tables
```

The runner was checked line by line against all ten convection, 1D-reaction and 1D-wave notebooks in
`pinnsformer/demo/`; the discrepancies between the paper and the code that this revealed are listed in
Section 5.2 of the report.
