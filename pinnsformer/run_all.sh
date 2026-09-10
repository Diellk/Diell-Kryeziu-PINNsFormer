#!/usr/bin/env bash
# Full experiment sweep for the reproducibility report.
# Usage:  bash run_all.sh [ITERS]      (default ITERS=500, as in the authors' notebooks)
# Runs are appended to results/results.csv; already-finished runs are skipped.
set -e
ITERS=${1:-500}
cd "$(dirname "$0")"

run () {  # skip a run if its directory already contains pred.npy
  local name
  name=$(python3 - "$@" <<'EOF'
import sys
a = sys.argv[1:]
def get(flag, default):
    return a[a.index(flag)+1] if flag in a else default
pde, model, seed = get('--pde', ''), get('--model', ''), get('--seed', '0')
iters = get('--iters', '500'); k = get('--k', '5')
grid = get('--train_grid', '51' if model == 'pinnsformer' else '101')
dt = get('--dt', '1e-4')
name = f"{pde}_{model}_g{grid}_it{iters}_s{seed}"
if model == 'pinnsformer':
    name += f"_k{k}_dt{float(dt):g}"
    act = get('--activation', 'wavelet')
    if act != 'wavelet': name += f"_{act}"
if '--tag' in a: name += f"_{get('--tag', '')}"
print(name)
EOF
)
  if [ -f "results/$name/pred.npy" ]; then echo "[skip] $name"; return; fi
  python3 run_experiment.py "$@"
}

echo "=================== TIER 1: Table 1 (convection, 1D-reaction), 4 models ==================="
for pde in convection 1d_reaction; do
  for model in pinn qres fls pinnsformer; do
    run --pde $pde --model $model --iters $ITERS --seed 0
  done
done

echo "=================== TIER 1b: QRes exactly as in the convection notebook (512 wide, 2 layers) ==================="
run --pde convection --model qres --iters $ITERS --seed 0 --qres_hidden 512 --qres_layers 2 --tag nbconfig

echo "=================== TIER 2a: seeds (variance the paper does not report) ==================="
for pde in convection 1d_reaction; do
  for seed in 1 2; do
    run --pde $pde --model pinn        --iters $ITERS --seed $seed
    run --pde $pde --model pinnsformer --iters $ITERS --seed $seed
  done
done

echo "=================== TIER 2b: is it the activation? PINN + Wavelet ==================="
for pde in convection 1d_reaction; do
  run --pde $pde --model pinn_wavelet --iters $ITERS --seed 0
done

echo "=================== TIER 2c: is it the pseudo-sequence? PINNsFormer k=1 ==================="
for pde in convection 1d_reaction; do
  run --pde $pde --model pinnsformer --iters $ITERS --seed 0 --k 1
done

echo "=================== TIER 2d: fairness of the training grid ==================="
for pde in convection 1d_reaction; do
  run --pde $pde --model pinn        --iters $ITERS --seed 0 --train_grid 51
  run --pde $pde --model pinnsformer --iters $ITERS --seed 0 --train_grid 101
done

echo "=================== TIER 2e: is Wavelet necessary? PINNsFormer with Sin / ReLU (paper Table 6, subset) ==================="
for pde in convection 1d_reaction; do
  run --pde $pde --model pinnsformer --iters $ITERS --seed 0 --activation sin
  run --pde $pde --model pinnsformer --iters $ITERS --seed 0 --activation relu
done

echo "=================== TIER 3: 1D-wave without NTK (Table 2, first two rows) ==================="
run --pde 1d_wave --model pinn        --iters 1000 --seed 0
run --pde 1d_wave --model pinnsformer --iters 1000 --seed 0

echo "Done. Summary:"
python3 summarize.py
