Here’s a clean, drop-in README you can paste into your repo. I kept it simple, no emojis, and added your email at the end.

````markdown
# Heterogeneous Graph Neural Networks for Scalable ATSP Optimization

This repository contains code accompanying the paper:

**Heterogeneous Graph Neural Networks for Scalable Asymmetric Traveling Salesman Problem Optimization**

Our approach tackles ATSP with a three-stage pipeline:
1) **HL(G)**: a heterogeneous line-graph transform that turns directed edges into nodes and preserves four typed adjacencies (parallel, source–source, source–target, target–target)
2) **Het-GAT**: a relation-aware GNN that predicts **approximate** edge regrets with fusion by **sum**, **concat**, or **attention**
3) **EB3O**: regret-guided construction with a batch-selecting Edge Builder and refinement with pre-evaluating directed 3-Opt

The method scales to 1,000 nodes via **subgraph covering** (overlapping 250-node induced subgraphs) and achieves strong accuracy–efficiency trade-offs on ATSP100–1000.

---

## Quick Start

### 1) Environment

We recommend Python 3.10+.

Using Pipenv (preferred in this repo):
```bash
# install git-lfs if you plan to pull large models/datasets
# macOS: brew install git-lfs
# Ubuntu: sudo apt-get install git-lfs
git lfs install

pip install pipenv
pipenv install
pipenv shell
````

Or using pip:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # if provided
```

Core dependencies (typical):

* PyTorch, DGL (heterogeneous graphs), NumPy, NetworkX, Pandas, TQDM
* (Optional) Matplotlib for plots
* (Optional) LKH-3 binary if you want to (re)generate supervision regrets

### 2) Data

**ATSP datasets** follow sizes {50, 100, 150, 250, 500, 1000}. You can:

* Use the same data protocol as MatNet/GLOP (if available locally), or
* Generate synthetic ATSP instances (complete directed graphs, triangle inequality enforced) with a helper script.

Example (synthetic):

```bash
python scripts/generate_atsp.py \
  --n_instances 2500 --n 50 \
  --out data/ATSP50
```

**Regret labels**: we train on **approximate regrets** (from LKH-3 or your chosen solver) for practicality.

```bash
# Option A: compute approx. regrets with LKH-3 (fast baseline)
python scripts/compute_regrets_lkh3.py \
  --instances data/ATSP50 \
  --out data/ATSP50_regrets

# Option B: load precomputed regrets if you have them
```

### 3) HL(G) preprocessing

Build the heterogeneous line graph representation per instance:

```bash
python scripts/build_hlg.py \
  --in data/ATSP50 \
  --out data/ATSP50_hlg \
  --relations ss tt st pp      # choose typed relations to include
```

### 4) Train Het-GAT

Choose a fusion:

* `concat` (best efficiency–accuracy trade-off)
* `attn` (lowest gaps, slightly higher overhead)
* `sum` (most memory-friendly)

Examples:

```bash
# Het-GAT with concatenation fusion
python src/train_hetgat.py \
  --train data/ATSP50_hlg \
  --regrets data/ATSP50_regrets \
  --fusion concat \
  --epochs 200 \
  --batch_size 15 \
  --lr 1e-3 \
  --save_dir runs/hetgat_concat_atsp50

# Het-GAT with attention fusion
python src/train_hetgat.py \
  --train data/ATSP50_hlg \
  --regrets data/ATSP50_regrets \
  --fusion attn \
  --epochs 200 \
  --batch_size 15 \
  --lr 1e-3 \
  --save_dir runs/hetgat_attn_atsp50
```

### 5) Inference and EB3O

Produce regret predictions and run the regret-guided pipeline (Edge Builder + Pre-Evaluating 3-Opt):

```bash
python src/infer_and_solve.py \
  --test data/ATSP250_hlg \
  --ckpt runs/hetgat_concat_atsp50/best.pt \
  --post EB3O \
  --out results/ATSP250_concat_EB3O
```

**Subgraph covering** for large instances (e.g., ATSP1000):

```bash
python src/infer_and_solve.py \
  --test data/ATSP1000 \
  --ckpt runs/hetgat_attn_atsp50/best.pt \
  --post EB3O \
  --subgraph_cover 250 \
  --out results/ATSP1000_attn_EB3O
```

---

## Reproducing Main Results

Below are typical commands to reproduce the results table (adjust paths/sizes as needed). Each uses the same trained model (from ATSP50) and evaluates at larger sizes.

```bash
# ATSP100
python src/infer_and_solve.py \
  --test data/ATSP100_hlg --ckpt runs/hetgat_concat_atsp50/best.pt \
  --post EB3O --out results/ATSP100_concat_EB3O

# ATSP150
python src/infer_and_solve.py \
  --test data/ATSP150_hlg --ckpt runs/hetgat_attn_atsp50/best.pt \
  --post EB3O --out results/ATSP150_attn_EB3O

# ATSP250
python src/infer_and_solve.py \
  --test data/ATSP250_hlg --ckpt runs/hetgat_concat_atsp50/best.pt \
  --post EB3O --out results/ATSP250_concat_EB3O

# ATSP500
python src/infer_and_solve.py \
  --test data/ATSP500_hlg --ckpt runs/hetgat_concat_atsp50/best.pt \
  --post EB3O --out results/ATSP500_concat_EB3O

# ATSP1000 with subgraph covering (250)
python src/infer_and_solve.py \
  --test data/ATSP1000 --ckpt runs/hetgat_attn_atsp50/best.pt \
  --post EB3O --subgraph_cover 250 \
  --out results/ATSP1000_attn_EB3O
```

**Notes**

* EB3O = **E**dge **B**uilder + **3-O**pt (regret-guided pair)
* NN2R = **N**earest **N**eighbor + **2-O**pt + **R**elocation (legacy baseline; optional)
* All reported gaps are computed against the same reference costs (e.g., LKH-3) for fairness
* Training uses **approximate regrets** for labels

---

## Repository Structure (typical)

```
.
├── scripts/
│   ├── generate_atsp.py
│   ├── compute_regrets_lkh3.py
│   ├── build_hlg.py
│   └── ...
├── src/
│   ├── models/hetgat.py
│   ├── train_hetgat.py
│   ├── infer_and_solve.py
│   ├── eb3o/       # Edge Builder + Pre-Evaluating 3-Opt
│   └── utils/
├── data/
├── runs/
├── results/
├── Pipfile / Pipfile.lock
└── README.md
```

(If your filenames differ, adjust the command examples accordingly.)

---

## Citation

If you use this code, please cite:

```bibtex
@article{Guettala2025HetGATATSP,
  title   = {Heterogeneous Graph Neural Networks for Scalable Asymmetric Traveling Salesman Problem Optimization},
  author  = {Walid Guettala and Ákos Levente Holló-Szabó and László Gulyás and János Botzheim},
  journal = {Preprint},
  year    = {2025}
}
```

You may also cite LKH-3 and any datasets you used in generating approximate regrets.

---

## License

This project is released under a permissive open-source license (see `LICENSE` if provided).

---

## Contact

For questions or collaboration, please contact: **[guettalawalid@inf.elte.hu](mailto:guettalawalid@inf.elte.hu)**
