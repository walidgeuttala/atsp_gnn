# Heterogeneous Graph Neural Networks for Scalable Asymmetric Traveling Salesman Problem Optimization

**Graph Neural Network-based Framework for Solving ATSP Efficiently**

This repository implements a novel three-stage pipeline for solving the Asymmetric Traveling Salesman Problem (ATSP) using Graph Neural Networks (GNNs). Our approach scales better and performs more accurately than previous GNN-based models on non-Euclidean TSP variants.

## 🚀 Results Summary

- ⏱️ **98% faster training** (1h vs. 55h) compared to GLOP  
- ✅ **29.58% reduction in optimality gap** on ATSP500 (9.38% vs. 38.96%)  
- ⚡ Inference time comparable to GLOP (15.05s vs. 14.86s on ATSP250)

## 🛠️ Installation

### Requirements

- Python ≥ 3.8  
- PyTorch ≥ 1.10  
- DGL ≥ 0.9  
- NumPy, SciPy, tqdm

### Setup

```bash
git clone https://github.com/your-username/atsp-gnn.git
cd atsp-gnn
pip install torch dgl numpy scipy tqdm
```

## 📂 Project Structure

```
.
├── configs/                # YAML configuration files
├── data/                   # Dataset and ATSP instances
├── models/                 # Saved model checkpoints
├── scripts/                # Helper scripts (data gen, experiments)
├── tours/                  # Generated and ground-truth tours
├── train.py                # Training script
├── infer.py                # Inference (tour generation)
├── evaluate.py             # Evaluation metrics
└── README.md               # Project documentation
```

## 🔄 Usage

### 1. Generate Synthetic Data

```bash
python scripts/generate_atsp.py --nodes 500 --seed 42
```

### 2. Train Model

```bash
python train.py   --data_dir data/atsp_instances   --model_type het_gat_concat   --epochs 100   --batch_size 32
```

### 3. Inference - Generate Tours

```bash
python infer.py   --checkpoint models/het_gat_concat_best.pth   --instance data/atsp500_instance_1.npy   --output tours/predicted_tour.json
```

### 4. Evaluate Tour Quality

```bash
python evaluate.py   --ground_truth tours/optimal_tours   --predictions tours/predicted_tour.json
```

## ⚙️ Configuration

Edit parameters in `configs/config_gnn.yaml`:

```yaml
hidden_dim: 128
attention_heads: 4
learning_rate: 0.001
batch_size: 32
dropout: 0.2
```

## 🔁 Reproduce Full Experiments

1. Place ATSPLIB benchmark instances into `data/atsp_benchmarks/`  
2. Run:

```bash
bash scripts/run_experiments.sh
```

## 📜 License

**Academic Use Only**: Non-commercial research purposes. Contact authors for other use cases.

## 📚 Citation (WIP)

```bibtex
@article{yourname2024heterogeneous,
  title={Heterogeneous Line Graph for Asymmetric TSP Solutions with GNNs},
  author={Your Name and Coauthors},
  journal={Under Review},
  year={2024}
}
```

## 📬 Contact

**Your Name**  
✉️ your.email@example.com  
🔗 [Lab Website](https://your-lab-site.com)

## 🧪 Example Run

```bash
# Prepare data
python scripts/generate_atsp.py --nodes 250 --seed 123

# Train
python train.py --data_dir data/atsp_instances --model_type het_gat_concat

# Inference
python infer.py --checkpoint models/best_model.pth --instance data/atsp_instance.npy --output out.json

# Evaluate
python evaluate.py --ground_truth data/gt_tours --predictions out.json
```
