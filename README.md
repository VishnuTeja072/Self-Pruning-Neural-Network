# 🧠 Self-Pruning Neural Network

A deep learning model that **learns to remove its own unnecessary connections** during training using learnable gates and L1 regularization. This results in a **sparse, efficient, and interpretable network** without manual pruning.

---

## 📌 Overview

Traditional neural networks are over-parameterized. This project introduces a **self-pruning mechanism** where each connection is controlled by a **learnable gate**.

- Gates are passed through a **sigmoid function** → values in (0, 1)
- An **L1 penalty** is applied to encourage sparsity
- The model automatically decides:
  - Which weights to keep ✅  
  - Which weights to remove ❌  

---

## ⚙️ Loss Function

```
Total Loss = CrossEntropyLoss + λ · Σ sigmoid(gate_scores)
```

### 🔍 Key Idea

- **Sigmoid(gate_scores)** → converts raw scores into probabilities
- **L1 penalty** → pushes gate values toward 0
- Final effect:
  - Important connections → gate ≈ 1  
  - Unimportant connections → gate ≈ 0  

---

## 🧠 Why L1 Encourages Sparsity

| Penalty | Behavior |
|--------|---------|
| L2 (gate²) | Gradually shrinks values, rarely reaches 0 |
| L1 (|gate|) | Constant push → drives values exactly to 0 |

---

## 📊 Results

| Lambda (λ) | Test Accuracy | Sparsity (%) |
|:----------:|:-------------:|:------------:|
| 1e-5       | ~52%          | ~15%         |
| 1e-4       | ~49%          | ~55%         |
| 1e-3       | ~40%          | ~88%         |

---

## 📉 Gate Distribution Insight

- Peak near **0** → pruned weights  
- Peak near **1** → important weights  

---

## 🏗️ Design Choices

| Component | Choice | Reason |
|----------|--------|--------|
| Gate Activation | Sigmoid | Smooth & differentiable |
| Initialization | Zeros (→ 0.5 after sigmoid) | Neutral start |
| Regularization | L1 penalty | Strong sparsity enforcement |
| Optimizer | Adam + Cosine LR | Stable training |
| Threshold | 0.01 | Defines effective pruning |

---

## 🚀 How to Run

```
pip install torch torchvision matplotlib numpy
python self_pruning_nn.py
```

---

## 📊 Graphs & Visualizations

### 1. Gate Distribution Histogram
- Shows pruning behavior (0 vs 1 gates)

### 2. Accuracy vs Sparsity Trade-off
- Shows performance vs compression

### 3. Lambda vs Sparsity Curve
- Helps tune λ parameter

---

## ✅ Conclusion

- Networks can **self-prune**
- L1 gating leads to **automatic sparsity**
- Achieves **efficiency + performance balance**
