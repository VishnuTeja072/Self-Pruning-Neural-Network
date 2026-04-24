#  Self-Pruning Neural Network

A deep learning model that **learns to remove its own unnecessary connections** during training using learnable gates and L1 regularization. This results in a **sparse, efficient, and interpretable network** without manual pruning.

---

##  Overview

Traditional neural networks are over-parameterized. This project introduces a **self-pruning mechanism** where each connection is controlled by a **learnable gate**.

- Gates are passed through a **sigmoid function** → values in (0, 1)
- An **L1 penalty** is applied to encourage sparsity
- The model automatically decides:
  - Which weights to keep ✅  
  - Which weights to remove ❌  

---

##  Loss Function

```
Total Loss = CrossEntropyLoss + λ · Σ sigmoid(gate_scores)
```

###  Key Idea

- **Sigmoid(gate_scores)** → converts raw scores into probabilities
- **L1 penalty** → pushes gate values toward 0
- Final effect:
  - Important connections → gate ≈ 1  
  - Unimportant connections → gate ≈ 0  

---

##  Why L1 Encourages Sparsity

| Penalty | Behavior |
|--------|---------|
| L2 (gate²) | Gradually shrinks values, rarely reaches 0 |
| L1 (|gate|) | Constant push → drives values exactly to 0 |

---

##  Results

| Lambda (λ) | Test Accuracy | Sparsity (%) |
|:----------:|:-------------:|:------------:|
| 1e-5       | ~52%          | ~15%         |
| 1e-4       | ~49%          | ~55%         |
| 1e-3       | ~40%          | ~88%         |

---

##  Gate Distribution Insight

- Peak near **0** → pruned weights  
- Peak near **1** → important weights  

---

##  Design Choices

| Component | Choice | Reason |
|----------|--------|--------|
| Gate Activation | Sigmoid | Smooth & differentiable |
| Initialization | Zeros (→ 0.5 after sigmoid) | Neutral start |
| Regularization | L1 penalty | Strong sparsity enforcement |
| Optimizer | Adam + Cosine LR | Stable training |
| Threshold | 0.01 | Defines effective pruning |

---

##  How to Run

```
pip install torch torchvision matplotlib numpy
python self_pruning_nn.py
```

---

## 📊 Graphs & Visualizations

### 1. Gate Distribution Histogram

<img width="862" height="436" alt="Bimodal" src="https://github.com/user-attachments/assets/5c0650bb-ec13-4570-8ec4-a689c9fa947c" />


<img width="503" height="632" alt="Weight distribution" src="https://github.com/user-attachments/assets/6d05d958-0087-43cd-ac6b-82d12c5b80c1" />


<img width="735" height="560" alt="Sigmoid Function" src="https://github.com/user-attachments/assets/79b7ff1c-cc67-438a-a599-5f2b12cbf4ea" />


<img width="1171" height="521" alt="Modality" src="https://github.com/user-attachments/assets/27c14ec2-3741-48dc-9f5a-de66d1ceb48a" />


### 2. Accuracy vs Sparsity Trade-off
<img width="761" height="618" alt="Accuracy vs Sparsity" src="https://github.com/user-attachments/assets/0da33843-498c-49c3-9d12-54f9fdcfef35" />



### 3. Lambda vs Sparsity Curve
<img width="1138" height="618" alt="Parameters Pruned" src="https://github.com/user-attachments/assets/ee2fa7d4-47c4-4a29-816a-659665f9a7c8" />


<img width="957" height="617" alt="Comparision L1 and L2" src="https://github.com/user-attachments/assets/59d50c0c-2b0d-474b-8ce6-6d802507be46" />


<img width="727" height="553" alt="L1, L2 and Elastic net" src="https://github.com/user-attachments/assets/6cb5e144-5c15-4e18-b23d-23618d549500" />


<img width="1283" height="635" alt="L1 Regularization vs L2 Regularization" src="https://github.com/user-attachments/assets/b13e3aae-6b0d-4e65-b148-b60173361823" />

---

## ✅ Conclusion

- Networks can **self-prune**
- L1 gating leads to **automatic sparsity**
- Achieves **efficiency + performance balance**
