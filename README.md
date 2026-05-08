# GPU Computing – KNN & MLP Implementation  
Practical Assignment for BM40A1401 GPU Computing  
LUT University – Computational Engineering  
Authors: **Maham Asif**, **Sebastian Knackstedt**  
Date: 02.04.2026

---

## 📌 Project Overview
This project explores **GPU‑accelerated machine learning** by implementing two algorithms:

1. **k‑Nearest Neighbors (kNN)** using:
   - Custom CUDA kernels (tiled & non‑tiled)
   - CuPy for GPU array operations  
2. **Multilayer Perceptron (MLP)** using:
   - PyTorch  
   - GPU‑accelerated training and inference  

The dataset contains **4000 samples**, each with **7 features**, belonging to **7 classes**.  
A **70/30 train‑validation split** is used for both algorithms.

---

## 🧠 Part 1 — kNN Implementation

### ✔ CPU Version (NumPy)
- Computes full distance matrix using `np.linalg.norm`
- Uses batching to avoid memory overflow  
- Uses `np.argpartition` for efficient k‑nearest selection  
- Majority voting via `np.bincount`

### ✔ GPU Version (CuPy + Custom Kernels)
Two CUDA kernels were implemented:

#### **1. Tiled Kernel**
- Based on shared‑memory matrix multiplication  
- Modified to compute squared Euclidean distance  
- Uses 32×32 tiles  

#### **2. Simple Kernel**
- Loop‑unrolled distance computation  
- Optimized for small feature dimension (d = 7)  
- Faster than tiled version due to low dimensionality  

### ✔ Performance Summary
- GPU version is **significantly faster** than CPU for n > 10 samples  
- Best performance achieved with:
  - **Batch size = 8192 (GPU)**
  - **Batch size = 64 (CPU)**  
- GPU speedup increases linearly with dataset size  

### ✔ Accuracy
Accuracy for different values of k:

| k | Accuracy |
|---|----------|
| 1 | 0.5358   |
| 5 | 0.5058   |
| 9 | 0.5292   |

---

## 🧠 Part 2 — MLP Implementation (PyTorch)

### ✔ Network Architecture
- Input layer: **7 features**  
- Output layer: **7 classes**  
- Configurable hidden layers  
- Activation: **ReLU**  
- Loss: **CrossEntropyLoss**  
- Optimizer: **Adam (lr = 0.01)**  

### ✔ Training Pipeline
1. Forward pass  
2. Loss computation  
3. Backpropagation  
4. Parameter update  
5. Evaluation on test set  

### ✔ Dataset Handling
- Converted to PyTorch tensors  
- Labels converted to 0‑based `int64`  
- Train/test split: **70/30**

### ✔ Results
- Model successfully trained on GPU  
- Accuracy tracked across epochs  
- Stable convergence observed  

---
