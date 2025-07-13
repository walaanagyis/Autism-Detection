
---

## 🔄 Data Preprocessing and  Splitting (`data_preprocessing.py`)

- **Feature Extraction:** Uses ResNet to extract embeddings from images.
- **Leakage Detection:** Applies cosine similarity to detect potential data overlap.
- **K-means Clustering:** Clusters similar images and splits them by cluster IDs.
- **Manual Reassignment:** Reassigns highly similar test images to the training set if needed.
- **Final Verification:** Ensures training and test sets are statistically distinct (based on similarity threshold).

---

## 🧠 Main ViT Model (`Script_main_model.py`)

- Implements a standard **Vision Transformer (ViT)** architecture.
- Trained on cleanly split datasets.
- Focused on baseline evaluation for classification tasks.

---

## 🧬 WSO-Optimized ViT (`script2_Optimized.py`)

- Applies **Whale Swarm Optimization (WSO)** to:
  - Tune ViT hyperparameters (learning rate, filter count, batch size).
  - Improve model generalization and reduce error.
- Includes grid and Bayesian optimization for comparison.

---

## ✅ Highlights

- Prevents **data leakage** via clustering and similarity-based reassignment.
- Demonstrates **ViT + bioinspired optimization** in practice.
- Provides **reproducible and scalable** preprocessing for computer vision datasets.

---

## 📌 Requirements

- Python ≥ 3.8  
- `torch`, `sklearn`, `opencv-python`, `numpy`, `matplotlib`, `scipy`  
- Optional: `tqdm`, `kornia`, `transformers` (for ViT)

---

## 💡 Suggested Use Cases

- Medical imaging classification  
- Object detection pipelines  
- Any image-based ML task that needs strict data separation  

---


