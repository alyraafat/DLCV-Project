

# 🎭 Emotion Recognition from Egyptian Movie Frames

*A Deep Learning for Computer Vision Course Project (Spring 2025)*

This repository contains the full pipeline we built to **classify facial emotions** (Happy, Sad, Angry, Surprised, Neutral) from frames extracted out of *Egyptian movies*. The project was delivered in three milestones:

1. **Milestone 1 – Dataset Creation**
2. **Milestone 2 – Modeling (From‑Scratch CNN + PyTorch CNN + Bonus Experiments)**
3. **Milestone 3 – Final Evaluation (Cross‑Validation, Metrics, Reporting)**

---

## 1️⃣ Milestone 1 – Dataset Construction

We created a **custom dataset** by sampling key frames from Egyptian films.

**Requirements & Process**

* Five emotion classes: *Happy, Sad, Angry, Surprised, Neutral*.
* \~100 high‑quality PNG images per class (single, centered face; no duplicates).
* Frames standardized to **640×480** resolution; both color and grayscale allowed.
* File naming convention: `CLASS_TEAMID_SERIAL.png` (e.g., `HAPPY_T01_0007.png`).
* Manual filtering to remove blurred/near‑duplicate frames.
* Challenges addressed: pose variation, illumination, occlusions.

---

## 2️⃣ Milestone 2 – Modeling

### 2.1 Data Preparation

* Split: **70% train / 20% validation / 10% test**.
* Resize to **512×512×3**, convert BGR→RGB.
* Normalize pixel values to `[0,1]`.
* (For supervised model) On‑the‑fly augmentation: random horizontal/vertical flips, rotation, brightness shifts.

### 2.2 **Model 1: From‑Scratch Convolutional Feature Extractor + Clustering**

We implemented all components **without deep learning libraries**:

* Custom `ConvLayer` supporting predefined or random filter initialization.
* Predefined 3×3 filters stacked into **three convolution blocks** with pooling + simple activation.
* Flatten → downsample to 128‑D feature vector.
* **Fast convolution** via an im2col‑style vectorized matrix multiplication.
* **K-Means clustering** used for unsupervised labeling. Two label‑assignment strategies:

  1. *Mode Assignment* (per‑cluster majority label)
  2. *Centroid Matching* (nearest class mean)
* Result: Low accuracy (\~22–23%) and low silhouette scores, reflecting **strong class overlap**.

### 2.3 **Model 2: Supervised PyTorch CNN**

Architecture (as specified by milestone):

* Five convolutional layers (kernel sizes 3×3,3×3,3×3,5×5,7×7) with filter counts 32,64,64,32,16
* ReLU activations + 2×2 MaxPooling after each block
* Flatten → Fully connected layer (Sigmoid) → Softmax output
* **Hyperparameters:** AdamW, LR=1e‑3, Batch Size=32, CrossEntropyLoss, 20 epochs

**Experiments**

* **Baseline vs. Augmented training:** Augmentation reduced validation loss volatility and improved generalization.
* **Transfer Learning (Bonus):** Fine‑tuned **ResNet‑18**, achieving the best qualitative and quantitative performance.
* **Regularization / BatchNorm:** Added after pooling layers to stabilize training.

---

## 3️⃣ Milestone 3 – Final Evaluation

### 3.1 Cross‑Validation

We merged train+validation and performed **5‑fold CV**:

* **Unsupervised model:** Accuracy ≈21–24% depending on initialization strategy.
* **Supervised CNN:** Mean validation accuracy ≈39% (macro‑F1 ≈0.26), showing consistent learning.

### 3.2 Test Performance

* **From‑Scratch + KMeans:** Test accuracy peaked around \~28% (predicting dominant clusters).
* **Supervised CNN:** Test accuracy ≈38% (higher recall for frequent classes like *Happy*/*Angry*; minority classes underrepresented).
* **ResNet‑18 Fine‑Tuned:** Best overall behavior (higher correct predictions across more classes).

### 3.3 Error Analysis

* Class imbalance (few *Surprised* / *Neutral* samples) led to biased predictions.
* Overlapping facial expressions and actor diversity contributed to low separability.
* Augmentation & transfer learning partially mitigated overfitting.

### 3.4 Pre/Post‑Processing Summary

**Pre:** resize, color conversion, normalization, augmentation
**Post:** logits → argmax → label mapping; for clustering, cluster index → majority label
