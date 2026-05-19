# Face Mask Detection — CNN with CBAM Attention Module

**Binary image classification: faces with and without mask**  
CNN · CBAM Attention · MobileNetV2 · Transfer Learning · Data Augmentation

---

## Overview

This project builds a deep learning pipeline to classify face images into two categories: **with mask** and **without mask**. Three architectures are trained and compared, with a focus on integrating an attention mechanism (CBAM) to improve feature extraction.

| | |
|---|---|
| **Dataset** | [Face Mask Dataset — Kaggle](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset) |
| **Task** | Binary classification (with mask / without mask) |
| **Classes** | `0` = with mask · `1` = without mask |
| **Best model** | MobileNetV2 : Accuracy: 98.68% · F1-Score: 98.68% |

---

## Methodology

### 1. Data Loading & Visualization
- Dataset split into two folders: `with_mask` and `without_mask`
- CSV generation mapping image paths to labels
- Class distribution visualization

### 2. Preprocessing
- Image resizing to 100×100 (CNN) and 224×224 (MobileNetV2)
- Normalization to [0, 1]
- Train / Validation / Test split (80% / 10% / 10%)

### 3. Data Augmentation
- Rotation, zoom, horizontal flip, width/height shift, shear
- Applied only on training data via `ImageDataGenerator`

### 4. Models Trained

**Custom CNN**
- 3 convolutional blocks (Conv2D + MaxPooling)
- Dense layers with Dropout (0.3)
- Sigmoid output for binary classification

**CNN + CBAM Attention Module**
- Same CNN backbone with a channel and spatial attention mechanism (CBAM)
- Better feature focus on relevant facial regions

**MobileNetV2 (Transfer Learning)**
- Pre-trained on ImageNet, top layers frozen
- GlobalAveragePooling + Dense head
- Fine-tuned with Early Stopping

### 5. Evaluation
- Metrics: Accuracy, Precision, Recall, F1-score
- Confusion matrix (heatmap)
- Training/validation loss and accuracy curves

---

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| CNN | 94.44% | 94.58% | 94.44% | 94.45% |
| CNN + CBAM | 94.97% | 95.04% | 94.97% | 94.97% |
| **MobileNetV2** | **98.68%** | **98.68%** | **98.68%** | **98.68%** |

> MobileNetV2 (transfer learning) significantly outperforms both custom CNN architectures, achieving ~98.7% across all metrics on the test set.

---

## Tech Stack

```
TensorFlow / Keras · OpenCV · scikit-learn
pandas · numpy · matplotlib · seaborn · PIL
```

---

## Project Structure

```
Classification-d-Images-Visages-Avec-et-Sans-Masque/
├── README.md
├── pfa-cnn-mask.ipynb
└── data/
    ├── with_mask/
    └── without_mask/
```

---

## Getting Started

```bash
git clone https://github.com/Nada-naffeti/Classification-d-Images-Visages-Avec-et-Sans-Masque.git
cd Classification-d-Images-Visages-Avec-et-Sans-Masque
pip install tensorflow opencv-python scikit-learn matplotlib seaborn pillow pandas numpy
jupyter notebook pfa-cnn-mask.ipynb
```

> The dataset is available on [Kaggle](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset). Download it and place it under `data/`.

---

## Author

**Nada Naffeti** — Data Science & AI Engineering Student @ [ESSAI](https://www.essai.tn/)

[LinkedIn](https://linkedin.com/in/nada-naffeti) · [GitHub](https://github.com/Nada-naffeti)
