---

# 🌿 Leaffliction

**Leaffliction** is an innovative computer vision project focused on **leaf image analysis** for plant disease recognition.

## Overview

This project is built with **Python** and **PyTorch**, leveraging deep learning to train models that classify and detect diseases in leaf images.

## Requirements

* **Python**: 3.12.x (latest stable)
* **PyTorch**: 2.8.0 (with CUDA 12.9 support)

### Install Dependencies

1. Install [Python](https://www.python.org/downloads/)
2. Install [PyTorch](https://pytorch.org/get-started/locally/)
3. Install [CUDA 12.9](https://developer.nvidia.com/cuda-12-9-0-download-archive) (if using GPU)
4. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## ⚙️ Usage with `make`

The project includes several Makefile commands to streamline workflow:

* `make distribution` → Analyze the dataset
* `make augmentation` → Apply data augmentation (flip, rotate, skew, shear, crop, distort)
* `make transformation` → Extract features from images
* `make train` → Train the model on the processed dataset
* `make` → Run all of the above steps in sequence

## Run Scripts Manually

Alternatively, you can run individual scripts:

```bash
# Dataset analysis
./Distribution.py [dataset_dir]

# Data augmentation
./Augmentation.py [image_path]

# Feature extraction
./Transformation.py -h

# Train the model
./train.py [dataset_dir]

# Predict on a single image
./predict.py [image_path]
```

---
