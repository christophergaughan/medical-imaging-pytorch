# Medical Imaging with PyTorch

Deep learning pipelines for medical image analysis — from preprocessing to interpretability.

## Overview

This repository contains end-to-end workflows for medical imaging tasks using PyTorch and PyTorch Lightning.

| Notebook | Task | Modality | Key Concepts |
|----------|------|----------|--------------|
| `03-Preprocessing-Enhanced` | Image preprocessing | Brain MRI, Lung CT, Cardiac MRI (NIfTI) | Affine matrices, orientation conventions, CT vs MRI normalization |
| `04-Pneumonia-Classification` | Binary classification | Chest X-ray (DICOM) | ResNet18, transfer learning, CAM interpretability |
| *More coming soon* | | | |

## Pneumonia Detection

Binary classifier trained on the [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) dataset (~26,000 chest X-rays).

**Highlights:**
- DICOM preprocessing pipeline
- Data augmentation for medical images
- Class imbalance handling
- Class Activation Maps (CAM) for model interpretability

**Results:**
- Validation Accuracy: ~86%
- Trained on Google Colab (A100 GPU)

## Setup

```bash
pip install torch torchvision pytorch-lightning torchmetrics pydicom numpy matplotlib
```

## Coming Soon

- 3D medical image preprocessing (NIfTI format)
- Object detection for pneumonia localization
- Cardiac MRI segmentation (U-Net)
- Lung tumor segmentation (3D CT)

## Acknowledgments

- Dataset: RSNA Pneumonia Detection Challenge (Kaggle)
- Course: AI Medical Imaging (Pierian Data)

## Author

Christopher Gaughan, PhD  
(https://github.com/christophergaughan)

---

*Work in progress — more notebooks and documentation coming soon.*
