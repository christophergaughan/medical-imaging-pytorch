# Medical Imaging with PyTorch

End-to-end deep learning pipelines for medical image analysis — from preprocessing to production-ready segmentation models.

## Overview

This repository demonstrates practical medical imaging workflows covering different modalities, tasks, and clinical applications. Built with PyTorch and PyTorch Lightning on Google Colab.

| Notebook | Task | Modality | Key Techniques | Results |
|----------|------|----------|----------------|---------|
| `04-Pneumonia-Classification` | Binary classification | Chest X-ray (DICOM) | ResNet18, transfer learning, CAM | 86% accuracy |
| `06-Atrium-Segmentation` | 2D Segmentation | Cardiac MRI (NIfTI) | U-Net, Dice loss | ~95% Dice |
| `07-Lung-Tumor-Segmentation` | 2D Segmentation | Chest CT (NIfTI) | Weighted sampling, BCE loss | ✓ |
| `08-3D-Liver-Tumor-Segmentation` | **3D Segmentation** | Abdominal CT (NIfTI) | **3D U-Net, TorchIO, multi-class** | **83% Val Dice** |

---

## Pneumonia Detection

Binary classifier for detecting pneumonia from chest X-rays.

**Dataset:** [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) (~26,000 images)

**Clinical relevance:**
- Pneumonia causes >15% of deaths in children under 5 globally
- 920,000 children under 5 died from pneumonia in 2015
- Rapid, accurate diagnosis is critical for treatment

**Highlights:**
- DICOM preprocessing pipeline
- Transfer learning from ImageNet (ResNet18)
- Data augmentation for medical images
- Class Activation Maps (CAM) for model interpretability

**Results:** ~86% validation accuracy

---

## Left Atrium Segmentation

Pixel-wise segmentation of the left atrium from cardiac MRI.

**Dataset:** [Medical Segmentation Decathlon - Task02_Heart](http://medicaldecathlon.com/)

**Clinical relevance:**
- Atrial fibrillation ablation planning
- Left atrial volume measurement (cardiovascular risk factor)

**Highlights:**
- U-Net encoder-decoder architecture
- Dice loss for segmentation
- Albumentations for synchronized image-mask augmentation
- 3D volume visualization

---

## Lung Tumor Segmentation

Segmentation of lung tumors from CT scans — addressing severe class imbalance.

**Dataset:** [Medical Segmentation Decathlon - Task06_Lung](http://medicaldecathlon.com/)

**Clinical relevance:**
- Lung cancer is the leading cause of cancer death worldwide
- Automated segmentation assists radiologists in treatment planning

**Highlights:**
- CT normalization (Hounsfield units)
- **WeightedRandomSampler** to handle extreme class imbalance (<1% tumor pixels)
- BCE loss (more stable than Dice for tiny objects)
- 3D volume prediction and visualization

---

## 3D Liver & Tumor Segmentation ⭐

**The capstone project** — true 3D volumetric segmentation with multi-class output.

**Dataset:** [Medical Segmentation Decathlon - Task03_Liver](http://medicaldecathlon.com/) (resampled to 256×256×Z)

**Clinical relevance:**
- Liver cancer is the 6th most common cancer worldwide
- Surgical planning for tumor resection
- Radiation therapy targeting
- Liver volumetry for transplant assessment

**What makes this different:**

| Previous Notebooks | This Notebook |
|-------------------|---------------|
| 2D convolutions (Conv2d) | 3D convolutions (Conv3d) |
| Binary segmentation | Multi-class (background/liver/tumor) |
| Standard augmentation | TorchIO for 3D transforms |
| Full images | Patch-based training (128³) |

**Highlights:**
- **3D U-Net** architecture with trilinear upsampling
- **Combined loss:** Cross-Entropy + Dice
- **TorchIO** for 3D medical image handling
- Patch-based training for memory efficiency
- Mixed precision (float16) on A100 GPU

**Results:**
- Training Dice: 93.0%
- Validation Dice: 83.1%
- Liver Dice: 90.0%

---

## Technical Stack

```
pytorch
pytorch-lightning
torchvision
torchio           # 3D medical imaging
nibabel           # NIfTI format
pydicom           # DICOM format
albumentations    # 2D augmentation
numpy
matplotlib
```

## Setup

```bash
pip install torch torchvision pytorch-lightning torchio nibabel pydicom albumentations
```

## Key Concepts Covered

| Concept | Where |
|---------|-------|
| DICOM vs NIfTI formats | Pneumonia vs Atrium/Lung/Liver |
| CT vs MRI normalization | Hounsfield units vs z-score |
| Transfer learning | Pneumonia (ResNet18) |
| U-Net architecture | Atrium, Lung, Liver |
| 2D vs 3D convolutions | Lung (2D) vs Liver (3D) |
| Dice vs BCE vs Combined loss | Task-dependent selection |
| Class imbalance handling | WeightedRandomSampler, Dice loss |
| Patch-based training | Liver (memory optimization) |
| Model interpretability | CAM (Pneumonia) |

## Lessons Learned

1. **Data matters more than models** — Preprocessing, augmentation, and class balancing are critical
2. **Domain knowledge > generic ML** — Understanding Hounsfield units, anatomical priors, and clinical metrics
3. **Loss function selection is task-dependent** — BCE for tiny objects, Dice for imbalanced segmentation, Combined for multi-class
4. **3D is expensive** — Patch-based training, mixed precision, and memory optimization are essential
5. **TorchIO APIs change** — Custom Dataset classes are more portable than Queue-based pipelines

## Acknowledgments

- Datasets: [Medical Segmentation Decathlon](http://medicaldecathlon.com/), [RSNA](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)
- Course: Pierian Data - AI for Medical Imaging (but greatly retoucheed and modernized for PyTorch 2.0+, Lightning 2.0+, TorchIO 0.21, plus we perfomed analyses on **full data volume** for RSNA-pneumona competition from `Kaggle`)

## Author

Christopher Gaughan, PhD  
(https://github.com/christophergaughan)

---

*Built with PyTorch and PyTorch Lightning on Google Colab (A100 GPU).*
