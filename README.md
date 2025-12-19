# Medical Imaging with PyTorch

End-to-end deep learning pipelines for medical image analysis — from preprocessing to deployment-ready models.

## Overview

This repository demonstrates practical medical imaging workflows covering different modalities, tasks, and clinical applications.

| Notebook | Task | Modality | Key Techniques |
|----------|------|----------|----------------|
| `03-Preprocessing-Enhanced` | Image preprocessing | Brain MRI, Lung CT, Cardiac MRI (NIfTI) | Affine matrices, orientation, CT vs MRI normalization |
| `04-Pneumonia-Classification` | Binary classification | Chest X-ray (DICOM) | ResNet18, transfer learning, CAM interpretability |
| `06-Atrium-Segmentation` | Semantic segmentation | Cardiac MRI (NIfTI) | U-Net, Dice loss, synchronized augmentation |
| `07-Lung-Tumor-Segmentation` | Semantic segmentation | Chest CT (NIfTI) | U-Net, BCE loss, weighted sampling for class imbalance |

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
- Transfer learning from ImageNet
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

## Technical Stack

```
pytorch
pytorch-lightning
torchvision
nibabel          # NIfTI format
pydicom          # DICOM format
albumentations   # Augmentation
numpy
matplotlib
```

## Setup

```bash
pip install torch torchvision pytorch-lightning nibabel pydicom albumentations numpy matplotlib
```

## Key Concepts Covered

| Concept | Where |
|---------|-------|
| DICOM vs NIfTI | Preprocessing, Pneumonia vs Atrium/Lung |
| CT vs MRI normalization | Preprocessing, Lung vs Atrium |
| Transfer learning | Pneumonia (ResNet18) |
| U-Net architecture | Atrium, Lung Tumor |
| Dice vs BCE loss | Atrium (Dice), Lung (BCE) |
| Class imbalance | Lung Tumor (WeightedRandomSampler) |
| Model interpretability | Pneumonia (CAM) |
| 3D medical volumes | Atrium, Lung Tumor |

## Acknowledgments

- Datasets: [Medical Segmentation Decathlon](http://medicaldecathlon.com/), [RSNA](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)
- Course: Pierian Data - AI for Medical Imaging

## Author

Christopher Gaughan, PhD  
[AntibodyML Consulting LLC](https://github.com/christophergaughan)

---

*Built with PyTorch and PyTorch Lightning on Google Colab.*
