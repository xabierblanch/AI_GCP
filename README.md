# Sub-pixel GCP Detection with Keypoint R-CNN

This repository contains the code and models developed for the study:

**“Sub-pixel automatic detection of GCPs coordinates in time-lapse images using a Deep Learning keypoint network”**  
_Xabier Blanch, Almut Jäschke, Melanie Elias, and Anette Eltner_

---

## 🧠 Overview

This project implements a modified **Keypoint R-CNN** (with **ResNet50-FPN** backbone) for the **automatic and sub-pixel accurate detection of Ground Control Points (GCPs)** in long-term time-lapse image series.  

Unlike other approaches, this is an **end-to-end method**: no pre- or post-processing is required, and no assumptions are made on GCP geometry, size, or color.

<p align="center">
  <img width="3780" height="1867" alt="taludes-01" src="https://github.com/user-attachments/assets/74f64e12-3dae-478c-90f1-89e9ac5c2769" />
</p>

📖 More figures and detailed explanations can be found in the publication:

**IEEE Transactions on Geoscience and Remote Sensing**  

👉 [https://doi.org/10.1109/TGRS.2024.3514854](https://doi.org/10.1109/TGRS.2024.3514854)

---

## ⚙️ Code Features

- Modified **Keypoint R-CNN** to detect a **single keypoint** per object (instead of 17 human pose landmarks)  
- Compatible with standard PyTorch + TorchVision APIs  
- Custom data augmentation using **Albumentations**  
- Sub-pixel accuracy evaluation using:  
  - `mDist_pix`: median Euclidean distance in pixels  
  - `mDist_%`: pixel distance normalized by GCP size
    
---
## 📷 Input Data Requirements

The input data must be an image with at least one visible GCP (Ground Control Point). Images can be:
- **Complete images**: Full resolution images
- **Image patches**: Cropped sections of larger images

In the original publication, **512x512 image patches** were consistently used. The performance with other dimensions or aspect ratios is unknown, making proper input data management essential before training.

⚠️ **Important**: Due to the unknown behavior with different dimensions, careful data preprocessing and validation is required to ensure optimal model performance.

---

## 🏷️ Annotations Format

Each training or validation image must be accompanied by a `.json` file with the same name (excluding extension), containing its annotations.

### Required Structure

```json
{
  "bboxes": [
    ["x_min", "y_min", "x_max", "y_max"]
  ],
  "keypoints": [
    [["x", "y", "visibility"]]
  ]
}
```

### Field Descriptions

- **bboxes**: list of bounding boxes in `[x_min, y_min, x_max, y_max]` format
- **keypoints**: list of keypoints in `[x, y, visibility]` format (1 keypoint per bbox)
- **visibility**: should be `1` (visible), following COCO convention

### Important Notes

📄 See `sample.json` for a working example.

For every image or image patch used in training or validation, there must be a corresponding `.json` annotation file with this structure.

---

## ⚠️ TorchVision Modifications

This project uses a **modified version of the Keypoint R-CNN implementation** from the TorchVision library. Specifically:

- The number of keypoints is reduced from **17 (COCO)** to **1**
- The **evaluation metrics** and **head architecture** have been adapted


> ⚠️ Make sure to use the **provided versions** of these files, or results may not be reproducible.

---

## 🧪 Evaluation

The model was evaluated on all three datasets using both **absolute** and **relative** metrics.  
Results show **median errors below 1 pixel** across datasets, with **robust performance** under adverse conditions (e.g. blur, occlusion, low light).

<img width="3683" height="2665" alt="taludes-02" src="https://github.com/user-attachments/assets/f4d52216-c364-4de5-873f-9874ba170c82" />

---

## 📄 Citation

If you use this repository, please cite the article:

> Blanch, X., Jäschke, A., Elias, M., & Eltner, A.  
> *Sub-pixel automatic detection of GCPs coordinates in time-lapse images using a Deep Learning keypoint network* (submitted).  
> DOI: [https://doi.org/10.1109/TGRS.2024.3514854](https://doi.org/10.1109/TGRS.2024.3514854)

---

## 🙏 Acknowledgments

This research was funded by:

- The **KIWA project** (BMBF, grant no. 13N15542)  
- The **German Research Foundation (DFG)** under project no. 405774238

Special thanks to all collaborators who contributed datasets and ground truth labels.
