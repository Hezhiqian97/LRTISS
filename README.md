# **LRTIIS: Efficient Semantic Segmentation Network**

This is a semantic segmentation project based on PyTorch, featuring **LRTIIS** as the core model. The project integrates **StarNet** as the backbone and utilizes **MRFA** (Multi-Resolution Feature Aggregation) and **LEGR** (Local Global Element-wise Routing) modules for feature extraction and fusion. Additionally, a **Dynamic Hybrid Loss** function (combining Cross Entropy Loss and Adaptive Gaussian Dice Loss) is employed during training to enhance performance in tasks such as defect detection.

## **Table of Contents**

* [Environment Requirements](https://www.google.com/search?q=%23environment-requirements)  
* [File Structure](https://www.google.com/search?q=%23file-structure)  
* [Dataset Preparation](https://www.google.com/search?q=%23dataset-preparation)  
* [Model Training](https://www.google.com/search?q=%23model-training)  
* [Model Prediction](https://www.google.com/search?q=%23model-prediction)  
* [Model Evaluation](https://www.google.com/search?q=%23model-evaluation)  
* [Visualization (CAM)](https://www.google.com/search?q=%23visualization-cam)  
* [Model Architecture](https://www.google.com/search?q=%23model-architecture)

## **Environment Requirements**

Please ensure the following core libraries are installed:

* Python 3.8+  
* PyTorch 1.7+ (Recommended 1.10+)  
* Torchvision  
* Numpy  
* Pillow  
* Matplotlib  
* Tqdm  
* Tensorboard  
* Thop (For FLOPs calculation)  
* Torchstat (Optional, for model statistics)  
* Grad-CAM (For visualization: pip install grad-cam)

Installation example:

pip install torch torchvision numpy pillow matplotlib tqdm tensorboard thop grad-cam

## **File Structure**

├── nets/  
│   ├── LRTIIS.py          \# LRTIIS Model Definition  
│   ├── starnet.py         \# Backbone (StarNet), MRFA, LEGR Definitions  
│   ├── unet\_training.py   \# Loss Functions & Weights Initialization  
│   └── ...  
├── utils/  
│   ├── callbacks.py       \# Training Callbacks (Eval, LossHistory)  
│   ├── dataloader.py      \# Data Loader  
│   ├── utils\_fit.py       \# Single Epoch Training Logic  
│   ├── utils\_metrics.py   \# Evaluation Metrics (mIoU, F-score, etc.)  
│   └── ...  
├── train.py               \# Training Script  
├── predict.py             \# Prediction Script (Single/Batch/FPS Test)  
├── get\_miou.py            \# Accuracy Evaluation Script  
├── get\_cam.py             \# Class Activation Map Visualization Script  
├── summary.py             \# Model Parameters & FLOPs Analysis  
└── voc\_annotation.py      \# Dataset Split Script

## **Dataset Preparation**

This project uses the **VOC Format** for data management.

1. Data Placement:  
   Place your dataset inside the VOCdevkit folder (or specify the path in the script) with the following structure:  
   VOCdevkit/  
   └── VOC2007/  
       ├── JPEGImages/      \# Original Images (.jpg)  
       └── SegmentationClass/ \# Label Images (.png, 8-bit grayscale or palette)

2. Generate Index Files:  
   Run voc\_annotation.py to split the dataset into training, validation, and test sets.  
   python voc\_annotation.py \--voc\_path path/to/VOCdevkit \--trainval\_percent 1.0 \--train\_percent 0.9

   This will generate train.txt, val.txt, and test.txt in VOC2007/ImageSets/Segmentation/.

## **Model Training**

Use train.py for model training. This script supports command-line configuration.

**Basic Usage:**

python train.py \--cuda \--num-classes 4 \--batch-size 6 \--save-dir logs/experiment1

**Key Arguments:**

* \--num-classes: Number of classes (including background).  
* \--input-shape: Input image size, e.g., 224 224 or 512 512\.  
* \--dice-loss: Enable Adaptive Gaussian Dice Loss (Default: True).  
* \--model-path: Path to pretrained model weights (Optional).  
* \--unfreeze-epoch: Total number of training epochs.  
* \--save-period: Epoch interval for saving weights.

Training logs and weights will be saved in the directory specified by \--save-dir.

## **Model Prediction**

Use predict.py for inference. Supports single image prediction, batch prediction, and FPS testing.

**Modes (--mode)**:

* predict: Interactive single image prediction.  
* dir\_predict: Batch prediction for images in a folder.  
* fps: Test model inference speed.

**Example: Batch Prediction**

python predict.py \--mode dir\_predict \\  
    \--dir\_origin\_path img\_test/ \\  
    \--dir\_save\_path img\_out/ \\  
    \--model\_path logs/experiment1/best\_epoch\_weights.pth

## **Model Evaluation**

Use get\_miou.py to calculate metrics such as mIoU, Recall, and Precision.

python get\_miou.py \\  
    \--miou\_mode 0 \\  
    \--num\_classes 4 \\  
    \--voc\_path path/to/VOCdevkit \\  
    \--output\_dir miou\_results

* miou\_mode: 0=Predict & Calculate, 1=Predict Only, 2=Calculate Only (requires existing results).

## **Visualization (CAM)**

Use get\_cam.py to generate Class Activation Maps (CAM) for analyzing model focus areas.

python get\_cam.py \\  
    \--img-root img\_test/ \\  
    \--model-path logs/experiment1/best\_epoch\_weights.pth \\  
    \--save-path cam\_results/ \\  
    \--classes BW HD PF WR

* \--classes: List of class names (space-separated, excluding background, must match training order).  
* This script automatically attempts multiple CAM algorithms (LayerCAM, GradCAM++, etc.) and generates comparison grids.

## **Model Architecture**

The LRTIIS model is designed to be lightweight and efficient, consisting of the following core components:

1. **Backbone (StarNet)**: A minimalist network design without LayerScale or EMA, mapping high-dimensional features via Element-wise Multiplication.  
2. **LEGR (Local Global Element-wise Routing)**: Enhances the interaction between local and global features.  
3. **MRFA (Multi-Resolution Feature Aggregation)**: Effectively fuses semantic information across different scales.