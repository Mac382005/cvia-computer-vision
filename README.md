# Computer Vision and Image Analysis (CVIA) – University of Canberra, 2026

Two coursework projects from the CVIA unit (Bachelor of Engineering (Honours), AI & Robotics).

| Project | Topic | Dataset | Key result |
|---|---|---|---|
| [Assignment 1](assignment1-image-segmentation/) | Classical image segmentation | Oxford-IIIT Pet (96-image subset) | Canny edge-based segmentation was best overall (global Dice 0.45) |
| [Assignment 2](assignment2-bird-classification/) | Fine-grained bird classification | CUB-200-2011 (200 species) | MobileNetV2 transfer learning reached 60.8% on bounding-box crops vs 8.2% for HOG + SVM |

## Assignment 1: Image segmentation

Notebook: [`image_segmentation.ipynb`](assignment1-image-segmentation/image_segmentation.ipynb)

Compares four unsupervised segmentation methods against ground-truth pet masks, using IoU and Dice
(per image and global):

| Method | Global Dice |
|---|---|
| Edge-based (Canny) | **0.45** |
| K-means (HSV) | 0.41 |
| K-means (RGB) | 0.36 |
| Region-based | 0.24 |

Canny edge segmentation performed best overall; HSV colour space clustered better than RGB.
Predicted masks for every image are in [`results/`](assignment1-image-segmentation/results/).

**Data:** the 96-image subset of the Oxford-IIIT Pet Dataset (Parkhi et al., *Cats and Dogs*, CVPR 2012)
was supplied by the unit and is not redistributed here. To re-run, place it at
`assignment1-image-segmentation/Oxford-IIIT Pet Subset/` with `image/` and `annotation/` folders.

## Assignment 2: Bird species classification

Notebooks:
- [`hog_svm_classical.ipynb`](assignment2-bird-classification/hog_svm_classical.ipynb): HOG features + linear SVM
- [`mobilenetv2_transfer_learning.ipynb`](assignment2-bird-classification/mobilenetv2_transfer_learning.ipynb): MobileNetV2 (ImageNet) transfer learning in TensorFlow/Keras

Four experiments on the official CUB-200-2011 train/test split, each on whole images and on bounding-box crops:

| Experiment | Method | Input | Test accuracy |
|---|---|---|---|
| 1 | HOG + linear SVM | Whole image | 2.7% |
| 2 | MobileNetV2 transfer learning | Whole image | 53.7% |
| 3 | HOG + linear SVM | Bounding-box crop | 8.2% |
| 4 | MobileNetV2 transfer learning | Bounding-box crop | **60.8%** |

A bonus 5-fold cross-validation of HOG + SVM on crops gave 9.4% ± 0.5%. Hand-crafted HOG features
cannot separate 200 visually similar species, while pretrained CNN features can; cropping to the bird
helps both methods. Per-class accuracies and comparison plots are in [`outputs/`](assignment2-bird-classification/outputs/).

**Data:** download [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/) and extract it
so that `assignment2-bird-classification/data/CUB_200_2011/images/` exists. Trained model files
(`.joblib`, `.keras`) are not included because of their size.

## Setup

```bash
pip install -r requirements.txt
jupyter notebook
```

Author: Hoang Nhat Minh Mac
