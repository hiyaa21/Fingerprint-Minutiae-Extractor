# Automated Fingerprint Minutiae Extractor

## Project Overview
This project is an automated biometric system designed to extract Ridge Endings and Bifurcations from raw fingerprint images. It utilizes a robust image processing pipeline including Histogram Equalization, Adaptive Thresholding, Morphological Thinning, and a multi-stage pruning process to remove noise artifacts like islands, spurs, and clusters.

## Features
* **Enhancement:** Improves low-contrast synthetic prints.
* **Skeletonization:** Uses the Zhang-Suen thinning algorithm.
* **Smart Pruning:**
    1.  **Island Removal:** Removes disconnected components.
    2.  **Spur Pruning:** Removes false minutiae caused by ridge hairs.
    3.  **Cluster Pruning:** Removes false bifurcation clusters caused by blotchy ridges.

## Dataset
The project uses the **FVC2000 DB4_B** dataset.
You can download it here: [Kaggle Link](https://www.kaggle.com/datasets/peace1019/fingerprint-dataset-for-fvc2000-db4-b)

## How to Run
1.  Download the dataset and place the `real_data` and `train_data` folders inside a folder named `dataset`.
2.  Run the batch extraction script to process all images:
    ```bash
    python extract_skeleton.py
    ```
3.  Run the demo script to visualize the steps on a single image:
    ```bash
    python demo_one_image.py
    ```
4.  Run the Canny demo to see why edge detection fails:
    ```bash
    python canny_demonstration.py
    ```

## Files Included
* `extract_skeleton.py`: Main batch processing script.
* `demo_one_image.py`: Visualization tool for presentations.
* `canny_demonstration.py`: Proof of why Canny edge detection is unsuitable.
* `comparison_matrix.csv`: Results data comparing raw vs. pruned minutiae counts.
