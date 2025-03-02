# VR-Assignment1
# CODE 2
# Image Stitching using ORB and Homography

## Overview
This project implements an **image stitching pipeline** that takes **overlapping images** and merges them into a **panorama** using **ORB (Oriented FAST and Rotated BRIEF) keypoint detection** and **homography transformation**. The stitching is performed sequentially, aligning each image with its neighboring image.

## Features
- **Keypoint Detection**: Uses ORB to detect and extract feature descriptors.
- **Feature Matching**: Matches keypoints between consecutive images using FLANN-based matcher with Lowe’s ratio test.
- **Homography Estimation**: Computes transformation matrices to warp images into alignment.
- **Seamless Blending**: Blends overlapping regions smoothly to create a unified panorama.
- **Automatic Error Handling**: Skips image pairs if insufficient matches are found.

## Prerequisites
Ensure you have the following dependencies installed:

```bash
pip install opencv-python numpy matplotlib pillow
```

## Usage
1. **Prepare Input Images**:
   - Provide **four overlapping images** named as:
     - `cropped_1.jpg`
     - `cropped_2.jpg`
     - `cropped_3.jpg`
     - `cropped_4.jpg`
   - The images should have a **30-40% overlap** for better alignment.

2. **Run the Script**:
   ```bash
   python image_stitching.py
   ```

3. **View the Output**:
   - The stitched panorama will be saved as `stitched_output.jpg`.
   - It will also be displayed using Matplotlib.

## Code Explanation
1. **Loading Images**: Reads the images and converts them to grayscale.
2. **Keypoint Detection**: Uses ORB to extract feature points.
3. **Feature Matching**: Uses FLANN-based matching and Lowe’s ratio test to find corresponding points.
4. **Homography Calculation**: Aligns images using perspective transformation.
5. **Warping & Blending**: Places each transformed image on a large canvas.
6. **Saving & Displaying**: The final panorama is saved and shown.

## Troubleshooting
- If the output image is distorted or misaligned, ensure that:
  - The images **overlap sufficiently**.
  - The images **have distinct features** (e.g., edges, textures).
  - The input images **are not too blurry**.
# CODE 1
# Edge Detection, Segmentation, and Coin Counting

## Overview

This project implements edge detection, image segmentation, and coin counting using computer vision techniques. The goal is to analyze an input image containing coins, detect the edges, segment the coins, and accurately count them. This is achieved using OpenCV and Python-based image processing methods.

## Dependencies

To run this project, ensure you have the following dependencies installed:

```bash
pip install opencv-python numpy matplotlib
```

- **OpenCV** (`cv2`): Used for image processing and transformations.
- **NumPy**: For numerical computations.
- **Matplotlib**: For visualization of results.

## Methodology

### 1. **Preprocessing the Image**

- Convert the input image to grayscale.
- Apply Gaussian Blur to reduce noise and smooth the image.

### 2. **Edge Detection**

- Use the Canny Edge Detection algorithm to highlight edges in the image.

### 3. **Segmentation**

- Apply thresholding to create a binary image.
- Use morphological operations (dilation and erosion) to enhance segmented objects.

### 4. **Contour Detection and Coin Counting**

- Detect contours in the thresholded image.
- Filter contours based on size and shape.
- Count the number of detected coins.


## Observations

- Edge detection using Canny works well in highlighting the coin boundaries.
- Thresholding effectively segments the coins from the background.
- Morphological operations help refine the segmentation.
- Contour detection accurately identifies individual coins.

## Results

- The program successfully detects and counts the coins in the given image.
- It works efficiently on images with clear contrast and minimal noise.
- Accuracy may be affected by overlapping coins or uneven lighting.

## Inferences

- Preprocessing steps such as smoothing and thresholding significantly impact detection accuracy.
- Well-defined contours lead to more precise coin counting.
- Future improvements could include deep learning-based object detection for more robust performance.

## Usage Instructions

1. Place your image file (`coins.jpg`) in the project directory.
2. Run the script: `python coin_counter.py`
3. The total number of coins will be printed, and detected coins will be displayed.

## Future Enhancements

- Implement adaptive thresholding for better segmentation under varying lighting conditions.
- Use machine learning models for more robust object detection.
- Develop a GUI for interactive coin detection.





