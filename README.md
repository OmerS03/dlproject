---
title: Parking Spot Occupancy Detection
emoji: 🚗
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# Parking Spot Occupancy Detection (CNN)

Simple image classification project that detects whether a parking spot is empty or occupied using a custom CNN built with PyTorch.

## Dataset Structure

Place your images in the following folder layout:

```
dataset/
  train/
    empty/
    occupied/
  val/
    empty/
    occupied/
```

## How the Model Works

The model is a small CNN with three convolution blocks (Conv2D + ReLU + MaxPool) followed by fully connected layers to output two classes: empty and occupied.

## Run Locally

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Train the model:
   ```
   python train.py
   ```
3. Run the Gradio app:
   ```
   python app.py
   ```

## Deploy on Hugging Face Spaces

1. Create a new Space with the Gradio SDK.
2. Upload these files: `app.py`, `model.py`, `requirements.txt`, and `parking_model.pth`.
3. Ensure your trained weights file is named `parking_model.pth`.
4. The Space will launch automatically.
