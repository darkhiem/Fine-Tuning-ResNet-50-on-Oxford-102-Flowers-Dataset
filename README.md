# 🌸 Fine-Tuning ResNet-50 on Oxford 102 Flowers | Fellowship AI Challenge

This repository contains my solution for the **Fellowship AI Computer Vision Challenge**, where I fine-tuned a pre-trained ResNet-50 model on the [Oxford 102 Flowers dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/) to classify 102 categories of flowers.

---

## 🚀 Project Overview

The Oxford 102 Flowers dataset contains 8,189 images of flowers spanning 102 categories. The images are not sorted into folders by class; instead, their labels are provided in separate `.mat` files. The project involves:
- Processing and aligning images with their labels
- Fine-tuning a pre-trained ResNet-50 model
- Visualizing training and validation performance
- Evaluating model accuracy and F1-score

---

## 🧠 What I Learned

- How to structure a computer vision pipeline using PyTorch
- Handling `.mat` files and custom datasets
- Transfer learning using pretrained models (ResNet-50)
- Building a flexible training-validation loop
- Calculating and visualizing accuracy, loss, and F1-score

---

## 🛠️ Tech Stack

- Python 3.10+
- PyTorch
- torchvision
- NumPy, SciPy
- Matplotlib

---

## 📁 Dataset Preparation

1. Download the dataset from the official [Oxford VGG site](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/).
2. Place the `.mat` files and the `jpg/` folder inside a directory named `flowers/`:
    ```
    flowers/
    ├── jpg/
    │   ├── image_00001.jpg
    │   ├── ...
    ├── imagelabels.mat
    ├── setid.mat
    ```
