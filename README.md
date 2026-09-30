# 🚗 AI-Based Automatic Number Plate Recognition System

An end-to-end Computer Vision system for detecting and recognizing vehicle license plates from images and videos.

The system uses YOLO11 for license plate detection and OCR for extracting the characters from detected plates.

---

## 📌 Project Overview

This project implements an Automatic Number Plate Recognition (ANPR) pipeline:

Vehicle Image/Video
        ↓
License Plate Detection
        ↓
License Plate Cropping
        ↓
Image Preprocessing
        ↓
OCR
        ↓
Character Recognition
        ↓
Number Plate Result

Example:

JH01AB1234

---

## 🚀 Features

- License plate detection
- YOLO11-based object detection
- GPU accelerated training
- License plate cropping
- Image preprocessing
- OCR-based character recognition
- Video-based detection
- Real-time webcam support
- Multi-frame OCR
- Vehicle/plate tracking
- Indian license plate validation
- Detection confidence
- CSV/database logging

---

## 🧠 Technologies Used

- Python
- YOLO11
- Ultralytics
- OpenCV
- PaddleOCR
- PyTorch
- CUDA
- NumPy

---

## 📂 Project Structure

```text
number_plate_project/
│
├── dataset/
│
├── scripts/
│   ├── train_detector.py
│   ├── test_detector.py
│   ├── anpr_image.py
│   ├── anpr_video.py
│   ├── preprocess.py
│   └── clean_text.py
│
├── models/
│
├── outputs/
│
├── runs/
│
├── data.yaml
├── requirements.txt
├── .gitignore
└── README.md