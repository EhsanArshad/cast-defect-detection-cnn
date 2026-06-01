# Cast Defect Detection using CNN 🏭

A deep learning model that automatically detects defects in casting manufacturing 
parts using Convolutional Neural Networks (CNN).

## 📊 Results
| Metric | Value |
|--------|-------|
| Training Accuracy | 96.74% |
| Validation Accuracy | 99.86% |
| Test Images | 715 |
| False Positives | 0 |

## 🏗️ Model Architecture (CDD - Cast Defect Detector)
- 3 Convolutional layers with BatchNorm + ReLU + MaxPool
- Dropout (0.5) for regularization
- Adam optimizer, lr=0.001
- Trained for 15 epochs on T4 GPU

## 📁 Dataset
- Training: 6,633 images (def_front + ok_front)
- Testing: 715 images
- Image size: 128x128 (RGB, 3 channels)

## 🛠️ Tools Used
Python · PyTorch · Google Colab · torchvision · scikit-learn

## 👥 Team - Group 15
- Umair Ali
- Ehsan Arshad (FA23-BBD-036)
- Abdullah Khan (FA23-BBD-006)

COMSATS University Islamabad | MGT388 Deep Learning | Spring 2026
