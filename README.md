# 🐠 Multiclass Fish Image Classification

## 📘 Project Overview
This project focuses on **classifying fish images** into multiple categories using **Deep Learning (CNN)**.  
It demonstrates **data preprocessing, augmentation, model training, evaluation, and deployment** using a **Streamlit web app**.

---

## 🎯 Objective
To develop a robust model capable of identifying the fish species from an image.  
The model is trained from scratch using a **Convolutional Neural Network (CNN)** architecture.

---

## 🧰 Tech Stack
- **Programming Language:** Python  
- **Frameworks:** TensorFlow / Keras  
- **Visualization:** Matplotlib, Seaborn  
- **Deployment:** Streamlit  
- **Metrics:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix  

---

## 🗂️ Dataset
The dataset contains images of different fish species, organized in folders by class:

fish_dataset/
│
├── train/
│ ├── Betta/
│ ├── Goldfish/
│ ├── Guppy/
│ ├── Oscar/
│ └── Tetra/
│
├── val/
│ ├── Betta/
│ ├── Goldfish/
│ ├── Guppy/
│ ├── Oscar/
│ └── Tetra/
│
└── test/
├── Betta/
├── Goldfish/
├── Guppy/
├── Oscar/
└── Tetra/


Each folder contains labeled fish images for training, validation, and testing.

---

## ⚙️ Features
✅ CNN Model from scratch  
✅ Data Augmentation and Preprocessing  
✅ Model Evaluation with Accuracy, Precision, Recall, F1-score  
✅ Confusion Matrix and Training Visualization  
✅ Model Saving (`.h5`) and Class Mapping (`.json`)  
✅ Streamlit App for Real-Time Predictions  

---

## 🧠 Model Architecture

| Layer | Description |
|-------|--------------|
| Conv2D + ReLU | Extracts image features |
| MaxPooling2D | Reduces spatial dimensions |
| Conv2D + ReLU | Deeper feature extraction |
| MaxPooling2D | Further dimensionality reduction |
| Conv2D + ReLU | Complex feature extraction |
| Flatten | Converts feature maps into a vector |
| Dense (512, ReLU) | Fully connected hidden layer |
| Dropout (0.5) | Prevents overfitting |
| Dense (softmax) | Output layer for classification |

---


📊 Model Evaluation
Metric	Score
Accuracy	~88%
Precision	~87%
Recall	~86%
F1-Score	~86%
📈 Visual Results
🧩 Confusion Matrix

🧩 Training & Validation Accuracy

📁 fish-image-classification/
│
├── train_fish_classifier.py    # Model training and evaluation script
├── app.py                      # Streamlit deployment script
├── class_indices.json          # Class label mapping
├── fish_classifier_model.h5    # Saved model
├── confusion_matrix.png        # Confusion matrix visualization
├── training_history.png        # Training history visualization
├── requirements.txt            # Dependencies
└── README.md                   # Documentation


👨‍💻 Author

[Aswathy B]
📧 Email: aswathy.balky@gmail.com

