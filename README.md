# 📘 Deep Learning CA-4 Assignment

### 🧠 Subject: Introduction to Deep Learning  
**Name:** Atharva Kale  
**PRN:** 22070521071  
**Section:** C  

---

## 📌 Assignment Overview

This repository contains solutions to **CA-4** for the subject *Introduction to Deep Learning*, covering two practical scenarios:

1. **Handwritten Digit Recognition using MNIST Dataset**
2. **Heart Disease Prediction using Artificial Neural Networks**

Each scenario walks through the deep learning workflow — from data preprocessing to model building, training, evaluation, and improvement strategies.

---

## 🔍 Scenario 1: Handwritten Digit Recognition

### 📄 Problem Statement:
A startup aims to automate expense logging by recognizing handwritten digits on receipts. The task is to build a deep learning model using the **MNIST** dataset (70,000 grayscale images of size 28x28) to classify digits from **0 to 9**.

### 🧩 Steps Involved:
- Import libraries  
- Load & preprocess the MNIST dataset  
- Normalize the image pixel values to [0, 1]  
- Flatten 2D images to 1D arrays  
- One-hot encode labels  
- Build and train a **Feedforward Neural Network (FNN)** using **Keras**  
- Evaluate and improve model accuracy

### 🔧 Model Architecture:
- Input Layer: 784 neurons (28x28 flattened)  
- Hidden Layers: Dense layers with ReLU activation  
- Output Layer: 10 neurons with softmax activation (for multiclass classification)

### ✅ Output:  
Achieved high accuracy (~98%) on test data using basic architecture.

---

## ❤️ Scenario 2: Heart Disease Prediction

### 📄 Problem Statement:
A hospital wants to predict the risk of **heart disease** based on patient data (age, cholesterol, blood pressure, etc.) using a classification model.

### 🧩 Steps Involved:
- Import libraries  
- Load CSV dataset  
- Split into train and test sets  
- Normalize/Scale features  
- Compute and apply **class weights** to address class imbalance  
- Build and train an **Artificial Neural Network (ANN)** using **Keras**

### 🧠 ANN Details:
- Hidden Layers: Dense layers with ReLU activation  
- Output Layer: 1 neuron with **sigmoid** activation (binary classification)

### 🧪 Output:
Model trained to classify patients with or without heart disease and handles imbalanced datasets effectively using `class_weight`.

---

## 🛠️ Tools & Libraries Used
- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib / Seaborn (for optional visualizations)

---

## 📈 Future Improvements

### For MNIST Digit Recognition:
- Add **CNN (Convolutional Neural Network)** for spatial feature extraction  
- Implement **Dropout** and **Batch Normalization**  
- Use **Data Augmentation** to improve generalization

### For Heart Disease Prediction:
- Apply **Hyperparameter Tuning** using GridSearchCV or KerasTuner  
- Add **Feature Engineering** for better performance  
- Use **ROC-AUC**, **Precision-Recall** curves for deeper evaluation

---

## 📂 Repository Structure

```
Deep Learning CA-4/
│
├── Deep Learning CA-4.ipynb     # Jupyter Notebook containing both scenarios
├── README.md                    # Project Documentation (this file)
└── dataset/                     # (Optional) Folder for CSV file, if added
```

---

## 📬 Contact

For any queries or collaboration:
**Atharva Kale**  
📧 [GitHub](https://github.com/) | 🧑‍🎓 PRN: 22070521071 | 🎓 Section C

---

⭐ *If you found this helpful, consider starring the repository!*
