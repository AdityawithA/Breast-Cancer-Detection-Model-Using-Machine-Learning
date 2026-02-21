# 🏥 AI-Based Breast Cancer Detection System

An end-to-end Machine Learning web application that predicts whether a breast tumor is **Benign** or **Malignant** using 30 medical diagnostic features from the Wisconsin Breast Cancer Dataset.

---

## 🚀 Project Overview

This project implements a Random Forest classifier trained on structured medical data to assist in breast cancer classification. The model achieves approximately **95–96% accuracy** and is deployed using a Flask-based web application.

The system provides real-time predictions, confidence scores, feature importance visualization, CSV upload support, and downloadable PDF reports.

⚠ **Disclaimer:** This application is for educational and research purposes only and does not replace professional medical consultation.

---

## 🧠 Key Features

- ✅ Random Forest ML Model (≈96% accuracy)
- 📊 Prediction Probability Visualization
- 📈 Top 10 Feature Importance Chart
- 📁 CSV File Upload Support
- 📤 Downloadable PDF Report
- 🎯 Auto-fill Sample Data Button
- 🌙 Dark Mode Toggle
- 🏥 Educational Breast Cancer Information Section
- 💻 Fully Interactive Web Interface

---

## 🛠 Tech Stack

- **Python**
- **Flask**
- **Scikit-learn**
- **NumPy & Pandas**
- **Matplotlib**
- **ReportLab**
- **HTML / CSS / JavaScript**

---

## 📂 Project Structure

Breast-Cancer-App/
│
├── app.py
├── model.pkl
├── requirements.txt
├── README.md
└── templates/
└── index.html


---

## ⚙ Installation & Setup

### 1️⃣ Clone Repository
```bash
git clone <your-repo-link>
cd Breast-Cancer-App

2️⃣ Install Dependencies
pip install -r requirements.txt


If requirements.txt is not created yet:

pip install flask scikit-learn numpy pandas matplotlib reportlab

3️⃣ Run Application
python app.py

Open browser:

http://127.0.0.1:5000/

## 🌐 Live Demo
https://breast-cancer-detection-model-using.onrender.com/

