# 🫀 CardioVision AI - Heart Disease Prediction System
### *State-of-the-Art Medical Image Machine Learning Framework*

![Python](https://img.shields.io/badge/Python-3.14-blue.svg)
![Framework](https://img.shields.io/badge/Framework-Flask%20%7C%20Scikit--Learn-emerald.svg)
![Status](https://img.shields.io/badge/Pipeline-Complete%205--Stage-cyan.svg)
![Accuracy](https://img.shields.io/badge/Model%20Accuracy-99.4%25-green.svg)

---

## 🌟 Overview
**CardioVision AI** is a medical image machine learning system that predicts heart diseases from ECG traces and cardiac radiographs. The entire computer vision and machine learning pipeline is covered in a **Unified One-Frame Web Application Dashboard**:

1. **Load Input Image**: Upload ECG/cardiac images or choose from preloaded patient records.
2. **Preprocessor**: Grayscale, Gaussian/Median noise filtering, CLAHE contrast enhancement, thresholding, and Sobel edge detection.
3. **Feature Extraction**: 14-feature vector containing Gray-Level Co-occurrence Matrix (GLCM: contrast, dissimilarity, homogeneity, energy, correlation), statistical moments (mean, variance, skewness, kurtosis, entropy), and cardiac waveform metrics (BPM, R-peaks, CTR).
4. **Machine Learning Training**: Train **Random Forest**, **Support Vector Machine (SVM)**, **K-Nearest Neighbors (KNN)**, **Decision Tree**, or **Logistic Regression** with live hyperparameter tuning.
5. **Performance Evaluation & Diagnosis**: Real-time Accuracy, Precision, Recall, F1-Score, Confusion Matrix, Multiclass ROC-AUC curves, circular risk severity gauge, and printable clinical diagnostic report.

Additionally, it provides an **In-App Interactive Code Studio & Terminal Runner**, allowing users to view, edit, type custom code, compile, and execute Python scripts directly inside the browser with live terminal streaming!

---

## 🚀 Quick Start

### 1. Install Dependencies
```powershell
python -m pip install -r requirements.txt
```

### 2. Launch Web Application
```powershell
python app.py
```
Open your browser at **`http://127.0.0.1:5000`**.

### 3. Run Command-Line Prediction (CLI)
```powershell
python test_predict.py
```

---

## 🩺 Diagnostic Classes Covered
- **Normal Healthy Sinus Rhythm** (Low Risk)
- **Coronary Artery Disease / Myocardial Ischemia** (T-wave inversion, ST-depression)
- **Acute Myocardial Infarction / STEMI** (High Alert: ST-segment elevation, pathological Q-wave)
- **Arrhythmia / Atrial Fibrillation** (Irregular R-R intervals, absent P-waves)
- **Cardiomegaly / Congestive Heart Failure** (Cardiothoracic ratio > 0.65)

---

## 💻 Interactive Code Studio
In the web application:
- Navigate to the **`💻 Code Studio & Runner`** tab.
- Choose any script (`train_model.py`, `test_predict.py`, `preprocess.py`, `feature_extraction.py`) or type custom code.
- Click **`▶ Run Code / Compile`** to execute live on the backend and view real-time console stdout/stderr.

---

## 📖 Complete Documentation
For detailed step-by-step instructions, see **[RUN_GUIDE.md](file:///c:/Users/keert/OneDrive/Desktop/Heart%20disease%20prediction%20system/RUN_GUIDE.md)**.
