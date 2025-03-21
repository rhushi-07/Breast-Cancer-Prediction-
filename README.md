# Breast Cancer Prediction using Machine Learning

## 📌 Project Overview
This project is a **Breast Cancer Prediction System** using multiple machine learning models. The system is designed to classify breast cancer cases based on given features and provides predictions using various trained models, including **Random Forest, SVM, KNN, and XGBoost**.

## 🚀 Features
- Uses multiple machine learning models for prediction.
- Trained on a large dataset for high accuracy.
- Supports **Random Forest, SVM, KNN, and XGBoost**.
- Scalable and modular design.
- Implemented with **Python, Scikit-learn, XGBoost, and Pandas**.
- Handles large models using **Git LFS**.

## 📁 Project Structure
```
Breast-Cancer-Prediction/
│-- models/                  # Pre-trained models (tracked via Git LFS)
│   ├── RF_model600mb.pkl    # Random Forest Model
│   ├── svm_model.pkl        # SVM Model
│   ├── knn_model.pkl        # KNN Model
│   ├── xgboost_model.pkl    # XGBoost Model
│   ├── scaler.pkl           # Standard Scaler for preprocessing
│-- dataset/                 # Dataset used for training
│-- src/                     # Source code for model training & prediction
│   ├── train.py             # Script to train models
│   ├── predict.py           # Script to make predictions
│-- requirements.txt         # Dependencies
│-- README.md                # Project Documentation
```

## 🔧 Installation & Setup
1. **Clone the Repository**
   ```sh
   git clone https://github.com/your-username/Breast-Cancer-Prediction.git
   cd Breast-Cancer-Prediction
   ```

2. **Install Dependencies**
   ```sh
   pip install -r requirements.txt
   ```

3. **Download Models (Git LFS)**
   ```sh
   git lfs install
   git lfs pull
   ```

## 📊 How to Use
1. **Run Predictions**
   ```sh
   python src/predict.py --input data/sample_input.csv
   ```

2. **Train New Models**
   ```sh
   python src/train.py --dataset dataset/breast_cancer.csv
   ```

## 🤝 Contributing
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-branch`).
3. Commit changes (`git commit -m 'Added new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a **Pull Request**.

## 📜 License
This project is licensed under the **MIT License**.

---
💡 **Maintainers:** Rhushikes Gulave and Contributors.

