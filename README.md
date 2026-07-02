# 🎓 Student Performance Prediction using Machine Learning

## 📌 Project Overview

The Student Performance Prediction project uses Machine Learning classification algorithms to predict the Performance Risk Level of students based on their academic, demographic, behavioral, and lifestyle attributes.

The project includes data preprocessing, categorical feature encoding, model training, and performance evaluation using multiple Machine Learning algorithms to identify the best-performing model.

---

## 📊 Dataset

- Dataset Name: 'hybrid_student_performance_1200.csv'
- Number of Records: 1200
- Target Variable: 'performance_risk_level'

The dataset contains **35 student-related features**, including demographic information, academic performance, study habits, behavioral factors, and career development attributes.

---

## ✨ Features

- Data preprocessing and cleaning
- Encoding categorical features into numerical values
- Training multiple Machine Learning models
- Comparing model performance
- Predicting student performance risk level
- Saving the trained model using Pickle

---

## 🤖 Machine Learning Algorithms Used

- Logistic Regression
- Linear Discriminant Analysis (LDA)
- SGD Classifier
- Decision Tree Classifier
- Random Forest Classifier
- Gaussian Naive Bayes


---

## 📈 Model Performance

| Algorithm | Accuracy (%) |
|-----------|-------------:|
| Logistic Regression | **91.61** |
| Linear Discriminant Analysis (LDA) | **91.61** |
| SGD Classifier | **90.96** |
| Decision Tree Classifier | **90.96** |
| Random Forest Classifier | **80.64** |
| Support Vector Machine (SVM) | **77.41** |
| Gaussian Naive Bayes | **65.16** |

### 🏆 Best Performing Models

- Logistic Regression achieved the highest accuracy of **91.61%**.
- Linear Discriminant Analysis (LDA) also achieved **91.61%** accuracy.

---

## 🛠 Technologies Used

- Python
- Pandas
- Scikit-learn
- Pickle
- Git
- GitHub

---

## 📂 Project Structure

student-performance-prediction-ml/

├── hybrid_student_performance_1200.csv          # Original dataset
├── cleaning.py                                  # Data preprocessing and encoding
├── Student_Performance_Preprocessed.csv         # Cleaned dataset
├── model.py                                     # Model training
├── stu_model.data                               # Saved trained model
├── prediction.py                                # Prediction using trained model
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore

---

## 🎯 Result

The project successfully predicts student performance risk levels using multiple Machine Learning classification algorithms.

Among all the evaluated models, **Logistic Regression** and **Linear Discriminant Analysis (LDA)** achieved the highest accuracy of **91.61%**, making them the best-performing models for this dataset.

---

## 👨‍💻 Author

**Yash Ravindra Khandait**

B.Tech in Electronics and Communication Engineering

**Skills:** Python • Machine Learning • Scikit-learn • Pandas
