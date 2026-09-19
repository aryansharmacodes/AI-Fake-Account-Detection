# 🤖 AI-Based Fake Account Detection

An end-to-end Machine Learning project that predicts whether an Instagram account is likely to be real or fake based on account-level features.

## 🚀 Project Overview

This project uses Machine Learning to analyze Instagram account characteristics such as followers, following, bio length, media count, profile picture, privacy status and username patterns.

The trained model is deployed through an interactive Streamlit web application.

## 🔄 Project Workflow

1. Dataset Collection
2. Data Preprocessing
3. Exploratory Data Analysis
4. Feature Engineering
5. Machine Learning Model Training
6. Model Evaluation
7. Error Analysis
8. Streamlit Deployment

## 🧠 Machine Learning Models

- Random Forest
- XGBoost

The final application uses the Random Forest model.

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Random Forest | 99.14% | 100% | 95% | 97.44% |
| XGBoost | 97.00% | 97.14% | 85% | 90.67% |

Random Forest ROC-AUC: **99.80%**

> Performance is measured on the held-out test dataset used during development.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Joblib
- Streamlit
- Jupyter Notebook

## 💻 Application Features

- Real/Fake account prediction
- Prediction probability
- Risk level
- Account warning indicators
- Feature importance visualization
- Model performance display
- Account summary

## ▶️ How to Run

Clone the repository and open the project folder.

Install dependencies:

```bash
pip install -r requirements.txt