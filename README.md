# fraud-detection-app
Credit Card Fraud Detection using Machine Learning
# Real-Time Fraud Detection System 💳🚀

## 🎯 Project Overview
This project implements a machine learning pipeline to detect fraudulent credit card transactions. Using a highly imbalanced dataset (0.17% fraud), I developed a system that prioritizes **Precision** to minimize false alarms while maintaining high **Recall**.

## 🛠️ Tech Stack
- **Languages:** Python
- **Libraries:** Scikit-Learn, XGBoost, Imbalanced-Learn (SMOTE), Pandas, Matplotlib
- **Deployment:** Streamlit (Coming Soon)

## 📊 Key Results (XGBoost)
- **Recall:** 85.7% (Caught 84/98 frauds in the test set)
- **Precision:** High efficiency with only 33 false positives out of 56,000+ transactions.
- **Data Balancing:** Applied **SMOTE** to address the class imbalance, moving from 394 fraud cases to 227,451 synthetic samples for training.

## 📁 Project Structure
- `data/`: Raw dataset (gitignored)
- `notebooks/`: EDA and Model Training
- `models/`: Saved `.pkl` files (XGBoost, Random Forest, Scaler)
- `images/`: Performance visualizations (PR Curves, Confusion Matrix)