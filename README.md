# Heart Failure Prediction Project

## Overview
This project focuses on developing machine learning models to predict heart failure events (DEATH_EVENT) based on clinical records. The implementation follows a structured machine learning pipeline from data exploration to model evaluation and feature importance analysis.

## Project Structure
The project is organized as a Jupyter notebook (`heart_failure.ipynb`) containing:
1. **Data Loading and Exploration**
2. **Data Cleaning and Preprocessing**
3. **Exploratory Data Analysis (EDA)**
4. **Feature Engineering**
5. **Model Training and Evaluation**
6. **Feature Importance Analysis**
7. **Model Saving and Deployment**

## Dataset
The dataset contains 299 patient records with 13 clinical features:
* **Demographics**: age, sex
* **Medical Conditions**: anaemia, diabetes, high_blood_pressure, smoking
* **Clinical Measurements**: creatinine_phosphokinase, ejection_fraction, platelets, serum_creatinine, serum_sodium
* **Follow-up**: time (days)
* **Target Variable**: DEATH_EVENT (1 = death, 0 = alive)

## Key Features
* Comprehensive EDA with visualizations
* Data preprocessing including scaling and handling class imbalance
* Evaluation of multiple machine learning models:
   * Logistic Regression
   * Decision Tree
   * Random Forest
   * KNN
   * XGBoost
   * Naive Bayes
   * SVM
   * LightGBM/CatBoost
* Feature importance analysis using SHAP values
* Model saving using pickle/joblib

## Requirements
* Python 3.7+
* Libraries:
   * pandas, numpy
   * matplotlib, seaborn
   * scikit-learn
   * XGBoost, LightGBM
   * SHAP
   * imbalanced-learn

## Usage
1. Clone the repository
2. Install required packages: `pip install -r requirements.txt`
3. Run the Jupyter notebook: `jupyter notebook heart_failure.ipynb`

## Results
The project evaluates models using multiple metrics:
* Accuracy
* Precision
* Recall
* F1-score
* ROC AUC

Feature importance analysis reveals the most significant clinical factors contributing to heart failure predictions.

## Future Work
* Hyperparameter tuning with Optuna
* Deployment as a web application
* Integration with electronic health record systems
* Additional feature engineering

## License
This project is open-source and available under the MIT License.

## Acknowledgments
* Dataset source: UCI Machine Learning Repository
* Special thanks to the open-source community for the libraries used in this project
