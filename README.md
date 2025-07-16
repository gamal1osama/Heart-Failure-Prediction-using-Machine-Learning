# 🫀 Heart Failure Prediction using Machine Learning

Predicting the likelihood of death events in heart failure patients using clinical features and various classification algorithms.  
This project applies data cleaning, preprocessing, model training, evaluation, hyperparameter tuning (Optuna), and deployment using `Streamlit` and `Gradio`.

---

## 📂 Dataset

- **Source**: [Heart Failure Clinical Records Dataset – Kaggle](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data/data)
- **Samples**: 299 patients  
- **Features**: 13 clinical features + 1 target (`DEATH_EVENT`)

---

## 📊 Exploratory Data Analysis (EDA)

Key issues discovered during analysis:
- **Outliers** in numerical features
- **Skewed distributions** (e.g., creatinine, platelets)
- **Imbalanced target class** (more survivors than deaths)

### ✅ Solutions:
- Removed outliers using IQR method  
- Normalized skewed data using `Box-Cox Transformation`  
- Applied `SMOTE` to balance target class

---

## 🧹 Preprocessing

- **Train-Test Split** (after SMOTE)
- **Standardization** using `StandardScaler`
- **Feature Scaling** only applied to models that require it
- No feature engineering added – all original features used

---

## 🤖 Models Trained

| Type           | Model                   |
|----------------|--------------------------|
| Linear         | Logistic Regression      |
| Probabilistic  | Naive Bayes              |
| Distance-Based | K-Nearest Neighbors (KNN)|
| Margin-Based   | Support Vector Machine   |
| Tree-Based     | Decision Tree            |
| Ensemble       | Random Forest            |
| Boosting       | XGBoost, LightGBM        |

---

## 🏆 Model Comparison

All models were evaluated using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **ROC-AUC**

### 📌 Final Choice:
- ✅ **Model**: `LightGBM`
- ✅ **Why?**: Achieved **highest performance** across most metrics (especially F1-Score)

---

## 🔍 Hyperparameter Tuning

- Implemented using [`Optuna`](https://optuna.org/)  
- Tuned parameters like `num_leaves`, `learning_rate`, `max_depth`, etc.  
- Cross-validated with `StratifiedKFold`  
- Trained final model with `early_stopping` and best parameters

---

## 🧪 Final Model Pipeline

- Built using `Pipeline` from `scikit-learn`
- Trained on best `LightGBM` model
- Supports future integration into web apps / APIs
- Saved using `joblib`

---

## 🚀 Deployment

The model was deployed using both:

- **Streamlit App** → [🔗 Visit on Hugging Face](https://huggingface.co/spaces/MarwanAmin/Streamlit-Heart-Failure-Prediction-using-ML)
- **Gradio App** → [🔗 Visit on Hugging Face](https://huggingface.co/spaces/MarwanAmin/Heart-Failure-Prediction-using-ML)

---

## 🔮 Future Improvements

- Add Explainability using **SHAP** or **LIME**
- Improve UI/UX of Streamlit app
- Integrate as a full medical assistant tool
- Deploy to cloud (e.g., AWS / Render)
- Add patient history form for user input

---

## 📁 Project Structure

