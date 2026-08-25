# 🫀 Heart Disease Prediction

An end-to-end machine learning project that predicts a patient's risk of heart disease based on clinical attributes such as age, cholesterol, blood pressure, and chest pain type. Built from raw data all the way to a live, interactive web app.

🔗 **Live App:** [heart-disease-prediction-abhi.streamlit.app](https://heart-disease-prediction-abhi.streamlit.app/)

---

## 📌 Overview

Heart disease is one of the leading causes of death worldwide, and early risk assessment can help guide timely medical attention. This project uses patient clinical data to classify whether a person is likely at risk of heart disease, using a full ML pipeline — data cleaning, exploratory analysis, model comparison, explainability, and deployment.

## 📊 Dataset

UCI Heart Disease dataset — combined from **4 hospital sources**: Cleveland, Hungary, Switzerland, and VA Long Beach (920 patient records, 15 clinical attributes).

A key finding during EDA: three features (`ca`, `slope`, `thal`) were missing in ~99% of records outside the Cleveland source — a **data collection artifact**, not random missingness. These were dropped to avoid biasing the model toward Cleveland-only patterns.

## 🧠 Approach

1. **Exploratory Data Analysis** — examined missingness patterns across sources, target distribution, and feature relationships with disease presence.
2. **Preprocessing** — imputed remaining missing values (median for numeric, mode for categorical), label-encoded categorical features, and binarized the target (disease present / not present).
3. **Modeling** — trained and compared four models: Logistic Regression, Decision Tree, KNN, and Random Forest — evaluated on accuracy, precision, recall, and F1-score.
4. **Explainability** — analyzed feature coefficients to understand which clinical factors most influence predictions.
5. **Deployment** — built and deployed an interactive Streamlit app so anyone can input patient data and get an instant prediction.

## 🏆 Best Model

**Logistic Regression** — selected based on F1-score comparison across all four models (see `notebooks/03_modeling.ipynb` for full results and comparison table).

## 📁 Project Structure

heart-disease-prediction/
├── data/
│ ├── raw/ → original UCI dataset
│ └── processed/ → cleaned, encoded dataset
├── notebooks/
│ ├── 01_eda.ipynb → exploratory data analysis
│ ├── 02_cleaning.ipynb → preprocessing and encoding
│ └── 03_modeling.ipynb → model training, comparison, explainability
├── models/
│ ├── heart_model.pkl → trained model
│ ├── feature_columns.pkl → expected input column order
│ ├── encoders.pkl → label encoders for categorical fields
│ └── feature_importance.png → explainability chart
├── app/
│ └── app.py → Streamlit web app
└── requirements.txt


## ⚙️ Run Locally

```bash
git clone https://github.com/abhishekyadav59859/heart-disease-prediction.git
cd heart-disease-prediction
pip install -r requirements.txt
streamlit run app/app.py
```

## 🛠️ Tech Stack

Python · pandas · scikit-learn · Streamlit · matplotlib

## 👤 Author

**Abhishek Yadav**
B.Tech, Mathematics and Computing, NIT Mizoram

## 📄 License

This project is for educational purposes.