# Heart Disease Prediction

This is a machine learning project to predict whether a patient is at risk of heart disease based on clinical data like age, cholesterol, blood pressure, and chest pain type. It covers the full process from raw data to a deployed web app.

Live app: [Heart Disease Prediction](https://heart-disease-prediction-abhi.streamlit.app/)

## About the project

Heart disease is one of the most common causes of death, and predicting risk early can help with timely treatment. This project uses patient data from the UCI Heart Disease dataset to build a model that classifies whether someone is likely to have heart disease.

## Dataset

The UCI Heart Disease dataset, combined from four sources: Cleveland, Hungary, Switzerland, and VA Long Beach. In total it has 920 patient records and 15 clinical attributes.

While exploring the data, I found that three columns (ca, slope, thal) had almost no data outside the Cleveland source (around 99% missing). This wasn't random - those hospitals just didn't record those particular tests. Because of this, I dropped those three columns instead of imputing them, since filling in values the other hospitals never measured would have biased the model toward Cleveland patterns.

## Steps followed

1. Exploratory data analysis - looked at missing values, how they differed by source, and how features relate to the target.
2. Data cleaning - handled missing values (median for numeric columns, mode for categorical ones), encoded categorical features, and converted the target into a binary disease/no-disease label.
3. Model training - trained and compared four models: Logistic Regression, Decision Tree, KNN, and Random Forest. Compared them using accuracy, precision, recall, and F1-score.
4. Explainability - looked at feature coefficients to see which factors mattered most for predictions.
5. Deployment - built a Streamlit app so anyone can enter patient details and get a prediction.

## Best model

Logistic Regression performed best based on F1-score. Full comparison is in notebooks/03_modeling.ipynb.

## Project structure

## Running it locally

```bash
git clone https://github.com/abhishekyadav59859/heart-disease-prediction.git
cd heart-disease-prediction
pip install -r requirements.txt
streamlit run app/app.py

## Tech used

Python, pandas, scikit-learn, Streamlit, matplotlib

## Author

Abhishek Yadav
B.Tech, Mathematics and Computing, NIT Mizoram

## Note

This project is for learning purposes and is not meant for actual medical diagnosis.