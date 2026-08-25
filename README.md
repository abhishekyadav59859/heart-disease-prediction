# Heart Disease Prediction

A machine learning project that predicts whether a patient is at risk of heart disease, based on clinical attributes like age, cholesterol, blood pressure, and chest pain type. Built as an end-to-end pipeline — from raw data to a deployed, interactive web app.

## Live Demo
🔗 [Try the app](#) 
## Dataset
UCI Heart Disease dataset (combined from 4 sources: Cleveland, Hungary, Switzerland, VA Long Beach) — 920 patient records, 15 clinical attributes.

## Project Structure



## Approach
1. **EDA** — explored missingness patterns across the 4 source hospitals, found that `ca`, `slope`, and `thal` were missing in ~99% of non-Cleveland records (a collection artifact, not random) and dropped them accordingly.
2. **Preprocessing** — imputed remaining missing values (median for numeric, mode for categorical), label-encoded categorical features, binarized the target (disease present / not present).
3. **Modeling** — trained and compared Logistic Regression, Decision Tree, KNN, and Random Forest. Selected the best model based on F1-score.
4. **Explainability** — analyzed feature coefficients to understand which clinical factors drive predictions most.
5. **Deployment** — built a Streamlit app so anyone can input patient data and get an instant risk prediction.

## Best Model
Logistic Regression (selected based on F1-score comparison — see `notebooks/03_modeling.ipynb` for full results).

## Run Locally
```bash
git clone <your-repo-url>
cd heart-disease-prediction
pip install -r requirements.txt
streamlit run app/app.py
```

## Team

-  Team Lead, Modeling
- [Name] — Data Cleaning & EDA
- [Name] — App Development
- [Name] — Documentation & Testing

## Tech Stack
Python, pandas, scikit-learn, Streamlit, matplotlib