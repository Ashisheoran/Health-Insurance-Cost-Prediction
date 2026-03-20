# Health Insurance Cost Prediction

A machine learning project that estimates medical insurance charges from user demographics and lifestyle factors.

This project combines exploratory analysis, model training, and a Streamlit app for real-time insurance cost prediction.

## Problem Statement

Insurance charges are influenced by multiple attributes such as age, BMI, smoking status, and region. The goal is to build a regression model that predicts expected insurance cost from these inputs.

## Features

- Interactive Streamlit UI for real-time predictions
- End-to-end inference pipeline saved as a single artifact (`insurance_pipeline.pkl`)
- Consistent feature schema between training and serving
- Categorical handling for sex, smoker status, and region

## Tech Stack

- Python
- Pandas, NumPy
- scikit-learn
- Matplotlib, Seaborn
- Streamlit

## Project Structure

- `app.py` - Streamlit application for user input and predictions
- `cost_prediction.ipynb` - EDA, preprocessing, training, evaluation, and export workflow
- `insurance.csv` - Source dataset
- `insurance_pipeline.pkl` - Production inference artifact (preprocessing + model)
- `insurance_model.pkl` - Legacy model-only artifact (kept for reference)

## Model Overview

The current setup uses a scikit-learn Pipeline that combines:
1. `StandardScaler` for numerical scaling
2. `RandomForestRegressor` for charge prediction

Saving preprocessing and model together prevents training/serving mismatch issues and ensures feature alignment during inference.

## Requirements

Recommended Python version: 3.10+

## Quick Start

1. Clone or download the project.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

The app will open in your browser. Enter values for:
- Age
- Gender
- BMI
- Number of children
- Smoking status
- Region

Then click **Predict My Insurance Cost**.

## Retrain and Export Pipeline

1. Open `cost_prediction.ipynb`
2. Run all cells in order
3. Confirm `insurance_pipeline.pkl` is generated/updated in the project root

## Output

The app returns an estimated insurance charge based on user inputs:
- Age
- Sex
- BMI
- Number of children
- Smoker status
- Region

## Notes on Inference Consistency

`app.py` reads feature names from the saved pipeline and reindexes incoming input columns to the exact training order before prediction. This avoids common `ValueError` issues caused by feature-order mismatches.

## Limitations

- Predictions are estimates, not official insurer quotes
- Model quality depends on data representativeness and size
- External factors (plan type, provider network, comorbidities) are not included

## Suggested Next Improvements

- Hyperparameter tuning with cross-validation
- Confidence intervals or uncertainty estimates
- Model comparison (XGBoost, LightGBM, ElasticNet)
- Better experiment tracking (MLflow/W&B)
- Add unit tests for preprocessing and inference schema

## Author

Update this section with your name and portfolio links.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
