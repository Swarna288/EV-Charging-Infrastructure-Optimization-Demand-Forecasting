# EV Charging Infrastructure Optimization & Demand Forecasting

## Project Overview

This project is an end-to-end Data Science and Machine Learning solution for analyzing and forecasting Electric Vehicle (EV) charging demand. It combines data preprocessing, exploratory data analysis, temporal feature engineering, machine learning, model evaluation, Streamlit deployment, and Power BI visualization.

The main objective is to predict `charging_demand` and use the resulting analysis to support charging-station capacity planning and infrastructure decision-making.

---

## Business Problem

EV charging demand changes according to time, vehicle characteristics, station conditions, electricity price, traffic, weather, and other operational factors. Poor demand estimation can contribute to overloaded stations, long queues, inefficient charger utilization, and inadequate infrastructure planning.

This project aims to:

- Forecast EV charging demand.
- Identify temporal and operational demand patterns.
- Compare regression models and select the best-performing model.
- Provide an interactive prediction application.
- Present analytical and decision-support insights in Power BI.

---

## Dataset

The raw dataset is stored at:

```text
data/raw/ev_charging_dataset.csv
```

The project also contains cleaned and feature-engineered versions under `data/processed/`.

### Target Variable

```text
charging_demand
```

This is the continuous target predicted by the regression models.

### Main Variables

The dataset contains information such as:

- Station information
- Vehicle type and battery capacity
- Initial state of charge
- Charging power
- Queue length
- Station load
- Electricity price
- Renewable-energy ratio
- Traffic density
- Weather condition
- Charging time information
- Charging priority
- Charging demand

A separate data dictionary is available at:

```text
data/output/data_dictionary.csv
```

---

## Project Workflow

```text
Raw EV Charging Data
        |
        v
Data Cleaning & Preprocessing
        |
        v
Exploratory Data Analysis
        |
        v
Temporal Feature Engineering
        |
        v
Lag / Rolling Features
        |
        v
Chronological Train-Test Split
        |
        v
ML Preprocessing Pipeline
        |
        v
Model Training & Comparison
        |
        v
Hyperparameter Tuning
        |
        v
Final Model Selection
        |
        v
Model Artifacts (.pkl)
        |
        +-------------------+
        |                   |
        v                   v
 Streamlit App        Power BI Analysis
```

---

## Data Cleaning and Preprocessing

The project performs data inspection and preprocessing before model training.

The machine-learning preprocessing pipeline includes:

### Numerical Features

- Median imputation for missing values
- Standardization using `StandardScaler`

### Categorical Features

- Most-frequent-value imputation
- One-hot encoding using `OneHotEncoder`
- Unknown categories handled during prediction

A Scikit-learn `ColumnTransformer` combines numerical and categorical preprocessing into a single pipeline.

---

## Exploratory Data Analysis

EDA is used to understand the behavior of EV charging demand and its relationship with other variables.

Generated EDA outputs include:

- `eda_correlation_matrix.png`
- `eda_demand_time_trend.png`
- `eda_hourly_demand.png`
- `eda_location_type_demand.png`
- `eda_target_boxplot.png`
- `eda_target_distribution.png`
- `eda_vehicle_type_demand.png`
- `eda_weekly_demand.png`

These analyses help examine demand distribution, temporal patterns, correlations, vehicle behavior, location behavior, and possible outliers.

---

## Feature Engineering

Time-based features are generated from the timestamp to capture charging-demand patterns.

Examples include:

```text
year
month
day
day_of_week
day_of_year
week_of_year
quarter
hour
minute
is_weekend
is_peak_hour
```

### Cyclical Features

Sine and cosine transformations are used for cyclical time variables, including features such as:

```text
hour_sin
hour_cos
day_of_week_sin
day_of_week_cos
month_sin
month_cos
day_of_year_sin
day_of_year_cos
```

### Historical Demand Features

Lag and rolling-demand features are used to represent recent charging behavior, including:

```text
demand_lag_1h
demand_lag_24h
demand_rolling_avg_6h
```

Processed datasets are stored in:

```text
data/processed/
```

Current processed files include:

```text
ev_charging_feature_engineered...
ev_charging_processed.csv
feature_engineered_data.csv
```

---

## Train-Test Strategy

Because the project contains temporal information, a chronological split is used rather than randomly shuffling observations.

```text
Earlier Data                       Later Data
|----------------------------------|----------|
            Training                  Testing
              80%                      20%
```

`TimeSeriesSplit` is also used for cross-validation so that model validation respects chronological order.

---

## Machine Learning Models

The project trains and compares regression models including:

- Linear Regression
- Ridge Regression
- XGBoost Regressor

XGBoost is additionally tuned using `GridSearchCV` with time-series cross-validation.

---

## Model Evaluation

Regression models are evaluated using:

- **MAE** — Mean Absolute Error
- **MSE** — Mean Squared Error
- **RMSE** — Root Mean Squared Error
- **R²** — Coefficient of Determination

Lower MAE, MSE, and RMSE values indicate smaller prediction errors, while a higher R² indicates stronger explanatory/predictive performance on the evaluated data.

### Recorded Final Comparison

| Model | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| Ridge Regression | 2.452 | 8.109 | 2.848 | 0.9895 |
| Tuned XGBoost | 2.472 | 8.275 | 2.877 | 0.9893 |

Based on the recorded evaluation results, **Ridge Regression was selected as the final model**.

---

## Model Evaluation Outputs

The `notebooks/outputs/` folder contains model-analysis files including:

```text
actual_vs_predicted.csv
actual_vs_predicted.png
feature_importance.png
final_model_comparison.csv
final_model_metrics.csv
model_comparison_rmse.png
model_comparison.csv
residual_plot.png
ridge_feature_importance.csv
ridge_feature_importance.png
time_series_cv_results.csv
```

These outputs provide evidence of model comparison, prediction quality, residual behavior, feature relevance, and cross-validation performance.

---

## Model Artifacts

Trained models and preprocessing objects are stored under:

```text
notebooks/artifacts/
```

The project currently contains artifacts for:

```text
final_ev_charging_linear_regression...
final_ev_charging_ridge_model.pkl
final_ev_charging_ridge_regression...
final_ev_charging_tuned_xgboost...
preprocessor.pkl
```

### Why `.pkl` Files Do Not Open as Text

Pickle files are binary serialized Python objects. They are not normal text files and therefore VS Code may display:

```text
The file is not displayed in the text editor because it is either binary
or uses an unsupported text encoding.
```

This is normal.

A saved model should be loaded using Python:

```python
import joblib

model = joblib.load(
    "artifacts/final_ev_charging_ridge_model.pkl"
)

print(model)
```

The final Ridge artifact contains the preprocessing components and Ridge model as a Scikit-learn pipeline.

---

## Streamlit Application

The project includes:

```text
notebooks/app.py
```

This application provides an interactive interface for using the trained model.

### Run Streamlit

From the project root:

```powershell
streamlit run notebooks/app.py
```

The model path used inside `app.py` must match the selected model artifact stored in `notebooks/artifacts/`.

---

## Power BI

The machine-learning/analysis output prepared for Power BI is:

```text
notebooks/outputs/ev_charging_powerbi_data.csv
```

The Power BI report is:

```text
EV_Charging_Optimization.pbix
```

Power BI is used to transform model and charging data into interactive business visualizations for demand, station, temporal, vehicle, and operational analysis.

---

## Notebooks

### `EV_Charging.ipynb`

Main Data Science and Machine Learning notebook containing the core workflow, including data processing, EDA, feature engineering, model development, tuning, evaluation, and artifact generation.

### `EV_Decision_Support.ipynb`

Contains decision-support analysis associated with the EV charging infrastructure problem and model results.

---

## Project Structure

```text
EV_Charging_Optimization/
|
+-- data/
|   +-- output/
|   |   +-- data_dictionary.csv
|   |
|   +-- processed/
|   |   +-- ev_charging_feature_engineered...
|   |   +-- ev_charging_processed.csv
|   |   +-- feature_engineered_data.csv
|   |
|   +-- raw/
|       +-- ev_charging_dataset.csv
|
+-- notebooks/
|   +-- artifacts/
|   |   +-- final_ev_charging_linear_regression...
|   |   +-- final_ev_charging_ridge_model.pkl
|   |   +-- final_ev_charging_ridge_regression...
|   |   +-- final_ev_charging_tuned_xgboost...
|   |   +-- preprocessor.pkl
|   |
|   +-- outputs/
|   |   +-- actual_vs_predicted.csv
|   |   +-- actual_vs_predicted.png
|   |   +-- eda_correlation_matrix.png
|   |   +-- eda_demand_time_trend.png
|   |   +-- eda_hourly_demand.png
|   |   +-- eda_location_type_demand.png
|   |   +-- eda_target_boxplot.png
|   |   +-- eda_target_distribution.png
|   |   +-- eda_vehicle_type_demand.png
|   |   +-- eda_weekly_demand.png
|   |   +-- ev_charging_powerbi_data.csv
|   |   +-- feature_importance.png
|   |   +-- final_model_comparison.csv
|   |   +-- final_model_metrics.csv
|   |   +-- model_comparison_rmse.png
|   |   +-- model_comparison.csv
|   |   +-- residual_plot.png
|   |   +-- ridge_feature_importance.csv
|   |   +-- ridge_feature_importance.png
|   |   +-- time_series_cv_results.csv
|   |
|   +-- app.py
|   +-- EV_Charging.ipynb
|   +-- EV_Decision_Support.ipynb
|
+-- src/
|
+-- EV_Charging_Optimization.pbix
+-- environment.yml
+-- README.md
```

---

## Technologies Used

| Area | Technology |
|---|---|
| Programming | Python |
| Data Manipulation | Pandas, NumPy |
| Visualization | Matplotlib |
| Machine Learning | Scikit-learn |
| Boosting Model | XGBoost |
| Validation | TimeSeriesSplit, GridSearchCV |
| Model Storage | Joblib / Pickle |
| Application | Streamlit |
| Dashboard | Microsoft Power BI |
| Development | VS Code, Jupyter Notebook |
| Environment | Conda |

---

## Installation

The project contains:

```text
environment.yml
```

To recreate the Conda environment, run from the project root:

```powershell
conda env create -f environment.yml
```

Then activate the environment name specified inside `environment.yml`.

---

## How to Run the Project

### 1. Clone or open the project

Open the `EV_Charging_Optimization` folder in VS Code.

### 2. Create/activate the environment

```powershell
conda env create -f environment.yml
```

Then activate the environment defined in that file.

### 3. Run the main notebook

Open:

```text
notebooks/EV_Charging.ipynb
```

and execute the required cells in order.

### 4. Verify the saved model

The selected Ridge model is stored in:

```text
notebooks/artifacts/final_ev_charging_ridge_model.pkl
```

### 5. Run the Streamlit application

From the project root:

```powershell
streamlit run notebooks/app.py
```

### 6. Open the Power BI report

Open:

```text
EV_Charging_Optimization.pbix
```

in Microsoft Power BI Desktop.

---

## Key Project Deliverables

The project produces:

- Cleaned EV charging data
- Feature-engineered datasets
- Data dictionary
- EDA visualizations
- Regression models
- Time-series cross-validation results
- Model-comparison results
- Actual-vs-predicted analysis
- Residual analysis
- Feature-importance analysis
- Serialized `.pkl` model artifacts
- Power BI-ready dataset
- Power BI report
- Streamlit prediction application

---

## Business Value

The project demonstrates how EV charging data can be converted into useful operational information.

Potential applications include:

- Forecasting charging demand
- Identifying high-demand periods
- Supporting charger-capacity planning
- Understanding station utilization
- Supporting infrastructure-expansion decisions
- Reducing operational inefficiencies
- Improving data-driven EV charging management

---

## Limitations

- Forecast quality depends on the quality and representativeness of the available dataset.
- Historical behavior may not perfectly represent future charging patterns.
- Model performance on new stations or different geographic regions may differ.
- Real infrastructure-expansion decisions require additional financial, geographic, grid-capacity, and operational information.
- External conditions such as EV adoption, tariffs, traffic, and charging technology may change over time.

---

## Future Enhancements

Future development could include:

- Real-time charging-data integration
- Weather and traffic forecast integration
- Station-level forecasting
- Geospatial analysis for station expansion
- Clustering of high-demand locations
- Additional forecasting algorithms
- Automated model retraining
- Online Streamlit deployment
- Live database/API integration
- Automated Power BI refresh

---

## Conclusion

This project demonstrates an end-to-end Data Science workflow for **EV Charging Infrastructure Optimization and Demand Forecasting**.

It covers data preprocessing, EDA, temporal feature engineering, machine learning, time-aware validation, model comparison, model persistence, interactive prediction, and business visualization.

Based on the recorded model evaluation, **Ridge Regression** was selected as the final model with approximately:

```text
MAE  : 2.452
MSE  : 8.109
RMSE : 2.848
R²   : 0.9895
```

The project therefore demonstrates both technical machine-learning implementation and practical decision-support reporting for EV charging infrastructure.
