# SOL-igence - COM724_AE2

Crypto insights and forecaster

## Installation

1. install the necessary requirements from requirements.txt
   1. For example, create a new conda environment with the requirements (which can be found at the root of the repo):
      1. ``` conda create --name my_env --file  requirements.txt ```
      2. ``` conda activate my_env ```

## Running The Project
1. run the streamlit app. From the root of the project directory:
   1. ``` streamlit run soligence-streamlit/streamlit-crypto-dashboard.py ```

## Data Preprocessing And Analysis

[see data quality notebook](notebooks/data_quality.ipynb)

[see data cleansing notebook](notebooks/data_cleansing.ipynb)

### Exploratory Data Analysis

[see EDA notebook](notebooks/eda.ipynb)

[Feature importance](notebooks/feature_importance.ipynb)

### Time Series Decomposition

[see TS notebook](notebooks/time-series.ipynb)

## Modelling

### Clustering

[see EDA notebook](notebooks/eda.ipynb)

### Time series evaluation

[ARIMA](notebooks/arima-prototyping.ipynb)

[prophet](notebooks/prophet-prototyping.ipynb)

[Exponential smoothing](notebooks/single-exponential-smoothing.ipynb)

[Random forest](notebooks/single-exponential-smoothing.ipynb)

### Forecasting

[XGBoost](notebooks/xgboost-time-series.ipynb)

## Pre-trained models

- Sci-kit learn MultiOutputRegressor objects built on top of XGBoostRegressor objects.
- Sci-kit learn RobustScaler objects
- Trained-feature column lists.

## oustanding work
- real confidence intervals for forecasting.
- reactJS app.

## Bugs

- The confidence interval may need to be deselected to view the value on the interactive line plot, in forecast mode.
