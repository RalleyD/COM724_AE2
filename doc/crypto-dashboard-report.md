# Cryptocurrency Forecasting Dashboard: Development and Evaluation Report

## Introduction

The cryptocurrency market has experienced significant growth and volatility since its inception, creating both opportunities and challenges for investors, traders, and researchers (Corbet et al., 2019). With over 10,000 cryptocurrencies in existence and a market capitalization exceeding $1 trillion, making informed investment decisions requires sophisticated analytical tools capable of processing vast amounts of data and identifying meaningful patterns (Levi and Lipton, 2022). Traditional financial analysis methods often fall short when applied to cryptocurrencies due to their unique characteristics, including extreme volatility, 24/7 trading, and sensitivity to technological, regulatory, and market sentiment factors (Fang et al., 2022).

This report documents the development of a cryptocurrency forecasting dashboard designed to address these challenges. The dashboard leverages machine learning techniques, real-time data integration, and interactive visualizations to provide actionable insights for cryptocurrency market participants. Expanding on conventional financial dashboards, this solution incorporates advanced time-series forecasting capabilities specifically optimised for cryptocurrency price prediction, correlation analysis, and investment decision support.

## Objectives and Problem Definition

### Problem Definition

The cryptocurrency market presents several unique challenges that this project aims to address:

1. The cryptocurrency ecosystem generates vast amounts of data across multiple platforms, making it difficult for individuals to process and interpret market dynamics effectively.

2. Both novice and experienced market participants require comprehensive tools that combine historical analysis, real-time monitoring, and predictive capabilities in an accessible interface.

3. Cryptocurrency price movements exhibit high volatility and non-linear patterns that are difficult to interpret.

4. Researchers and cryptocurrency newcomers require different levels of analytical depth, visualization options, and interpretability.

### Project Objectives

To address these challenges, the project established the following key objectives:

1. Connectivity with cryptocurrency exchange APIs to access current market data, while maintaining the ability to fall back on static datasets for testing and development purposes.

2. Develop interactive visualisations that adapt to user preferences, including time interval selection, profit targets, and specific cryptocurrencies of interest.

3. Create predictive models capable of generating price forecasts across useful timeframes with quantifiable confidence levels.

4. Provide recommendations on optimal entry points and target prices based on user-defined profit goals.

5. Identify and visualise relationships between different cryptocurrencies to support portfolio diversification strategies.

6. Utilise Python for both backend processing and machine learning components, with Streamlit for frontend development to ensure rapid iteration and deployment capabilities.

## Analysis, Evaluation and Results

### Data Collection and Preprocessing

The data pipeline begins with collection from the Binance API, chosen for its reliability, comprehensive coverage of major cryptocurrencies, and granular historical data access (Katsiampa et al., 2023). To ensure system robustness, a fallback mechanism using cached CSV data was implemented for situations where API connectivity is compromised.

The preprocessing workflow consists of several key stages:

1. **Data Cleaning**: Handling missing values, removing outliers, reducing multi-level indexes and ensuring consistent datetime formatting across all datasets.

2. **Feature Engineering**: Creating technical indicators and derived features that capture market dynamics:
   - Lagged features (1, 3, 5-day) to capture short-term momentum
   - Moving averages (7 and 30-day) to identify trends
   - Volatility measures using rolling standard deviations
   - Relative Strength Index (RSI) to identify overbought/oversold conditions
   - Percentage price changes over various timeframes

3. **Sequence Creation**: Generating input-output pairs for model training, with 60-day input windows mapped to 30-day forecast horizons, creating a sliding window approach for multi-step forecasting.

### Clustering and Correlation

K-means clustering was used to determine relationships between cryptocurrencies. K-means clustering was chosen for its efficiency in identifying logical groupings of data (TODO CITE).

Using a dataset containing last year's close prices of the top 30 market cap coins, the dataset was transformed from long to wide format i.e each day's close price as columns and a single row for each coin. Providing each coin as an observation for clustering with a rich set of features. To aid clustering performance, the dimensionality of the data was reduced to its optimal principal components (2).

![](images/pca-explained-variance.png)  
*Figure - Explained variance*

![](images/silhouette.png)  
*Figure - Silhouette scores - K-means clustering*

Silhouette scoring was used to determine the optimal number of clusters. While fewer clusters achieved a higher score, higher-granularity could be achieved with four clusters while maintaining a strong separation performance (> 0.9).

![](images/pca_clusters.png)  
*Figure - Scatter Plot - Four clusters Plotted Against Principal Components*

The clusters provided a starting point for capturing different market states. Bitcoin variants were clustered together; this asset has demonstrated a significant increase in value, particularly over the past year. Etherium variants are clustered, representing a coin with significant price fluctuations, generally decreasing over the past year. BNB and Bitcoin Cash are clustered, both have been holding a stable value over the past year. The final cluster represent coins of lower value which haven't significantly increased in value.

From these four clusters, a representative coin was selected from each cluster for model evaluation.

Correlation analysis identifies coins that follow or oppose the market behaviour for the representative coins. This shall be leveraged to provide portfolio diversification insights.

|         |   BTC-USD |   WBTC-USD |   DOGE-USD |   HBAR-USD |    PI-USD |   TON-USD |   USDT-USD |    USDC-USD |
|:--------|----------:|-----------:|-----------:|-----------:|----------:|----------:|-----------:|------------:|
| BNB-USD |   0.82948 |   0.829018 |   0.806936 |   0.786174 | -0.193981 | -0.159604 | -0.0985299 | -0.00642017 |

|         |   WBTC-USD |   LEO-USD |   DOGE-USD |   XLM-USD |   TON-USD |    PI-USD |   USDC-USD |   USDT-USD |
|:--------|-----------:|----------:|-----------:|----------:|----------:|----------:|-----------:|-----------:|
| BTC-USD |   0.999971 |  0.954715 |   0.931514 |  0.930809 | -0.297925 | -0.201206 | -0.0540657 |  0.0495566 |

|         |   STETH-USD |   WETH-USD |   WSTETH-USD |   DOT-USD |    OM-USD |   TRX-USD |   USDC-USD |   USDT-USD |
|:--------|------------:|-----------:|-------------:|----------:|----------:|----------:|-----------:|-----------:|
| ETH-USD |    0.999967 |   0.999826 |     0.997976 |  0.848954 | -0.120443 | 0.0450684 |  0.0738517 |  0.0772101 |

|         |   HBAR-USD |   ADA-USD |   XRP-USD |   WBTC-USD |   TON-USD |   USDC-USD |   USDT-USD |   PI-USD |
|:--------|-----------:|----------:|----------:|-----------:|----------:|-----------:|-----------:|---------:|
| LTC-USD |   0.894071 |  0.884937 |    0.8645 |    0.86442 | -0.106227 | 0.00487731 |  0.0608996 | 0.132695 |

*Table - Top Positively and Negatively Correlated Coins - Pearson Correlation*

### Exploratory Data Analysis and Time-Series Decomposition

1. Stationarity: Applying time series decomposition and Augmented Dickey-Fuller tests to assess stationarity properties of different cryptocurrencies. For example, Bitcoin (BTC) demonstrated non-stationarity with a high p-value, while Binance Coin (BNB) showed more stationary characteristics.

|         |      ADF |    P_Value |       1% |       5% |      10% | Stationary   |
|:--------|---------:|-----------:|---------:|---------:|---------:|:-------------|
| BNB-USD | -3.67848 | 0.00442625 | -3.44844 | -2.86951 | -2.57102 | True         |
| BTC-USD | -1.1333  | 0.701611   | -3.44844 | -2.86951 | -2.57102 | False        |
| ETH-USD | -2.05408 | 0.263377   | -3.44844 | -2.86951 | -2.57102 | False        |
| LTC-USD | -1.96639 | 0.301503   | -3.44844 | -2.86951 | -2.57102 | False        |
  *Table - Statoinarity Analysis - ADF Test*

2. Distribution: revealed asymmetric patterns with heavy left or right tails for most cryptocurrencies, with Bitcoin showing particularly high kurtosis.

![](images/coin_box.png)  
*Figure - Box Plots*

![](images/freq-dist.png)  
*Figure - Frequency Distribution Plots*

3. Distrubtion over annual intervals: a general shift from postiviely skewed data to negatviely skewed from 2021 to 2025 with an increase in kurtosis. Indicating, increasing close price values and with a higher liklihood of extreme values. Litecoin being the exception which maintained a consistent concentration of close prices in a lower range.

![](images/bnb_dist.png)  
*Figure - Histograms over annual intervals - BNB*

![](images/btc_dist.png)  
*Figure - Histograms over annual intervals - BTC*

These non-normal distributions and lack of consistent patterns between coins that could be derived from decomposition (i.e trend and seasonality), informed the selection of modeling approaches that could handle such data characteristics.

4. Autocorrelation: short-term partial autocorrelation indicated a useful indicator for feature engineering.

![](images/btc-pacf.png)  
*Figure - Partial Autocorrelation - BTC*

I.e. two to three time lags of statistical significance.

### Feature Importance

Utilising random forest regression to determine the relative importance of engineered features derived from the close prices.

![](images/feature-importance-btc.png)

Accross the representative cryptocurrencies, the following derived features returned the highest relative importance:

- lags 1, 3 and 5.
- 7-day and 30-day moving averages.

### Model Selection and Optimization

The model selection process employed a structured evaluation of multiple forecasting approaches:

1. **Statistical Models**:
   - ARIMA: Traditional time series forecasting with autoregressive and moving average components.
   - Exponential Smoothing: Assumes no trend or seasonality and the ability to bias short-term dependencies through exponential weighting.

2. **Machine Learning Models**:
   - Random Forest.
   - XGBoost with optimised hyperparameters.
   - Prophet (Facebook's forecasting tool).

The evaluation metrics included Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE) and R-squared score; to assess model performance.

#### Model Assessment

#### Arima

ARIMA models have been widely used in financial forecasting (Box et al., 2015) to capture linear temporal relationships. This presents some limitations:

1. Most of the representative coins exhibit no stationarity, despite attempts to transform or difference the data.
2. Time series decomposition revealed non-linear patters over lagged periods also noted by Bariviera et al. (2017).
3. Capturing high volatility i.e high variance resulted in low R-squared scores.

| Model   |   MASE |   RMSSE |     MAE |    RMSE |   MAPE |   SMAPE |      R2 |
|:--------|-------:|--------:|--------:|--------:|-------:|--------:|--------:|
| ARIMA   | 1.2824 |  1.0809 | 3192.87 | 3989.93 | 0.0391 |  0.0379 | -1.4911 |  
*Table - Arima Performance Evaluation - BTC Past 5 Year Close Prices*

![](images/arima-eval.png)  
*Figure - Arima Prediction Line Plot*

Due to its popularity, it was selected for evaluation as a means to compare against other methods.

#### Exponential Smoothing

Single exponentia smoothing was selected for it's ability to perform regression on univariate data with decreasing weights for older observations (TODO cite). Despite this model's ability to adapt to changing trends, it exhibited similar limitations:

1. The long term autocorrelation in the close prices may have caused poor performance (Kristjanpoller and Minutolo, 2018).
2. Lacking complexity required to model short term volatility.
3. Acheiving slightly bettere results that ARIMA but underperforming.

| Model   |   MASE |   RMSSE |     MAE |    RMSE |   MAPE |   SMAPE |      R2 |
|:--------|-------:|--------:|--------:|--------:|-------:|--------:|--------:|
| Exponential Smoothing | 0.7698 |0.6517	| 5246.4672 | 6196.5309	| 0.0582	| 0.0556 |-1.3908 |  
*Table - Exponential Smoothing Evaluation*

4. Differencing the data showed noticable improvement.

| Model                  | MASE   | RMSSE  | MAE       | RMSE      | MAPE   | SMAPE  | R²    |
|------------------------|--------|--------|-----------|-----------|--------|--------|-------|
| Exponential Smoothing | 1.2419 | 1.1864 | 1635.2009 | 2288.3155	 | 1.1406 | 1.4772 | -0.0723 |  
*Table - Exponential Smoothing Evaluation - Differenced Data*

![](images/exp-smooth-differenced-tuned.png)  
*Figure - Tuned Exponential Smoothing Plot - Trained on Differenced Data*

#### Random Forest

Leverages ensebmle decision trees to model complex non-linear relationships. It is robust to outliers which handles extreme price movements (Breiman, 2001). However, Random Forest doesn't consider sequential dependencies, leading to a low R-squared score.

| Model                  | MASE   | RMSSE  | MAE       | RMSE      | MAPE   | SMAPE  | R²    |
|------------------------|--------|--------|-----------|-----------|--------|--------|-------|
| RandomForestRegressor | 1.1709 | 1.1426 | 1541.7756 | 2203.8583 | 1.3086 | 1.4074 | 0.0054 |  
*Table - Random Forest Performance Evaluation*

#### Prophet By Meta

Incorporates decomposable time-series models with trend and seasonality components. Similarly to ARIMA, this presented some limitations:

1. Lack of trend and seasonality.

![](images/prophet-eval.png)  
*Figure - Prophet Prediction Line Plot.*

2. Due to the inherently noisy data with spurious historical peaks, Prophet appears too sensitive with changepoint detection.

| Model   |    MASE |   RMSSE |     MAE |    RMSE |   MAPE |   SMAPE |       R2 |
|:--------|--------:|--------:|--------:|--------:|-------:|--------:|---------:|
| Prophet | 7.1681 | 5.3037 | 17327.8076 | 19026.5950 | 0.1915 | 0.1742 | -25.0629 |  
*Table - Prophet Evaluation*

Ultimately, leading to poor performance.

#### Extreme Gradient Boosting

Implements gradient boosting with regularisation and tree-pruning (TOOD cite). This demonstrated the following advantages:

1. Iteratively building trees from the residuals of the prior, helps to capture complex temporal patterns (Chen and Guestrin, 2016).
2. Prevents overfitting to historical patterns.
3. Designed to model non-linear relationships, it can capture non-linear patterns observed in the cryptocurrency markets.

PyCaret was utilised for efficient model optimisation. The optimisation process, employed cross-validation with a time-series split to prevent data leakage and ensure realistic performance estimation.

```Python
from pycaret.regression import *

# PyCaret Regression Setup
xgb_exp = RegressionExperiment().setup(
    data=btc_train, 
    target="close",
    session_id=123, 
    fold=3,  # K-fold cross-validation
    data_split_shuffle=False,  # **Important: Keeps time-series order**
    fold_strategy="timeseries",  # Ensures time-series split
)

# Train XGBoost Model
xgb_model = xgb_exp.create_model('xgboost')
```
*Figure 1 - XGBoost Experiment Code*

| Model                     | MAE       | MSE          | RMSE      | R²     | RMSLE  | MAPE   |
|---------------------------|-----------|--------------|-----------|--------|--------|--------|
| Extreme Gradient Boosting | 9053.9287 | 243341152.00 | 15599.3955 | 0.4389 | 0.1972 | 0.1044 |
*Table 1 XGBoost Results - Initial Experiment with 5 year BTC Close Price Data*

Utilising derived features for training the candidate model yielded 

XGBoost emerged as the best-performing model due to its ability to capture non-linear relationships and handle the high volatility characteristic of cryptocurrency data. The model was configured with the following hyperparmeters:

- Subsample: 0.8 (to reduce overfitting)
- Max depth: 4 (balancing model complexity with generalization)
- Learning rate: 0.05 (enabling regularisation between training iterations)
- Number of estimators: 200 (providing sufficient model complexity)

Lagged features were extracted from the close prices which also yielded strong performance scores. This introduced a challenge that necessitated these exogenous variables in each single forecast over the forecast window. With respect the objective of achieving the highest accuracy, modelling the exogenous variables on highly voltatile data was impractical and introduced additional computational overhead.

To facilitate and simplify multi-step forecasting, the optimized XGBoost model was wrapped in a MultiOutputRegressor, enabling simultaneous prediction of multiple future time points. This method was validated through extensive backtesting and yielded poorer performance scores than single-step forecasting:

| Cryptocurrency | MAE        | MSE            | RMSE    | R²    |
|----------------|------------|----------------|-------|-----|
| Bitcoin (BTC)  | 15885.20 | 453152984.84 | 21287.39 | -0.71 |  
*Table - Multi-output XGBoost Regressor Scores*

These scores are based on the inclusion of the recommended derived featues (7-day moving average, 30-day moving average). This is mainly due to the degradation in prediction accuracy over wider forecast windows (TODO cite). Increasing model complexity did not yield significant improvements. Removal of highly correlated features (lagged data) and including lower ranked features improved performance (see Table 2)

### Model Evaluation Results

The model evaluation revealed significant performance variations across different cryptocurrencies:

| Cryptocurrency | MAE        | MSE            | RMSE    | R²    |
|----------------|------------|----------------|---------|-------|   
| Bitcoin (BTC)  | 14622.54 | 421942655.86 | 20541.24 | -0.59 |  
| Ethereum (ETH) | 108.63     | 19445.99    | 139.45 | 0.93 |
| Litecoin (LTC) | 3.72      | 32.88        | 5.73 | 0.93  |
| Binance Coin (BNB) | 63.59 | 6536.66    | 80.85 | -0.84 |
*Table 2 - XGBoost Evaluation Across Multiple Cyptocurrencies*

These results highlight several important observations:

1. The absolute error metrics (MAE, MSE) vary significantly based on the price scale of each cryptocurrency. 

2. Litecoin showed the best predictive performance with an R² of 0.93, indicating that the model captured approximately 93% of the price variance.

3. Bitcoin and Binance Coin proved particularly difficult to forecast, with negative R² values suggesting that the model performed worse than a simple mean predictor for these assets.

The short-term partial autocorrelation and performance scores highlight the importance of training the model on as much recent data as possible. Table 2, presents scores determined over a holdout period (20 %) equivalent to one year of data.

Further experiments with feature combinations and model configurations revealed that:

1. Using moving average, price change and RSI features consistently produced better results than short term lags.

2. The performance discrepancies across currencies correlate with their non-stationary properties, supporting the hypothesis that stationary assets are inherently more predictable.

The final model configuration adopted a best-fit approach, using the best-performing feature set and hyperparameters accross coins selected from each cluster. This decision prioritised forecast accuracy, aligning with the project's objective of maximising predictive performance.

## Interactive Dashboard Development

The dashboard implementation utilsed Streamlit, a Python library designed for creating data applications with minimal frontend development overhead. This choice enabled rapid iteration and deployment while maintaining full access to Python's data science ecosystem.

### Design Overview

The dashboard architecture implements distinct functional components:

1. **Data management**:
   - API connection via the `get_crypto_data()` function.
   - Fallback mechanism through `generate_mock_data()`.
   - News retrieval via `get_crypto_news()`.
   - Caching with `@st.cache_data` for performance optimisation.

2. **Feature processing**:
   - Technical indicator generation in `add_features()`.
   - Sequence creation for model input in `create_sequences()`.
   - Market state assessment in `get_market_state()`.

3. **Forecasting components**:
   - Model training pipeline in `train_forecast_model()`.
   - Investment recommendation in `calculate_buy_recommendation()`.
   - What-if scenario analysis in `calculate_profit_scenarios()`.

4. **Visualisation modules**:
   - Price history charts
   - Moving average visualizations
   - Forecast projection with confidence intervals
   - Correlation heatmaps
   - Performance metrics displays

The user interface employs tab-based navigation to organise related visualisations and controls, maintaining consistent visual style. This structure preserves shared context (selected cryptocurrency, time range) while organising specific analytical functions.

### User Interface Features

The dashboard provides several key user interface elements designed for intuitive interaction:

1. **Dashboard controls**:
   - Cryptocurrency selector with support for major coins (BTC, ETH, LTC, BNB).
   - Time interval slider for adjusting analysis timeframes (7, 14, 30, 60 and 90 days).
   - Target profit parameter for customising investment recommendations.
   - Data refresh button for real-time updates.

2. **Market overview**:
   - Current price display with change indicators.
   - Volatility metrics calculated over a 30-day window.
   - Market state assessment (bullish/bearish) with confidence rating.
   - 24-hour trading volume metrics.

3. **Analysis tabs**:
   - Price History: Line charts showing historical price movements.
   - Moving Averages: Technical analysis view with 7-day and 30-day MAs.
   - Price Forecast: Projection charts with confidence intervals.

4. **Correlation analysis**:
   - Positive correlation panel showing most closely aligned assets.
   - Negative correlation panel highlighting diversification opportunities.
   - Pearson correlation strength indicators.

5. **Investment calculator**:
   - Buy recommendation card with recommended buy-date and price.
   - Sell target calculation based on user-defined profit goals.
   - Confidence metric for recommendation reliability.
   - What-if analysis showing potential returns for different scenarios.

6. **Supplementary information**:
   - Recent news panel with cryptocurrency-specific headlines.
   - Key performance indicators section with market capitalisation and volume.
   - Major coins trend prediction with probability indicators.

The visual design employs a consistent color scheme with semantic meaning (green for positive changes, red for negative) and responsive layouts that adapt to different screen sizes. Card-based components with clear visual hierarchy enhance readability, while interactive plots provide fine-grained information.

## Limitations, Challenges and Future Enhancements

### Current Limitations

The dashboard implementation faces several limitations that affect its utility and performance:

1. The forecasting process requires approximately one minute to complete when a coin is selected or data is refreshed, impacting the real-time analysis experience.
   - This latency stems from the and the computational demands of the XGBoost model training, compounded by the multi-step configuration.

2. The forecasting performance varies significantly across different cryptocurrencies, with Bitcoin showing particularly poor results despite being the most commonly analyzed asset.
   - This inconsistency complicates the user experience, as reliability expectations must be managed differently for each cryptocurrency.

3. The dashboard should implement more sophisticated cache mechanisms to balance freshness with availability. For example, saving live data and using this during periods of API unavailability, refreshing the static dataset when it becomes stale.

4. The evaluation metrics might not fully capture the practical utility of forecasts for investment decision-making (TODO cite).

### Technical Challenges

Several technical challenges emerged during development:

1. **Data stationarity**:
   - Most cryptocurrencies exhibit non-stationary behavior that traditional time series models struggle to handle.
   - Transformation techniques (differencing, log transformation) provided limited improvement in making the data more amenable to modeling.

2. **Multi-Step forecasting**:
   - The direct multi-step approach using MultiOutputRegressor introduced model complexity and training requirements.
   - Error propagation in long-horizon forecasts remains problematic, with accuracy degrading significantly beyond 14 days.

3. **Feature engineering at scale**:
   - Balancing feature richness with performance requirements necessitated compromises in the final implementation. 

### Future Enhancements

Based on the identified limitations and challenges, several future enhancements are proposed:

1. **Performance Optimization**:
   - Create smaller, faster models that approximate the performance of the full XGBoost ensemble.
   - Iterative model training on short-term data, trained initially on long-term data and deployed to the application.

2. **Application stability**:
   - Implement a queueing mechanism to mitigate API limit restrictions.
   - Cache live data as a temporary fallback to maintain a consistent level of service.

3. **Model Improvements**:
   - Develop a cryptocurrency-specific model architecture which captures the inter-cluster differences between representative coins.
   - Investigate deep learning approaches i.e LSTM, that may better capture long-range dependencies in price data.
   - Implement ensemble methods that combine multiple model types or configurations to increase robustness.

4. **Enhanced features**:
   - Incorporate sentiment analysis from news and social media to capture market mood indicators.
   - Add on-chain metrics (transaction volume, active addresses) as additional features for improved forecasting.
   - Develop discrete risk profiles that adjust recommendations based on user risk preference.

5. **Expanded capabilities**:
   - Implement portfolio optimisation tools that leverage the correlation analysis.
   - Add scenario modeling for major market events (regulatory changes, technological developments).

## Conclusion

The cryptocurrency forecasting dashboard addresses the initial objectives by providing an integrated platform for market analysis, visualisation, and prediction. The implementation demonstrates key achievements:

1. Combining real-time data access, modeling techniques, and interactive visualisations in a coherent user experience.

2. Complementary perspectives on cryptocurrency markets, from historical patterns to forecasted trends and asset correlation.

3. Investment recommendation and what-if analysis components transform predictions into actionable insights tailored to user preferences.

4. Multi-step forecasting approach using optimised XGBoost models represents a practical application of machine learning techniques in a challenging domain.

However, the varying model performance across different cryptocurrencies highlights the inherent challenges in this domain. The significant differences in predictability between Bitcoin, Ethereum, and Litecoin suggest that cryptocurrency-specific modeling approaches are necessary for better results.

Despite these challenges, the dashboard provides value as both an analytical tool and a research platform. For cryptocurrency newcomers, it offers accessible visualisations and simplified decision support. For researchers, it provides a foundation for exploring alternative modeling approaches and feature engineering techniques.

Future work should focus on addressing the identified limitations, particularly in terms of performance optimisation and model accuracy for challenging cryptocurrencies like Bitcoin. The incorporation of additional data sources, especially sentiment and on-chain metrics, presents avenues for enhancing the predictive capabilities of the system.

In conclusion, the cryptocurrency forecasting dashboard represents a step toward more informed, data-driven decision-making in the volatile cryptocurrency markets, while also highlighting the continuing challenges in financial time series forecasting.

## References

Corbet, S., Lucey, B., Urquhart, A. and Yarovaya, L. (2019) 'Cryptocurrencies as a financial asset: A systematic analysis', International Review of Financial Analysis, 62, pp. 182-199.

Fang, F., Ventre, C., Basios, M., Kanthan, L., Martinez-Rego, D., Wu, F. and Li, L. (2022) 'Cryptocurrency trading: a comprehensive survey', Financial Innovation, 8(1), pp. 1-59.

Katsiampa, P., Corbet, S. and Lucey, B. (2023) 'High-frequency volatility co-movements in cryptocurrency markets', Journal of International Financial Markets, Institutions and Money, 74, article 101659.

Levi, S. and Lipton, A. (2022) 'Bitcoin: Basics, Dominance, and Investment Aspects', in Financial Cryptography and Data Security, Springer, pp. 101-122.

Kumar, D. and Rath, S.K. (2020) 'Predicting the Trend of Stock Market using Ensemble based Machine Learning Techniques', International Conference on Computational Intelligence in Data Science, pp. 1-6.

Chen, S., Härdle, W.K., Hou, A.J. and Wang, W. (2022) 'Network quantile autoregression', Journal of Econometrics, 232(2), pp. 1356-1384.

Demirer, R., Gupta, R., Lv, Z. and Wong, W.K. (2019) 'Equity return dispersion and stock market volatility: Evidence from multivariate linear and nonlinear causality tests', Sustainability, 11(2), article 351.

Sezer, O.B., Gudelek, M.U. and Ozbayoglu, A.M. (2020) 'Financial time series forecasting with deep learning: A systematic literature review: 2005–2019', Applied Soft Computing, 90, article 106181.

Chen, T. and Guestrin, C. (2016) 'XGBoost: A Scalable Tree Boosting System', Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 785-794.

Sirignano, J. and Cont, R. (2019) 'Universal features of price formation in financial markets: perspectives from deep learning', Quantitative Finance, 19(9), pp. 1449-1459.

----

## Appendix

### Project repository

The readme contains an overview of the analysis notebooks and their location. In addition, installation and running instructions:

https://github.com/RalleyD/COM724_AE2

### Model Performance Metrics

| Model | MASE | RMSSE | MAE | RMSE | MAPE | SMAPE | R² | TT (Sec) |
|:---------|--------:|--------:|----------:|----------:|-------:|--------:|---------:|----------:|
| Naive Forecaster | 2.5327 | 2.1194 | 6056.3326 | 7501.5005 | 0.0669 | 0.0678 | -1.6501 | 0.2160 |
| AdaBoost w/ CD | 2.6866 | 2.1923 | 6451.2172 | 7790.0061 | 0.0722 | 0.0710 | -1.9247 | 0.3600 |
| Auto ARIMA | 2.6645 | 2.2160 | 6389.6435 | 7865.9940 | 0.0708 | 0.0704 | -2.0378 | 1.6940 |
| Exponential Smoothing | 2.6740 | 2.2224 | 6412.8554 | 7889.1502 | 0.0710 | 0.0706 | -2.0772 | 0.2860 |
| Random Forest w/ CD | 2.9986 | 2.4268 | 7183.2460 | 8600.3892 | 0.0800 | 0.0802 | -2.4607 | 0.8320 |
| ARIMA | 2.8555 | 2.3083 | 6860.5286 | 8222.2546 | 0.0760 | 0.0741 | -3.1024 | 0.0380 |
| Prophet | 6.7363 | 4.9985 | 16263.5200 | 17906.3268 | 0.1788 | 0.1674 | -22.3559 | 0.2120 |

### Key Feature Importance

The table below shows the relative importance of features for the XGBoost model optimized for Bitcoin prediction:

| Feature | Importance |
|:--------|----------:|
| close_lag_1 | 0.89 |
| close_lag_3 | 0.027 |
| close_lag_5 | 0.012 |
| ma7 | 0.05 |
| ma30 | 0.03 |
| rsi | 0.0002 |
| std7 | 8.48e-05 |
| price_change_7d | 0.0002 |
| price_change_1d | 0.0023 |

### PyCaret Optimization Process

The PyCaret optimization workflow involves:

1. Setup phase with time series cross-validation
2. Model comparison across multiple algorithms
3. Hyperparameter tuning of the best-performing model (XGBoost)
4. Final model training with optimized parameters
5. Integration with MultiOutputRegressor for sequence prediction
