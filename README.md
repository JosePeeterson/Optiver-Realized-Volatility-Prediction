
# Optiver: Realized Volatility Prediction

Kaggle competition link: https://www.kaggle.com/competitions/optiver-realized-volatility-prediction

## Competition objectives 

- Given a cross sectional time series of 112 stocks’ data derived from the order book and trade registers spanning from an unknown period, the main objective is to build models that predict short-term volatility (future 10 minutes volatility) for hundreds of stocks across different sectors in the S&P500 universe. 

<img src="https://raw.githubusercontent.com/JosePeeterson/Optiver-Realized-Volatility-Prediction/main/docs/images/28.png" alt="Alt Text" width="600" height="500">

The final model will be evaluated against real market data collected in a three-month evaluation period after training using the Root Mean Squared Percentage Error (RMSPE) metric.


### Raw Datasets
The training datasets provided to build the models consist of 4 parquet files and 1 csv files.  They are book_[train/test].parquet, trade_[train/test].parquet, and train.csv.

All predictor variables and features can be found inside the 4 parquet files. The target can be found in the “train.csv” file.

The labels/metadata in each of these files are described below:
##### book_[train/test].parquet: A parquet file partitioned by stock_id. Provides order book data on the most competitive buy and sell orders entered into the market. The top two levels of the book are shared. 
- stock_id - ID code for the stock. Not all stock IDs exist in every time bucket. 
- time_id - ID code for the time bucket. Time IDs are not necessarily sequential but are consistent across all stocks.
- seconds_in_bucket - Number of seconds from the start of the bucket, always starting from 0.
- bid_price[1/2] - Normalized prices of the most/second most competitive buy level.
- ask_price[1/2] - Normalized prices of the most/second most competitive sell level.
- bid_size[1/2] - The number of shares on the most/second most competitive buy level.
- ask_size[1/2] - The number of shares on the most/second most competitive sell level.
##### trade_[train/test].parquet: A parquet file partitioned by stock_id. Contains data on trades that actually executed. Usually, in the market, there are more passive buy/sell intention updates (book updates) than actual trades, therefore one may expect this file to be more sparse than the order book.
- stock_id - Same as above.
- time_id - Same as above.
- seconds_in_bucket - Same as above. Note that since trade and book data are taken from the same time window and trade data is more sparse in general, this field is not necessarily starting from 0.
- price - The average price of executed transactions happening in one second. Prices have been normalized and the average has been weighted by the number of shares traded in each transaction.
- size - The sum number of shares traded.
- order_count - The number of unique trade orders taking place. (size per order_count can be used as a single variable to measure average shares per trade)
##### train.csv: The ground truth values for the training set.
- stock_id - Same as above, 
- time_id - Same as above.
- target - The realized volatility computed over the 10 minute window following the feature data's 10 minute window under the same stock_id and time_id. There is no overlap between feature and target data.

### Quirks of the competition
The host of the competition obfuscated and anonymized the ticker symbols of the stocks. To prevent inferring the actual stock ticker; Temporal ordering of the time ids for All stocks are deliberately jumbled and the prices of stocks are normalized by price at seconds_in_bucket = 0 within each time id. 

### Reverse engineering the denormalized prices and correct time id order

We try to roughly recreate the correct time_id order so that time series techniques such as walk-forward cross validation can be performed. However, we cannot create time series features because we cannot infer the time_id order in the test set. 

* using information about minimum tick level of $0.01 in denormalized prices and minimum difference in normalized prices, we can infer the price at seconds_in_bucket = 0 within a time_id as $0.01 / minimum_normalized_price_difference. We can multiply this with normalized price difference of all remaining seconds_in_bucket to get the actual denormalized price time series in each time_id
* Given 112 stock_ids with 3830 (jumbled) time_ids where each time id has a price time series spanning 600 seconds, we want to reorder the time ids so that the nearest and most continuous prices are aligned together. The time series in each time_id can be aggreated into a single number like the average price so that a matrix of 3830 time_ids x 112 stock_ids can be created. In the high dimensional feature space samples that will be distributed and localized based on their closeness in terms of distances like Euclidean distance to their true neighbouring samples. The non-linear dimensionality reduction technique, T-SNE allows to faithfully visualize this distribution and localization as a 2d manifold. 

### Measuring and Quantifying Volatility

<img src="https://raw.githubusercontent.com/JosePeeterson/Optiver-Realized-Volatility-Prediction/main/docs/images/3.png" alt="Alt Text" width="400" height="100">

## Feature Engineering

### 1.	Liquidity Features:
* Bid-ask spread: Widening spreads may indicate increased uncertainty or market stress, leading to higher volatility.
* Bid-ask size: Larger bid-ask sizes may indicate deeper liquidity and lower volatility, while smaller sizes may suggest thinner markets and higher volatility.
* Order book depth: Deeper order books with more substantial quantities at various price levels may provide stability and dampen volatility.
### 2.	Price and Volume Features:
* Trade price: Extreme price movements may signal increased volatility, while stable prices may indicate lower volatility.
* Trade size: Larger trade sizes may lead to more significant price changes and higher volatility, especially in illiquid markets.
* Trade volume: Higher trading volumes may coincide with increased volatility, as heightened market activity reflects changing investor sentiment and price dynamics.
### 3.	Volatility Indicators:
* High-low range: Wider price ranges between highs and lows suggest greater price fluctuations and higher volatility.
* Volatility clustering: Periods of high volatility tend to cluster together, leading to persistence in volatility levels over time.
* Implied volatility: Changes in implied volatility may precede shifts in realized volatility, as options traders adjust their expectations of future price movements.
### 4.	Temporal Features:
* Lagged variables: Past volatility levels and price movements can influence current volatility, as volatility tends to exhibit serial correlation over time.
* Moving averages: Moving averages smooth out short-term fluctuations and can help identify trends or changes in volatility regimes.
* Autocorrelation: Positive autocorrelation in volatility suggests persistence in volatility levels, while negative autocorrelation may indicate mean-reverting behaviour.
### 5.	Market Microstructure Features:
* Order flow imbalance: Imbalances between buy and sell orders may lead to price pressure and subsequent volatility spikes.
* Price impact: Large trades can have a significant impact on prices, leading to short-term volatility.
* Spread dynamics: Changes in bid-ask spreads may reflect shifts in market sentiment and liquidity conditions, influencing volatility.
### 6.	Correlation and Co-movement:
* Cross-correlations: Correlations between stock returns can affect portfolio volatility, especially during periods of market-wide shocks or systemic risk.
* Co-integration: Long-term relationships between stock prices may influence volatility dynamics, especially in pairs trading or arbitrage strategies.


## Exploratory Data Analysis

### PCA
Clustering on the input feature (wap1_log_price_ret_buks). Perform PCA on the feature matrix (time_id x stock_id) to reduce noise in the returns data.

<img src="https://raw.githubusercontent.com/JosePeeterson/Optiver-Realized-Volatility-Prediction/main/docs/images/4.png" alt="Alt Text" width="800" height="300">

### Clustering on Temporal target realized volatility correlation

#### Correlation matrix before clustering
<img src="https://raw.githubusercontent.com/JosePeeterson/Optiver-Realized-Volatility-Prediction/main/docs/images/5.png" alt="Alt Text" width="600" height="500">

#### Hierarchical Agglomerative Clustering
<img src="https://raw.githubusercontent.com/JosePeeterson/Optiver-Realized-Volatility-Prediction/main/docs/images/7.png" alt="Alt Text" width="800" height="250">

#### Correlation matrix after clustering
<img src="https://raw.githubusercontent.com/JosePeeterson/Optiver-Realized-Volatility-Prediction/main/docs/images/6.png" alt="Alt Text" width="600" height="500">

### Target Exploration
#### Target realized volatility clustering analysis

Clustering Conditional correlation between stocks from precision matrix using affinity propagation. split time frame into three regimes with 2 quarters (half year) for each regime.

63*9*2 = 1134 time ids each regime. [1134, 2268, 3831]

##### Time_id_index : 0 - 1134 

<img src="https://raw.githubusercontent.com/JosePeeterson/Optiver-Realized-Volatility-Prediction/main/docs/images/8.png" alt="Alt Text" width="400" height="400">

##### Time_id_index : 1134 - 2268
<img src="https://raw.githubusercontent.com/JosePeeterson/Optiver-Realized-Volatility-Prediction/main/docs/images/9.png" alt="Alt Text" width="400" height="400">

##### Time_id_index : 2268 - 3831 
<img src="https://raw.githubusercontent.com/JosePeeterson/Optiver-Realized-Volatility-Prediction/main/docs/images/10.png" alt="Alt Text" width="400" height="400">

#### Target distribution of individual stock ids in the form of box plots

<img src="https://raw.githubusercontent.com/JosePeeterson/Optiver-Realized-Volatility-Prediction/main/docs/images/11.png" alt="Alt Text" width="1000" height="300">

#### Skew and Kurtosis transformation of Target

<img src="https://raw.githubusercontent.com/JosePeeterson/Optiver-Realized-Volatility-Prediction/main/docs/images/12.png" alt="Alt Text" width="1000" height="300">

### Feature Exploration and Selection
<img src="https://raw.githubusercontent.com/JosePeeterson/Optiver-Realized-Volatility-Prediction/main/docs/images/13.png" alt="Alt Text" width="600" height="500">
<img src="https://raw.githubusercontent.com/JosePeeterson/Optiver-Realized-Volatility-Prediction/main/docs/images/14.png" alt="Alt Text" width="600" height="500">
<img src="https://raw.githubusercontent.com/JosePeeterson/Optiver-Realized-Volatility-Prediction/main/docs/images/15.png" alt="Alt Text" width="800" height="400">

## Visualize Code control flow (code2flow package)
<img src="https://raw.githubusercontent.com/JosePeeterson/Optiver-Realized-Volatility-Prediction/main/docs/images/16.png" alt="Alt Text" width="800" height="1600">

## Model Training and Automatic Hyperparameter Optimization

### Search space:
### Hyperparameter Configuration Dictionary

```python
params = {
    'disable_default_eval_metric': 1,
    "max_depth": trial.suggest_int('max_depth', 3, 10),
    "eta": trial.suggest_float(name='eta', low=0.0001, high=0.1, log=True),
    "subsample": round(trial.suggest_float(name='subsample', low=0.1, high=1.0, step=0.1), 1),
    "colsample_bytree": round(trial.suggest_float(name='colsample_bytree', low=0.1, high=1.0, step=0.1), 1),
    'gamma': trial.suggest_int('gamma', 0, 5),
    'reg_alpha': trial.suggest_int('reg_alpha', 0, 5),
    'reg_lambda': trial.suggest_int('reg_lambda', 0, 5),
    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
    "tree_method": 'hist',
    "device": "cuda",
    "seed": seed1,
    # 'missing': missing_value
}
```

### Learning Curves
<img src="https://raw.githubusercontent.com/JosePeeterson/Optiver-Realized-Volatility-Prediction/main/docs/images/18.png" alt="Alt Text" width="800" height="500">



### Optuna hyperparameter optimization
<img src="https://raw.githubusercontent.com/JosePeeterson/Optiver-Realized-Volatility-Prediction/main/docs/images/17.png" alt="Alt Text" width="1000" height="300">

``` python
Best trial:
Best number of iteration/boosting rounds:  1135
Trial no.:  32
  Value:  0.22699900086611163
  Params: 
    max_depth: 17
    eta: 0.012302593098587278
    subsample: 0.75
    colsample_bytree: 0.8
    gamma: 0
    reg_alpha: 11
    reg_lambda: 8
    min_child_weight: 6

####################################### PREDICTION #################################################
corr(y_pred/v1ts, y_true/v1ts) 0.4592863365106767
log(corr( )) 0.5351106470319219
corr(y_pred, y_true) 0.9057492349057186
log(corr( )) 0.9185397908698034
RMSPE train score:  0.19065680871350316
RMSPE test score:  0.23036724443746043

Competition score: 0.22021 (Version 19)
```



## Train set Error Analysis
<img src="https://raw.githubusercontent.com/JosePeeterson/Optiver-Realized-Volatility-Prediction/main/docs/images/19.png" alt="Alt Text" width="1000" height="300">
##### Smallest RMSPE stock ids: [56,10,14,84,126,94,51,20,50,123]
##### Largest RMSPE stock ids: [31,37,18,80,33,81,60,88,110,27]

### scatter plot of Prediction Vs. Groundtruth   |   Histogram of Prediction Vs. Groundtruth
<img src="https://raw.githubusercontent.com/JosePeeterson/Optiver-Realized-Volatility-Prediction/main/docs/images/20.png" alt="Alt Text" width="1200" height="250">

### stock ids with with largest gaps (discontinous support) in histograms
<img src="https://raw.githubusercontent.com/JosePeeterson/Optiver-Realized-Volatility-Prediction/main/docs/images/21.png" alt="Alt Text" width="1200" height="300">

### Residual Analysis
<img src="https://raw.githubusercontent.com/JosePeeterson/Optiver-Realized-Volatility-Prediction/main/docs/images/22.png" alt="Alt Text" width="800" height="650">

### Feature Interactions analysis

<img src="https://raw.githubusercontent.com/JosePeeterson/Optiver-Realized-Volatility-Prediction/main/docs/images/24.png" alt="Alt Text" width="600" height="300">

## Test set Error Analysis

<img src="https://raw.githubusercontent.com/JosePeeterson/Optiver-Realized-Volatility-Prediction/main/docs/images/23.png" alt="Alt Text" width="1000" height="300">

##### Smallest RMSPE stock ids: [77,56,35,14,61,126,84,44,96,76]
##### Largest RMSPE stock ids: [31,37,88,18,110,112,27,33,58,60]

## Improvement

### A separate ‘Medium-Bias-Low-variance’ RF model for Stock_id = 31 

``` python 
Study statistics: 
  Number of finished trials:  75
  Number of pruned trials:  0
  Number of complete trials:  75
Best trial:
Trial no.:  70
  Value:  0.8187703265947197
  Params: 
    n_estimators: 1200
    max_depth: 53
    max_leaves: 195
    max_features: auto
    min_samples_leaf: 1
    min_samples_split: 9
Final model
0.310187551436994
0.37951193883504114

Train RMSPE for st 31 only: 0.310
Test RMSPE  for st 31 only: 0.379
```

<img src="https://raw.githubusercontent.com/JosePeeterson/Optiver-Realized-Volatility-Prediction/main/docs/images/25.png" alt="Alt Text" width="800" height="300">


# Final Competition Score (After deadline)

### Our score
<img src="https://raw.githubusercontent.com/JosePeeterson/Optiver-Realized-Volatility-Prediction/main/docs/images/26.png" alt="Alt Text" width="600" height="300">

### Our standing amongst top 100 rank
<img src="https://raw.githubusercontent.com/JosePeeterson/Optiver-Realized-Volatility-Prediction/main/docs/images/27.png" alt="Alt Text" width="600" height="300">





### Additional Documentation

Final report location: https://github.com/JosePeeterson/Optiver-Realized-Volatility-Prediction/blob/main/docs/Final_summary_report.zip

Final presentation location: https://github.com/JosePeeterson/Optiver-Realized-Volatility-Prediction/blob/main/docs/final_presentation_n_submission.pptx
