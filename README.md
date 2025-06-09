
# Optiver: Realized Volatitliy Prediction

## Competition Description 
Kaggle competition link: https://www.kaggle.com/competitions/optiver-realized-volatility-prediction

### Competition objectives 

- Given a cross sectional time series of 112 stocks’ data derived from the order book and trade registers spanning from 1/1/2020 to 1/4/2020, the main objective is to build models that predict short-term volatility (future 10 minutes volatility) for hundreds of stocks across different sectors in the S&P500 universe. 

The final model will be evaluated against real market data collected in a three-month evaluation period after training using the Root Mean Squared Percentage Error (RMSPE) metric.


### Raw Datasets
The training datasets provided to build the models consist of 4 parquet files and 1 csv files.  They are book_[train/test].parquet, trade_[train/test].parquet, and train.csv.
All predictor variables and features can be found inside the 4 parquet files. The target can be found in the “train.csv” file
The labels/metadata in each of these files are described below:
##### book_[train/test].parquet A parquet file partitioned by stock_id. Provides order book data on the most competitive buy and sell orders entered into the market. The top two levels of the book are shared. The first level of the book will be more competitive in price terms, it will then receive execution priority over the second level.
- stock_id - ID code for the stock. Not all stock IDs exist in every time bucket. Parquet coerces this column to the categorical data type when loaded; you may wish to convert it to int8.
- time_id - ID code for the time bucket. Time IDs are not necessarily sequential but are consistent across all stocks.
- seconds_in_bucket - Number of seconds from the start of the bucket, always starting from 0.
- bid_price[1/2] - Normalized prices of the most/second most competitive buy level.
- ask_price[1/2] - Normalized prices of the most/second most competitive sell level.
- bid_size[1/2] - The number of shares on the most/second most competitive buy level.
- ask_size[1/2] - The number of shares on the most/second most competitive sell level.
##### trade_[train/test].parquet A parquet file partitioned by stock_id. Contains data on trades that actually executed. Usually, in the market, there are more passive buy/sell intention updates (book updates) than actual trades, therefore one may expect this file to be more sparse than the order book.
- stock_id - Same as above.
- time_id - Same as above.
- seconds_in_bucket - Same as above. Note that since trade and book data are taken from the same time window and trade data is more sparse in general, this field is not necessarily starting from 0.
- price - The average price of executed transactions happening in one second. Prices have been normalized and the average has been weighted by the number of shares traded in each transaction.
- size - The sum number of shares traded.
- order_count - The number of unique trade orders taking place. (size/order_count can be used as a single variable to measure average shares/trade)
##### train. csv The ground truth values for the training set.
- stock_id - Same as above, but since this is a csv the column will load as an integer instead of categorical.
- time_id - Same as above.
- target - The realized volatility computed over the 10 minute window following the feature data under the same stock/time_id. There is no overlap between feature and target data. You can find more info in our tutorial notebook.


![3](https://raw.githubusercontent.com/JosePeeterson/Optiver-Realized-Volatility-Prediction/main/docs/images/3.png)
![4](https://raw.githubusercontent.com/JosePeeterson/Optiver-Realized-Volatility-Prediction/main/docs/images/4.png)
![5](https://raw.githubusercontent.com/JosePeeterson/Optiver-Realized-Volatility-Prediction/main/docs/images/5.png)
![6](https://raw.githubusercontent.com/JosePeeterson/Optiver-Realized-Volatility-Prediction/main/docs/images/6.png)
![7](https://raw.githubusercontent.com/JosePeeterson/Optiver-Realized-Volatility-Prediction/main/docs/images/7.png)
![8](https://raw.githubusercontent.com/JosePeeterson/Optiver-Realized-Volatility-Prediction/main/docs/images/8.png)
![9](https://raw.githubusercontent.com/JosePeeterson/Optiver-Realized-Volatility-Prediction/main/docs/images/9.png)
![10](https://raw.githubusercontent.com/JosePeeterson/Optiver-Realized-Volatility-Prediction/main/docs/images/10.png)
![11](https://raw.githubusercontent.com/JosePeeterson/Optiver-Realized-Volatility-Prediction/main/docs/images/11.png)
![12](https://raw.githubusercontent.com/JosePeeterson/Optiver-Realized-Volatility-Prediction/main/docs/images/12.png)
![13](https://raw.githubusercontent.com/JosePeeterson/Optiver-Realized-Volatility-Prediction/main/docs/images/13.png)
![14](https://raw.githubusercontent.com/JosePeeterson/Optiver-Realized-Volatility-Prediction/main/docs/images/14.png)
![15](https://raw.githubusercontent.com/JosePeeterson/Optiver-Realized-Volatility-Prediction/main/docs/images/15.png)
![16](https://raw.githubusercontent.com/JosePeeterson/Optiver-Realized-Volatility-Prediction/main/docs/images/16.png)
![17](https://raw.githubusercontent.com/JosePeeterson/Optiver-Realized-Volatility-Prediction/main/docs/images/17.png)
![18](https://raw.githubusercontent.com/JosePeeterson/Optiver-Realized-Volatility-Prediction/main/docs/images/18.png)
![19](https://raw.githubusercontent.com/JosePeeterson/Optiver-Realized-Volatility-Prediction/main/docs/images/19.png)
![20](https://raw.githubusercontent.com/JosePeeterson/Optiver-Realized-Volatility-Prediction/main/docs/images/20.png)
![21](https://raw.githubusercontent.com/JosePeeterson/Optiver-Realized-Volatility-Prediction/main/docs/images/21.png)
![22](https://raw.githubusercontent.com/JosePeeterson/Optiver-Realized-Volatility-Prediction/main/docs/images/22.png)
![23](https://raw.githubusercontent.com/JosePeeterson/Optiver-Realized-Volatility-Prediction/main/docs/images/23.png)
![24](https://raw.githubusercontent.com/JosePeeterson/Optiver-Realized-Volatility-Prediction/main/docs/images/24.png)
![25](https://raw.githubusercontent.com/JosePeeterson/Optiver-Realized-Volatility-Prediction/main/docs/images/25.png)
![26](https://raw.githubusercontent.com/JosePeeterson/Optiver-Realized-Volatility-Prediction/main/docs/images/26.png)
![27](https://raw.githubusercontent.com/JosePeeterson/Optiver-Realized-Volatility-Prediction/main/docs/images/27.png)





### Additional Documentation

Final report location: https://github.com/JosePeeterson/Optiver-Realized-Volatility-Prediction/blob/main/docs/Final_summary_report.zip

Final presentation location: https://github.com/JosePeeterson/Optiver-Realized-Volatility-Prediction/blob/main/docs/final_presentation_n_submission.pptx
