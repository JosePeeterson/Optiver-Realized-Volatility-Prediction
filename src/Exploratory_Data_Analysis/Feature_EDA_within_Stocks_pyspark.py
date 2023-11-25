from pyspark import SparkContext
from pyspark.sql import SparkSession

# Create a SparkSession
spark = SparkSession.builder.appName('app_name') \
.master('local[*]') \
.config('spark.sql.execution.arrow.pyspark.enabled', True) \
.config('spark.sql.session.timeZone', 'UTC') \
.config('spark.driver.memory','200G') \
.config('spark.ui.showConsoleProgress', True) \
.config('spark.sql.repl.eagerEval.enabled', True).getOrCreate()

#sc = SparkContext.getOrCreate()

# Read the DataFrame from Parquet file
book_wap_log_returns_df = spark.read.parquet("/home/optimusprime/Desktop/peeterson/optiver/Optiver-Realized-Volatility-Prediction/data/wap_log_returns.parquet/book_first_10_min_wap_log_returns-001.parquet")

# Filter the DataFrame based on the row_id condition
filtered_df = book_wap_log_returns_df.filter(book_wap_log_returns_df['row_id'] == '0-5')

# Show the filtered DataFrame
filtered_df.show()
