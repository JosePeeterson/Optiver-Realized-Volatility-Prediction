# %%
import os
import glob
import pandas as pd

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
import plotly.subplots as sub_plots
import plotly.graph_objects as go
import statsmodels.api as sm
import scipy.stats as stats


from sklearn.cluster import KMeans
import re

import warnings
#warnings.filterwarnings("ignore")
from sklearn.metrics import confusion_matrix
#from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


from sklearn.utils import class_weight
import optuna
from optuna.trial import TrialState

from xgboost import XGBRegressor
from mlxtend.evaluate import bias_variance_decomp

import glob
import pandas as pd
import numpy as np
import glob
import os
from numba import jit, njit
import numba as nb
import plotly_express as px
from itertools import combinations, permutations, product, combinations_with_replacement
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.signal import find_peaks
import pickle
from joblib import Parallel, delayed
import seaborn as sns
from sklearn import model_selection
from sklearn.metrics import r2_score
import gc
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import scipy as sp
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import SpectralClustering, MiniBatchKMeans, MeanShift, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import skew, kurtosis
import shap
from datetime import datetime
import ipywidgets as widgets
from matplotlib.patches import Rectangle
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from xgboost import plot_tree, plot_importance
from sklearn.model_selection import RepeatedKFold, cross_val_score, TimeSeriesSplit
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.genmod.generalized_linear_model import GLM
import warnings
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import class_weight
import optuna
from optuna.trial import TrialState
from xgboost import XGBRegressor
from mlxtend.evaluate import bias_variance_decomp
import re

from matplotlib.pyplot import cm


from sklearn.manifold import TSNE
from sklearn.preprocessing import minmax_scale

from scipy.stats import spearmanr

# %%
os.chdir('/home/optimusprime/Desktop/peeterson/optiver/Optiver-Realized-Volatility-Prediction/data/Final_submission_data/partial_train_n_full_test')
#os.chdir('/content/drive/MyDrive/optiver_real_vol/Final Submission/kaggle_submission_versions')
#os.chdir('/kaggle/input/optiver-submission-training-data')

with open('train_feat_df_reordered.pkl','rb') as f:
  train_feat_df_reordered = pickle.load(f)

with open('test_feat_df.pkl','rb') as f:
  test_feat_df = pickle.load(f)

#os.chdir('/content/drive/MyDrive/optiver_real_vol/kaggle/input/optiver-realized-volatility-prediction')

##### remove test from training data #####

df_train_reordered = train_feat_df_reordered.copy()
del train_feat_df_reordered

df_test = test_feat_df.copy()
del test_feat_df


# %%

####################### Improvement 1 #######################
## # drop the bottom 25 xgb features
bottom_25_xgb_feat = ['as_ap_corr1',
 'as_ap_corr2',
 'ask_lvl2_min_lvl1_size_feat',
 'bp_as_corr1',
 'bp_as_corr2',
 'bs_ap_corr1',
 'bs_ap_corr2',
 'bs_as_corr2',
 'bs_bp_corr1',
 'bs_bp_corr2',
 'liqt20rf29',
 'liqvol3',
 'max_ask_price1',
 'max_ask_price2',
 'max_bid_price1',
 'max_bid_price2',
 'robust_sum_stats_2_clusters_labels',
 'tlog_1p_range_ask_size1',
 'tlog_1p_range_bid_size1',
 'tlog_max_ask_size1',
 'tlog_max_ask_size2',
 'tlog_max_bid_size1',
 'tlog_max_bid_size2',
 'tlog_range_ask_size2',
 'tlog_tlinear_sad_size1',
 'tlog_tlog1p_target_vol_robust_sum_stats_20_clusters']

# ['tlog_tlog1p_eps1e4_range_bid_price1', 'bp_as_corr2', 'tlog_eps1e4_trade_price_std', 'v1liq2projt20', 'max_price1',
#                      'max_bid_price2', 'max_ask_price2', 'bs_ap_corr2', 'min_price1', 'ask_lvl2_min_lvl1_size_feat', 'tlog_1p_trade_order_count_std',
#                      'min_ask_price1', 'tlog_1p_trade_order_count_mean', 'min_bid_price2', 'lvl2_minus_lvl1_bid_n_ask_size_feat', 'bs_ap_corr1',
#                      'liqt20rf29', 'as_ap_corr1', 'bp_as_corr1', 'as_ap_corr2', 'bs_bp_corr1', 'max_bid_price1', 'min_bid_price1', 'max_ask_price1']
df_train_reordered.drop(columns=bottom_25_xgb_feat, inplace=True)
df_test = df_test.drop(columns=bottom_25_xgb_feat, inplace=False)


####################### Improvement 2 #######################
## # drop these clusterings as they are least important clustering type
cluster_drop_cols = ['tlog_tlog1p_target_vol_sum_stats_4_clusters',
'tlog_tlog1p_target_vol_sum_stats_10_clusters',
'tlog_tlog1p_target_vol_sum_stats_16_clusters',
'tlog_tlog1p_target_vol_sum_stats_30_clusters',
'sum_stats_4_clusters_labels',
'sum_stats_10_clusters_labels',
'sum_stats_16_clusters_labels',
'sum_stats_30_clusters_labels']
df_train_reordered.drop(columns=cluster_drop_cols, inplace=True)
df_test = df_test.drop(columns=cluster_drop_cols, inplace=False)


####################### Improvement 3 #######################
## # Add shap. interaction terms
interaction_terms_list = [('liqvol1_smean', 'tvpl3_rmed2v1lf20'),
 ('liqvol1_smean', 'tlog_target_vol_pcorr_3_clusters'),
 ('log_liq2_ret_*_wap_eqi_price1_ret_vol_15_ratio_median_stock',
  'tlog_wap1_lprice_ret_vol_ati_*_wap1_lprice_ret_avg_ast_per_wap1_lprice_ret_vol_:0'),
 ('log_liq2_ret_*_wap_eqi_price1_ret_vol_15_ratio_median_stock',
  'tlog_wap1_lprice_ret_vol_ati_*_wap1_lprice_ret_avg_ast_per_wap1_lprice_ret_vol_:10'),
 ('root_trade_count_smean', 'tlog_target_vol_pcorr_3_clusters'),
 ('root_trade_count_smean', 'tvpl3_rmed2v1lf20'),
 ('soft_stock_mean_tvpl2_:20', 'tvpl3_rmed2v1lf20'),
 ('soft_stock_mean_tvpl2_:20', 'tlog_target_vol_pcorr_3_clusters'),
 ('soft_stock_mean_tvpl2_liqf_volf20', 'tvpl3_rmed2v1lf20'),
 ('soft_stock_mean_tvpl2_liqf_volf20', 'tlog_target_vol_pcorr_3_clusters'),
 ('texp_wap1_log_price_ret_vol_from_20',
  'wap1_log_price_ret_quart_volstock_mean_from_25'),
 ('texp_wap1_log_price_ret_vol_from_20',
  'wap1_log_price_ret_quart_volstock_mean_from_20'),
 ('tlog_eps523_trade_price_n_wap_eqi_price0_dev',
  'wap1_log_price_ret_quart_volstock_mean_from_0'),
 ('tlog_eps523_trade_price_n_wap_eqi_price0_dev',
  'wap1_log_price_ret_volstock_mean_from_0'),
 ('tlog_target_vol_pcorr_3_clusters', 'root_trade_count_smean'),
 ('tlog_target_vol_pcorr_3_clusters', 'soft_stock_mean_tvpl2_:0'),
 ('tlog_wap1_lprice_ret_vol_ati_*_wap1_lprice_ret_vol_ast_per_wap1_lprice_ret_vol_:20',
  'v1spprojt15f25'),
 ('tlog_wap1_lprice_ret_vol_ati_*_wap1_lprice_ret_vol_ast_per_wap1_lprice_ret_vol_:20',
  'v1spprojt15f29'),
 ('tvpl2_smean_vol', 'soft_stock_mean_tvpl2_:0'),
 ('tvpl2_smean_vol', 'soft_stock_mean_tvpl2_:10'),
 ('tvpl3_rmed2v1lf20', 'soft_stock_mean_tvpl2_:20'),
 ('tvpl3_rmed2v1lf20', 'soft_stock_mean_tvpl2_:10'),
 ('v1proj_25', 'v1proj_25_15_q1'),
 ('v1proj_25', 'v1proj_25_q1'),
 ('v1proj_29_15_q3', 'root_trade_count_smean'),
 ('v1proj_29_15_q3', 'v1proj_29_q1'),
 ('v1proj_29_q3', 'v1proj_25_15'),
 ('v1proj_29_q3', 'v1proj_29_15_q3'),
 ('v1spprojt15f25_q1', 'root_trade_count_smean'),
 ('v1spprojt15f25_q1', 'v1proj_25'),
 ('wap1_log_price_ret_volstock_mean_from_20', 'v1spprojt15f29'),
 ('wap1_log_price_ret_volstock_mean_from_20', 'v1spprojt15f25'),
 ('wap1_log_price_ret_volstock_mean_from_25', 'v1spprojt15f25'),
 ('wap1_log_price_ret_volstock_mean_from_25', 'v1spprojt15f29'),
 ('tvpl3','v1spprojt15f29')]


# [
# ('tlog_tlinear_sad_ask_size2', 'vol1_mean'),
# ('tlog_tlinear_sad_ask_size2', 'log_wap1_log_price_ret_vol'),
# ('tlog_tlinear_sad_size1', 'tlog_eps523_trade_price_n_wap_eqi_price0_dev'),
# ('tlog_tlinear_sad_size1', 'tlog_eps523_trade_price_n_wap1_dev'),
# ('tlog_eps523_trade_price_n_wap_eqi_price0_dev','log_wap1_log_price_ret_vol'),
# ('tlog_eps523_trade_price_n_wap_eqi_price0_dev', 'tlog_first_10_min_vol'),
# ('tlog_wap1_lprice_ret_vol_ati_*_wap1_lprice_ret_avg_ast_per_wap1_lprice_ret_vol_:20',
#  'log_wap1_log_price_ret_vol'),
# ('tlog_wap1_lprice_ret_vol_ati_*_wap1_lprice_ret_avg_ast_per_wap1_lprice_ret_vol_:20',
#  'tlog_first_10_min_vol'),
# ('tvpl2_rmed2v1', 'tlog_first_10_min_vol'),
# ('tvpl2_rmed2v1', 'log_wap1_log_price_ret_vol'),
# ('wap1_log_price_ret_normalized*mean_centered_per_wap1_lprice_ret_vol_:0',
#  'log_wap1_log_price_ret_vol'),
# ('wap1_log_price_ret_normalized*mean_centered_per_wap1_lprice_ret_vol_:0',
#  'tlog_first_10_min_vol'),
# ('wap1_log_price_ret_volstock_mean_from_25', 'log_wap1_log_price_ret_vol'),
# ('wap1_log_price_ret_volstock_mean_from_25', 'tlog_first_10_min_vol'),
# ('v1spprojt15f25_q1', 'log_wap1_log_price_ret_vol'),
# ('v1spprojt15f25_q1', 'tlog_first_10_min_vol'),
# ('soft_stock_mean_tvpl2_:20', 'wap1_log_price_ret_volstock_mean_from_25'),
# ('soft_stock_mean_tvpl2_:20', 'wap1_log_price_ret_volstock_mean_from_20'),
# ('tlog_target_vol_pcorr_3_clusters', 'soft_stock_mean_tvpl2_:20'),
# ('v1proj_29_15_q3', 'log_wap1_log_price_ret_vol'),
# ('v1proj_29_15_q3', 'tlog_first_10_min_vol'),
# ('root_trade_count_smean', 'tlog_first_10_min_vol'),
# ('root_trade_count_smean', 'log_wap1_log_price_ret_vol'),
# ('tlog_wap1_lprice_ret_vol_ati_*_wap1_lprice_ret_vol_ast_per_wap1_lprice_ret_vol_:20',
#  'log_wap1_log_price_ret_vol'),
# ('tlog_wap1_lprice_ret_vol_ati_*_wap1_lprice_ret_vol_ast_per_wap1_lprice_ret_vol_:20',
#  'tlog_first_10_min_vol'),
# ('wap1_log_price_ret_volstock_mean_from_20', 'log_wap1_log_price_ret_vol'),
# ('wap1_log_price_ret_volstock_mean_from_20', 'tlog_first_10_min_vol')]


## # create and add interaction terms to train and test set
for pair in interaction_terms_list:
    df_train_reordered[f'{pair[0]}_XXX_{pair[1]}'] = df_train_reordered[pair[0]] * df_train_reordered[pair[1]]
    df_test[f'{pair[0]}_XXX_{pair[1]}'] = df_test[pair[0]] * df_test[pair[1]]

# %%
# ####################### find stocks' target similar to stock 31 target in terms of MSE and Spearman correlation #######################
# os.chdir('/home/optimusprime/Desktop/peeterson/optiver/Optiver-Realized-Volatility-Prediction/data')
# #os.chdir('/content/drive/MyDrive/optiver_real_vol/Final Submission/kaggle_submission_versions')

# full_train = pd.read_csv('train.csv')
# full_train_31 = full_train[full_train['stock_id']==31]

# distances_dict = {}

# for stock in full_train['stock_id'].unique()[full_train['stock_id'].unique() != 31]:
#     full_train_stock = full_train[full_train['stock_id']==stock]
#     merged_df = pd.merge(full_train_31, full_train_stock, on='time_id', suffixes=('_31', f'_{stock}')).dropna()
#     MSE = np.mean((merged_df['target_31'] - merged_df[f'target_{stock}'])**2)
#     spearman_corr = 1 - spearmanr(merged_df['target_31'], merged_df[f'target_{stock}'])[0] # 1 - spearman correlation to get distance
#     distances_dict[stock] = (MSE, spearman_corr)


# mse_values = [value[0] for value in distances_dict.values()]
# spearman_values = [value[1] for value in distances_dict.values()]

# # Initialize the MinMaxScaler
# scaler = MinMaxScaler()
# # Reshape the data to fit the scaler
# mse_values = np.array(mse_values).reshape(-1, 1)
# spearman_values = np.array(spearman_values).reshape(-1, 1)
# # Fit and transform the data
# mse_values = scaler.fit_transform(mse_values).flatten()
# spearman_values = scaler.fit_transform(spearman_values).flatten()

# # Replace distances_dict values with standardized mse_values and spearman_values
# for i, stock_id in enumerate(distances_dict.keys()):
#     distances_dict[stock_id] = (mse_values[i], spearman_values[i])

# stock_ids = list(distances_dict.keys())
# plt.figure(figsize=(20, 10))
# plt.scatter(mse_values, spearman_values)
# for i, stock_id in enumerate(stock_ids):
#     plt.annotate(stock_id, (mse_values[i], spearman_values[i]))
# plt.xlabel('Min-max Normalized Mean Squared Error')
# plt.ylabel('Min-max Normalized (1 - Spearman Correlation)')
# plt.title('Stock Distance from Stock 31')
# plt.grid(True)
# plt.show()


# # Calculate the magnitude (Euclidean distance) for each stock
# magnitudes = {stock: np.sqrt(mse**2 + spearman**2) for stock, (mse, spearman) in distances_dict.items()}
# # Sort the stocks by magnitude
# sorted_stocks = sorted(magnitudes.items(), key=lambda item: item[1])
# # Display the sorted stocks
# sorted_stocks[:10]

# %%
# ####################### Improvement 4 #######################
# ## # create seperate training and testing dataframes for stock id 31 using the 5 most similar stocks to stock 31
# ####################### copy these similar stocks to a different dataframe #######################

# similar_stocks = [stock for stock, _ in sorted_stocks[:5]]  # 5 most similar stocks to stock 31
# similar_stocks.append(31)
# similar_stocks = sorted(similar_stocks)

# df_train_reordered_for_stock_31 = df_train_reordered.copy()
# df_test_for_stock_31 = df_test.copy()

# df_train_reordered_for_stock_31 = df_train_reordered_for_stock_31[df_train_reordered_for_stock_31['stock_id'].isin(similar_stocks)]
# df_test_for_stock_31 = df_test_for_stock_31[df_test_for_stock_31['stock_id'].isin(similar_stocks)]

# %%
####################### Improvement 5 #######################
## # drop stock id 31 from training set. 
df_train_reordered.drop(index=df_train_reordered[df_train_reordered['stock_id']==31].index, inplace=True)
df_test.drop(index=df_test[df_test['stock_id']==31].index, inplace=True)
#os.chdir('/home/optimusprime/Desktop/peeterson/optiver/Optiver-Realized-Volatility-Prediction/data/Final_submission_data/partial_train_n_full_test')

# %%
class train_validate_n_test(object):

    def __init__(self,df_train_reordered, df_test) -> None:

        #self.time_id_order = df.loc[:3829,'time_id'].values # select ordered unique time_ids
        #self.train_time_id_ind = int(len(self.time_id_order)*0.7)

        largest_num_time_id_stock_idx = df_train_reordered.groupby('stock_id')['time_id'].apply(lambda x: x.nunique()).argmax()
        largest_num_time_id_stock = df_train_reordered['stock_id'].unique()[largest_num_time_id_stock_idx]
        self.time_id_order = df_train_reordered[df_train_reordered['stock_id'] == largest_num_time_id_stock]['time_id'].values # select reordered unique time_ids
        self.n_folds = 10
        folds = TimeSeriesSplit(n_splits=self.n_folds,)# max_train_size=None, gap=10)
        #self.splits = folds.split( range( self.train_time_id_ind ) ) # split 70% train time_ids into n_fold splits
        nunique_train_time_ids = df_train_reordered['time_id'].nunique()
        self.splits = folds.split( range( nunique_train_time_ids ) )

        #train_time_ids = self.time_id_order[:self.train_time_id_ind]
        #self.train_stock_id = df[df['time_id'].isin(train_time_ids)]['stock_id']
        #self.train_time_id = df[df['time_id'].isin(train_time_ids)]['time_id']
        self.train_stock_id = df_train_reordered['stock_id']
        self.train_time_id = df_train_reordered['time_id']

        # test_time_ids = self.time_id_order[self.train_time_id_ind:]
        # self.test_df = df[df['time_id'].isin(test_time_ids)]
        self.test_time_id = df_test[df_test['stock_id'] == largest_num_time_id_stock]['time_id'].values # select reordered unique time_ids
        self.test_df = df_test
        self.test_stock_id = self.test_df['stock_id']
        self.test_time_id = self.test_df['time_id']

        #self.df = df
        self.df_train_reordered = df_train_reordered

        # feature_importances = pd.DataFrame()
        cols = list(df_train_reordered.columns)
        cols.remove('tlog_target')
        cols.remove('target')
        cols.remove('time_id')
        #cols.remove('stock_id')
        self.feat_cols_list =  cols #cat_feat_labels+float32_feat_labels+float64_feat_labels # int32_feat_labels+int64_feat_labels+float32_feat_labels+float64_feat_labels
        # feature_importances['feature'] = self.feat_cols_list

        self.target_name = 'target' # _standardized' log target is easier to transform back than log_target_standardized

        self.target_shift = 3
        #del df
        del df_train_reordered, df_test
        gc.collect()

    # def onehotencode_cat_var(self,full_set):
    #     full_set = cat_feat_labels #full_set.astype({"stn_id":str,"block_id":str,"ts_of_day":str,"hr_of_day":str,"day_of_wk":str,"day_of_mn":str,"wk_of_mon":str })
    #     full_set = pd.get_dummies(full_set, prefix_sep="_",columns =cat_feat_labels,drop_first=True)
    #     #ds_df = ds_df.drop('rem_blk_outf_'+self.stn,axis=1)
    #     return full_set

    #### RMSPE cost function
    def rmspe(self,y_true, y_pred):
        y_true = np.exp(y_true - self.target_shift) # inverse 'tlog_target' transform
        y_pred = np.exp(y_pred - self.target_shift) # inverse 'tlog_target' transform
        return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))


    # Custom RMSPE objective function
    def rmspe_objective(self,preds, dtrain):
        labels = dtrain.get_label()
        preds = np.exp(preds - self.target_shift) # inverse 'tlog_target' transform
        labels = np.exp(labels - self.target_shift) # inverse 'tlog_target' transform
        errors = (preds - labels) / labels
        gradient = 2 * errors / (1 + errors**2)
        hessian = 2 * (1 - errors**2) / (1 + errors**2)**2
        return gradient, hessian


    def xgb_RMSPE(self,preds, train_data):
        labels = train_data.get_label()
        return 'RMSPE', round(self.rmspe(y_true = labels, y_pred = preds),5)


    def nancorr(self,a, b):
        v = np.isfinite(a)*np.isfinite(b) > 0
        return np.corrcoef(a[v], b[v])[0,1]


    def xgb_train_validate(self,params_xgb,n_rounds,esr,trial):
        rmspe_val_score = []
        #models= []
        test_y_preds = np.zeros(len(self.test_df))
        best_iterations = []
        learning_train_rmspe = []
        learning_val_rmspe = []

        for fold_n, (train_index, valid_index) in enumerate(self.splits):
            print('Fold:',fold_n+1)
            # print('train_index',train_index)
            # print('valid_index',valid_index)
            train_time_ids = self.time_id_order[train_index]
            val_time_ids = self.time_id_order[valid_index]
            train_df = self.df_train_reordered[self.df_train_reordered['time_id'].isin(train_time_ids)]
            val_df = self.df_train_reordered[self.df_train_reordered['time_id'].isin(val_time_ids)]

            X_train = train_df[self.feat_cols_list]
            y_train = train_df[self.target_name] # target
            #train_weight = train_df['target']
            X_valid = val_df[self.feat_cols_list]
            y_val = val_df[self.target_name] # target
            #val_weight = val_df['target']

            v1tr = np.exp(X_train['log_wap1_log_price_ret_vol']) # double exponential to nullify the log
            #v1tr = np.array([1]*len(X_train['log_wap1_log_price_ret_vol']))
            v1v = np.exp(  X_valid['log_wap1_log_price_ret_vol']) # double exponential to nullify the log
            #v1v = np.array([1]*len(X_valid['log_wap1_log_price_ret_vol']))

            # v1tr = np.exp(np.exp(X_train['log_wap1_log_price_ret_vol'])) # double exponential to nullify the log
            # v1v = np.exp(np.exp(  X_valid['log_wap1_log_price_ret_vol'])) # double exponential to nullify the log
            #v1ts = np.exp(np.exp( self.test_df['log_wap1_log_price_ret_vol'])) # double exponential to nullify log

            w_train = y_train **-2 * v1tr**2
            w_val = y_val **-2 * v1v**2

            print('Training....')
            dtrain = xgb.DMatrix(X_train, label=(np.log(y_train/v1tr) + self.target_shift),weight=w_train,enable_categorical=True)
            dvalid = xgb.DMatrix(X_valid,   label= (np.log( y_val/v1v) + self.target_shift),weight=w_val,enable_categorical=True)
            watchlist  = [(dtrain,'train_loss_fold_'+str(fold_n+1)), (dvalid, 'val_loss_fold_'+str(fold_n+1))]
            evals_result = {}
            reg = xgb.train(params=params_xgb, dtrain=dtrain, num_boost_round=n_rounds, evals=watchlist, obj=self.rmspe_objective,custom_metric=self.xgb_RMSPE,  evals_result=evals_result,maximize=False,  early_stopping_rounds=esr,verbose_eval=False)
            learning_train_rmspe.append(evals_result['train_loss_fold_'+str(fold_n+1)])
            learning_val_rmspe.append(evals_result['val_loss_fold_'+str(fold_n+1)])

            #models.append(reg)
            best_iterations.append(reg.best_iteration)

            p = np.exp(reg.predict(dvalid) - self.target_shift) *v1v
            p_inv = p # inverse 'tlog_target' transform
            y_val_inv = y_val # np.exp(y_val - self.target_shift) # inverse 'tlog_target' transform
            val_score = np.mean( ((p_inv-y_val_inv)/y_val_inv)**2 )**0.5

            # full_score += y_val.shape[0]*score**2

            print(f'fold: {fold_n+1}, val rmspe score is {val_score}')
            print('corr(p/v1v, y_val/v1v)',self.nancorr(       p_inv/v1v ,        y_val_inv/v1v ))
            print('log(corr( ))',self.nancorr(np.log(p_inv/v1v), np.log(y_val_inv/v1v)))
            print('corr(p, y_val)',self.nancorr(p_inv, y_val_inv))
            print('log(corr( ))',self.nancorr(np.log(p_inv), np.log(y_val_inv)))

            #test_pred = reg.predict(self.test_df[self.feat_cols_list] )*v1ts ## this method is not suitable for Timeseries cross validation because initial splits are too far from test set.
            #test_y_preds += test_pred/self.n_folds

            rmspe_val_score.append(val_score)

        mean_rmspe_val_score = np.mean(rmspe_val_score)
        print(f'mean rmspe val score over {self.n_folds} splits is',mean_rmspe_val_score)
        #print(f'mean rmspe test score: ',  np.mean( ((test_y_preds-self.test_df[self.target_name])/self.test_df[self.target_name])**2 )**0.5  ) # target

        # Plot learning curves
        fig,ax = plt.subplots(2,1,figsize=(10,6))
        for fold_n in range(len(rmspe_val_score)):
            ax[0].plot(learning_train_rmspe[fold_n]['RMSPE'], label=f'Fold {fold_n+1} Train RMSPE')
            ax[0].plot(learning_val_rmspe[fold_n]['RMSPE'],linestyle='dashed', label=f'Fold {fold_n+1} Validation RMSPE')
        last_fold = len(rmspe_val_score) - 1
        ax[1].plot(learning_val_rmspe[last_fold]['RMSPE'],linestyle='dashed', label=f'Fold {last_fold+1} Validation RMSPE')
        ax[1].set_xlabel('Boosting Round')
        ax[0].set_ylabel('RMSPE')
        ax[1].set_ylabel('RMSPE')
        ax[0].legend()
        ax[1].legend()
        ax[0].grid(True)
        ax[1].grid(True)
        fig.suptitle(f'Learning Curves, Trial: {trial.number}')
        fig.show()

        del self.df_train_reordered, X_train, X_valid, y_train, y_val,train_df,val_df,dtrain,dvalid, v1tr, v1v
        gc.collect()
        return mean_rmspe_val_score,best_iterations[-1]




    def make_predictions(self,best_params,num_rounds ):
        #train_time_ids = self.time_id_order[:self.train_time_id_ind]
        #full_train_df = self.df[self.df['time_id'].isin(train_time_ids)]
        full_train_df = self.df_train_reordered

        X_train = full_train_df[self.feat_cols_list]
        y_train = full_train_df[self.target_name] # target
        y_train_inv = y_train  # no change
        X_test = self.test_df[self.feat_cols_list]
        #train_weight = full_train_df['target']
        #y_test = self.test_df[self.target_name] # target
        #test_weight = self.test_df['target']

        v1tr = np.exp(X_train['log_wap1_log_price_ret_vol']) # double exponential to nullify log
        #v1tr = np.array([1]*len(X_train['log_wap1_log_price_ret_vol']))
        v1ts = np.exp( self.test_df['log_wap1_log_price_ret_vol']) # double exponential to nullify log
        #v1ts = np.array([1]*len(X_test['log_wap1_log_price_ret_vol'])) # double exponential to nullify log
        w_train = y_train **-2 * v1tr**2
        #w_test = test_weight **-2 * v1ts**2

        print('Final model')
        raw_train_gn = np.log(y_train/v1tr) + self.target_shift
        dtrain = xgb.DMatrix(X_train, label= raw_train_gn ,weight=w_train,enable_categorical=True )
        #dtest = xgb.DMatrix(X_test, label=y_test/v1ts,weight=w_test,enable_categorical=True )
        dtest = xgb.DMatrix(X_test,enable_categorical=True )
        watchlist  = [(dtrain,'train_loss')]
        evals_result = {}
        final_reg = xgb.train(params=best_params, dtrain=dtrain, num_boost_round=num_rounds, evals=watchlist, obj=self.rmspe_objective,custom_metric=self.xgb_RMSPE, evals_result=evals_result,maximize=False, verbose_eval=False)
        #test_error = evals_result['test_loss']
        
        raw_train_pred = final_reg.predict(dtrain)
        train_pred = np.exp(raw_train_pred - self.target_shift)*v1tr
        train_pred_inv = train_pred #np.exp(train_pred - self.target_shift) # inverse 'tlog_target' transform
        
        raw_test_pred = final_reg.predict(dtest)
        test_pred = np.exp(raw_test_pred - self.target_shift)*v1ts
        test_pred_inv = test_pred #np.exp(test_pred - self.target_shift) # inverse 'tlog_target' transform


        del full_train_df#,X_train,X_test #,feat_names
        gc.collect()

        return final_reg,train_pred_inv,test_pred_inv,y_train_inv,X_train,X_test,v1tr,v1ts,w_train, self.target_name, raw_train_pred, raw_test_pred,raw_train_gn,self.target_shift




    def local_manual_shapley_additivity_check(self,model_base_value,all_stock_train_pred_df,all_stock_v1tr_df,all_stock_y_train_df,shap_values,stock_id,view_time_ids_start,view_time_ids_end,feature_name):

        y_train_true = all_stock_y_train_df[stock_id].values[view_time_ids_start:view_time_ids_end]
        model_pred = all_stock_train_pred_df[stock_id].values[view_time_ids_start:view_time_ids_end]
                #### ONLY for Explainer
        shap_pred = ( shap_values.base_values + shap_values.values.sum(axis=1) )* all_stock_v1tr_df[stock_id].values[view_time_ids_start:view_time_ids_end]

                #### ONLY for TreeExplainer
        #shap_pred = ( model_base_value + shap_values.sum(axis=1) )* all_stock_v1tr_df[stock_id].values[view_time_ids_start:view_time_ids_end]

        #print('shap_values.sum(axis=1)',shap_values.sum(axis=1))
        # print('shap_values.base_values',shap_values.base_values[0])
        # print('shap_values.values',shap_values.values[0].sum())
        #print('len(shap_values.values.sum(axis=1))',len(shap_values.values.sum(axis=1)))

        model_shap_rmspe = self.rmspe(model_pred, shap_pred)

        fig, ax = plt.subplots(2,1,figsize=(30,10))
        ax[0].plot(np.arange(0,len(y_train_true)),y_train_true,label='true rvol.',linestyle='dashed',c='g',marker='*',alpha=0.2)
        ax[0].plot(np.arange(0,len(model_pred)),model_pred,label='model prediction',linestyle='dashed',c='b',marker='*',alpha=0.6)
        ax[0].set_title(f'True Rvol. Vs. model predicted Rvol.' )
        ax[0].text(0,0.01,f"stock_id: {stock_id}, view_time_ids_start: {view_time_ids_start}, view_time_ids_end:{view_time_ids_end}")
        ax[0].set_ylabel('rvol.')
        ax[0].legend()
        ax[0].grid(True)

        ax[1].plot(np.arange(0,len(model_pred)),model_pred,label='model prediction',linestyle='dashed',c='b',marker='*',alpha=0.4)
        ax[1].plot(np.arange(0,len(shap_pred)),shap_pred,label='summed shap values prediction',linestyle='dashed',c='r',marker='*',alpha=0.4)
        ax[1].set_title(f'Check additivity of shap values, RMSPE:{model_shap_rmspe} between model and shap values prediction' )
        ax[1].text(0,0.01,f"stock_id: {stock_id}, view_time_ids_start: {view_time_ids_start}, view_time_ids_end:{view_time_ids_end}")
        ax[1].set_ylabel('rvol.')
        ax[1].legend()
        ax[1].grid(True)
        fig.tight_layout()
        fig.show()


        del all_stock_train_pred_df,all_stock_v1tr_df,all_stock_y_train_df,shap_values
        gc.collect()
        return


    def compute_shapley_PDP_n_Scatter(self,feature_name,shap_values,groundtruth,prediction,stock_id,view_time_ids_start,view_time_ids_end,set_name):
        ####### compute partial dependence plot of most important features

        ###### Partial dependence plot
        #fig,ax = plt.subplots()
        #shap.plots.partial_dependence(feature_name, model.predict, xgb.DMatrix(X_train,enable_categorical=True), model_expected_value=True, feature_expected_value=True)
        #fig.show()

        ##### scatter plot
        print(f'\n scatter plot of {feature_name} vs. shap values')
        print(f'stock id: {stock_id}, view_time_ids_start:  {view_time_ids_start}, view_time_ids_end: {view_time_ids_end}')

        fig, ax = plt.subplots(2, 2, figsize=(20, 20))
        inds = shap.utils.potential_interactions(shap_values[:, feature_name], shap_values)
        shap.plots.scatter(shap_values[:, feature_name], color=shap_values[:, inds[0]], title=f'scatter plot of {feature_name} Vs. Shap values on {set_name} set', ax=ax[0, 0])
        shap.plots.scatter(shap_values[:, feature_name], color=shap_values[:, inds[1]], ax=ax[0, 1])
        shap.plots.scatter(shap_values[:, feature_name], color=shap_values[:, inds[2]], ax=ax[1, 0])
        plt.show()

        error = groundtruth['target'] - prediction['target']
        ##### scatter plot of feature shap values vs. error
        fig, ax = plt.subplots(figsize=(5, 5))
        print(f'\n scatter plot of {feature_name} shap values vs. error')
        print(f'stock id: {stock_id}, view_time_ids_start:  {view_time_ids_start}, view_time_ids_end: {view_time_ids_end}')
        ax.scatter(error.values,shap_values[:, feature_name].values, alpha=0.4 )
        ax.set_xlabel('error')
        ax.set_ylabel(f'{feature_name} shap values')
        ax.grid()
        ax.set_title(f'scatter plot of {feature_name} shap values Vs. error on {set_name} set')
        plt.tight_layout()
        plt.show()


        ##### scatter plot of feature vs. True target rvol. on trianing set
        # fig,ax = plt.subplots()
        # yval = all_stock_y_train_df[stock_id].values[view_time_ids_start:view_time_ids_end]
        # xval = X[feature_name]
        # ax.scatter(xval,yval)
        # ax.plot([min(xval), max(xval)], [min(yval),max(yval)], color = 'red', linewidth = 1)
        # ax.set_xlabel(feature_name)
        # ax.set_ylabel('True target rvol.')
        # ax.grid
        # ax.set_title(f'scatter plot of {feature_name} Vs. True Rvol. for stock_id: {stock_id}, from {view_time_ids_start} to {view_time_ids_end}')
        # fig.show()


        ##### scatter plot of feature vs. predicted target rvol.on trianing set
        # fig,ax = plt.subplots()
        # yval1 = all_stock_train_pred_df[stock_id].values[view_time_ids_start:view_time_ids_end]
        # ax.scatter(xval,yval1)
        # ax.plot([min(xval), max(xval)], [min(yval1),max(yval1)], color = 'red', linewidth = 1)
        # ax.set_xlabel(feature_name)
        # ax.set_ylabel('Predicted target rvol.')
        # ax.grid
        # ax.set_title(f'scatter plot of {feature_name} Vs. Predicted Rvol. for stock_id: {stock_id}, from {view_time_ids_start} to {view_time_ids_end}')
        # fig.show()


        del shap_values
        gc.collect()
        return



    def compute_shapley_beeswarm(self,shap_values,top_n_feat,stock_id,view_time_ids_start,view_time_ids_end,set_name):

        #### ONLY for TreeExplainer
        # plt.figure()
        # stock_v1tr_df = all_stock_v1tr_df[stock_id].values[view_time_ids_start:view_time_ids_end]
        # shap_values = np.multiply(shap_values.T ,stock_v1tr_df).T
        # shap.summary_plot(shap_values)
        # plt.title(f'stock id: {stock_id}, view_time_ids_start:  {view_time_ids_start}, view_time_ids_end: {view_time_ids_end}')
        # plt.show()

        #### ONLY for Explainer
        print(f'\n stock id: {stock_id}, view_time_ids_start:  {view_time_ids_start}, view_time_ids_end: {view_time_ids_end}')
        fig,ax = plt.subplots()
        ax.set_title(f'Beeswarm plot for {set_name} set showing top {top_n_feat} features')
        shap.plots.beeswarm(shap_values, max_display=top_n_feat)
        #ax.set_title(f' stock id: {stock_id}, view_time_ids_start:  {view_time_ids_start}, view_time_ids_end: {view_time_ids_end}')
        plt.show()

        del shap_values
        gc.collect()
        return



    def compute_shapley_barplot(self,shap_values,top_n_feats,X,Y,stock_id,view_time_ids_start,view_time_ids_end,set_name):

        #### ONLY for TreeExplainer
        # plt.figure()
        # stock_v1tr_df = all_stock_v1tr_df[stock_id].values[view_time_ids_start:view_time_ids_end]
        # shap_values = np.multiply(shap_values.T ,stock_v1tr_df).T
        # plt.bar(shap_values.abs().sum(axis=1))
        # plt.title(f'stock id: {stock_id}, view_time_ids_start:  {view_time_ids_start}, view_time_ids_end: {view_time_ids_end}')
        # plt.show()

        #### ONLY for Explainer
        print(f'\nMEAN ABSOLUTE of feature bar plot \n stock id: {stock_id}, view_time_ids_start:  {view_time_ids_start}, view_time_ids_end: {view_time_ids_end}')
        fig,ax = plt.subplots()
        ax.set_title(f'MEAN ABSOLUTE of feature bar plot on {set_name} set showing top {top_n_feats} features')
        #clustering = shap.utils.hclust(X,Y)
        shap.plots.bar(shap_values, max_display=top_n_feats)#,clustering=clustering,clustering_cutoff=0.9)
        #ax.title(f'MEAN ABSOLUTE of feature bar plot \n stock id: {stock_id}, view_time_ids_start:  {view_time_ids_start}, view_time_ids_end: {view_time_ids_end}')
        fig.show()


        #### ONLY for Explainer
        print(f'\nMAXIMUM ABSOLUTE of feature bar plot \n stock id: {stock_id}, view_time_ids_start:  {view_time_ids_start}, view_time_ids_end: {view_time_ids_end}')
        fig,ax = plt.subplots()
        #clustering = shap.utils.hclust(X,Y)
        ax.set_title(f'MAXIMUM ABSOLUTE of feature bar plot on {set_name} set showing top {top_n_feats} features')
        shap.plots.bar(shap_values.abs.max(0), max_display=top_n_feats)#,clustering=clustering,clustering_cutoff=0.9)
        #ax.title(f'nMAXIMUM ABSOLUTE of feature bar plot \n stock id: {stock_id}, view_time_ids_start:  {view_time_ids_start}, view_time_ids_end: {view_time_ids_end}')
        fig.show()

        del shap_values, X,Y,stock_id,view_time_ids_start,view_time_ids_end,set_name,fig,ax,top_n_feats
        gc.collect()
        return


    def compute_individual_stock_SHAP_values(self,final_reg,X_train,all_stock_train_pred_df,all_stock_v1tr_df,all_stock_y_train_df,feature_name,stock_id,view_time_ids_start,view_time_ids_end):

        # plot shapley feature importances for all samples
        final_reg.set_param({"device": "cuda"})
        shap.initjs()

        stock_id = stock_id
        view_time_ids_start = view_time_ids_start
        view_time_ids_end = view_time_ids_end
        X = X_train[X_train['stock_id'].isin([stock_id])].iloc[view_time_ids_start:view_time_ids_end]

        ###### Explainer #######
        #explainer = shap.Explainer(final_reg,X)
        explainer = shap.TreeExplainer(final_reg, feature_perturbation="tree_path_dependent")
        shap_values = explainer(np.array(X),check_additivity=False)
        shap_values.feature_names = final_reg.feature_names

        ###### TreeExplainer #######
        # explainer = shap.TreeExplainer(final_reg,feature_perturbation='interventional')
        # shap_values = explainer.shap_values(np.array(X),check_additivity=False)
        # shap_values.feature_names = final_reg.feature_names

        model_base_value = explainer.expected_value
        # print(f'Model base value: {model_base_value} before scaling by v1tr')

        ####### GLOBAL ALL feature contributions ##############################
        ###### Do manual additivity check because it fails
        self.local_manual_shapley_additivity_check(model_base_value,all_stock_train_pred_df,all_stock_v1tr_df,all_stock_y_train_df,shap_values,stock_id,view_time_ids_start,view_time_ids_end,feature_name )

        ####### Manually correct the shap values to accomodate v1tr scaling
        shap_values.base_values = shap_values.base_values * all_stock_v1tr_df[stock_id].values[view_time_ids_start:view_time_ids_end] # all_stock_v1tr_df.iloc[view_time_ids_start:view_time_ids_end,stock_id].values
        stock_v1tr_df = all_stock_v1tr_df[stock_id].values[view_time_ids_start:view_time_ids_end] # all_stock_v1tr_df.iloc[view_time_ids_start:view_time_ids_end,stock_id].values
        shap_values.values = np.multiply(shap_values.values.T ,stock_v1tr_df).T
        ###### check correctness of shap_values
        # sp = shap_values.base_values + shap_values.values.sum(axis=1)
        # plt.figure(figsize=(30,5))
        # plt.plot(range(len(sp)),sp)
        # model_pred = all_stock_train_pred_df[stock_id].values[view_time_ids_start:view_time_ids_end]
        # plt.plot(range(len(sp)), model_pred )
        # plt.show()

        self.compute_shapley_beeswarm(X,shap_values,all_stock_v1tr_df,stock_id,view_time_ids_start,view_time_ids_end)

        #shap_values = final_reg.predict(dtrain, pred_contribs=True)
        ### Calculate SHAP values for a specific instance (e.g., the first test instance)
        ### shap_values = explainer.shap_values(X_test.iloc[0])

        self.compute_shapley_barplot(shap_values,all_stock_v1tr_df,stock_id,view_time_ids_start,view_time_ids_end)

        self.compute_shapley_heatmap(shap_values,stock_id,view_time_ids_start,view_time_ids_end,all_stock_train_pred_df)

        ####### INDIVIDUAL feature contributions ##############################
        ####### compute partial dependence plot of most important features
        self.compute_shapley_PDP_n_Scatter(feature_name,shap_values,stock_id,view_time_ids_start,view_time_ids_end,X,all_stock_y_train_df,all_stock_train_pred_df)

        #self.compute_shapley_decision(model_base_value,shap_values.data,shap_values.feature_names,stock_id,view_time_ids_start,view_time_ids_end)

        ##### force plot has some error
        #self.compute_shapley_force(model_base_value,shap_values.data,X,shap_values.feature_names,stock_id,view_time_ids_start,view_time_ids_end)


        del final_reg,X_train
        gc.collect()
        return



    def identiy_largest_overall_n_under_n_over_prediction_errors(self,groundtruth,prediction,set_name):

        # overall error
        oveall_error = groundtruth['target'] - prediction['target']
        # overall squared percentage error
        ovearll_SPE = pd.DataFrame()
        ovearll_SPE['target_spe'] = (oveall_error / groundtruth['target'] )**2
        ovearll_SPE['stock_id'] = groundtruth['stock_id']
        ovearll_SPE['time_id'] = groundtruth['time_id']

        # sort the overall_SPE by descending order
        top_n_instances = 50
        sorted_ovearll_SPE = ovearll_SPE.sort_values('target_spe',ascending=False)[:top_n_instances].reset_index(drop=True)
        # Plot the sorted_ovearll_SPE as a bar plot with dual x-axis for stock_id and time_id
        fig, ax1 = plt.subplots(figsize=(15, 8))
        # Plotting the bar graph for 'target_spe'
        ax1.bar(sorted_ovearll_SPE.index, sorted_ovearll_SPE['target_spe'], color='b', alpha=0.6)
        ax1.set_xlabel('Instance Index')
        ax1.set_ylabel('Target Squared Percentage Error', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        # Creating a twin Axes sharing the x-axis
        ax2 = ax1.twiny()
        # Setting the x-ticks and labels for stock_id and time_id
        ax2.set_xticks(sorted_ovearll_SPE.index)
        ax2.set_xticklabels(sorted_ovearll_SPE['stock_id'], rotation=90, ha='center')
        ax2.set_xlabel('Stock ID')
        # Creating another twin Axes sharing the x-axis
        ax3 = ax1.twiny()
        # Offset the twin axis below the original x-axis
        ax3.spines['top'].set_position(('outward', 40))
        ax3.set_xticks(sorted_ovearll_SPE.index)
        ax3.set_xticklabels(sorted_ovearll_SPE['time_id'], rotation=90, ha='center')
        ax3.set_xlabel('Time ID')
        plt.title(f'{top_n_instances} Largest Target SPE instances with Stock ID and Time ID on ' + set_name + ' Set')
        plt.show()

        overall_RMSPE = np.sqrt(np.mean(ovearll_SPE['target_spe']))
        corrected_overall_SPE = ovearll_SPE.sort_values('target_spe',ascending=False)[top_n_instances:].reset_index(drop=True)
        corrected_overall_RMSPE = np.sqrt(np.mean(corrected_overall_SPE['target_spe']))
        print(f'\nOverall RMSPE: {overall_RMSPE} on {set_name} set')
        print(f'Corrected RMSPE: {corrected_overall_RMSPE} on {set_name} set')
        print(f'Percentage improvment in RMSPE of ovearall error on {set_name} set after correcting the top {top_n_instances} instances:')
        print(f'{(overall_RMSPE - corrected_overall_RMSPE )/overall_RMSPE*100}%')


        # underprediction squared percentage error
        up_SPE = ovearll_SPE[oveall_error > 0].reset_index(drop=True)
        # sort the up_SPE by descending order
        sorted_up_SPE = up_SPE.sort_values('target_spe',ascending=False)[:top_n_instances].reset_index(drop=True)
        # Plot the sorted_up_SPE as a bar plot with dual x-axis for stock_id and time_id
        fig, ax1 = plt.subplots(figsize=(15, 8))
        # Plotting the bar graph for 'target_spe'
        ax1.bar(sorted_up_SPE.index, sorted_up_SPE['target_spe'], color='b', alpha=0.6)
        ax1.set_xlabel('Instance Index')
        ax1.set_ylabel('Target Squared Percentage Error', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        # Creating a twin Axes sharing the x-axis
        ax2 = ax1.twiny()
        # Setting the x-ticks and labels for stock_id and time_id
        ax2.set_xticks(sorted_up_SPE.index)
        ax2.set_xticklabels(sorted_up_SPE['stock_id'], rotation=90, ha='center')
        ax2.set_xlabel('Stock ID')
        # Creating another twin Axes sharing the x-axis
        ax3 = ax1.twiny()
        # Offset the twin axis below the original x-axis
        ax3.spines['top'].set_position(('outward', 40))
        ax3.set_xticks(sorted_up_SPE.index)
        ax3.set_xticklabels(sorted_up_SPE['time_id'], rotation=90, ha='center')
        ax3.set_xlabel('Time ID')
        plt.title(f'{top_n_instances} Largest target SPE Underprediction instances with Stock ID and Time ID on ' + set_name + ' Set')
        plt.show()

        up_RMSPE = np.sqrt(np.mean(up_SPE['target_spe']))
        corrected_up_SPE = up_SPE.sort_values('target_spe',ascending=False)[top_n_instances:].reset_index(drop=True)
        corrected_up_RMSPE = np.sqrt(np.mean(corrected_up_SPE['target_spe']))
        print(f'\nUnderprediction RMSPE: {up_RMSPE} on {set_name} set')
        print(f'Corrected Underprediction RMSPE: {corrected_up_RMSPE} on {set_name} set')
        print(f'Percentage improvment in RMSPE of underprediction error on {set_name} set after correcting the top {top_n_instances} instances:')
        print(f'{(up_RMSPE - corrected_up_RMSPE )/up_RMSPE*100}%')


        # overprediction squared percentage error
        op_SPE = ovearll_SPE[oveall_error < 0].reset_index(drop=True)
        # sort the op_SPE by descending order
        sorted_op_SPE = op_SPE.sort_values('target_spe',ascending=False)[:top_n_instances].reset_index(drop=True)
        # Plot the sorted_op_SPE as a bar plot with dual x-axis for stock_id and time_id
        fig, ax1 = plt.subplots(figsize=(15, 8))
        # Plotting the bar graph for 'target_spe'
        ax1.bar(sorted_op_SPE.index, sorted_op_SPE['target_spe'], color='b', alpha=0.6)
        ax1.set_xlabel('Instance Index')
        ax1.set_ylabel('Target Squared Percentage Error', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        # Creating a twin Axes sharing the x-axis
        ax2 = ax1.twiny()
        # Setting the x-ticks and labels for stock_id and time_id
        ax2.set_xticks(sorted_op_SPE.index)
        ax2.set_xticklabels(sorted_op_SPE['stock_id'], rotation=90, ha='center')
        ax2.set_xlabel('Stock ID')
        # Creating another twin Axes sharing the x-axis
        ax3 = ax1.twiny()
        # Offset the twin axis below the original x-axis
        ax3.spines['top'].set_position(('outward', 40))
        ax3.set_xticks(sorted_op_SPE.index)
        ax3.set_xticklabels(sorted_op_SPE['time_id'], rotation=90, ha='center')
        ax3.set_xlabel('Time ID')
        plt.title(f'{top_n_instances} Largest target SPE Overprediction instances with Stock ID and Time ID on ' + set_name + ' Set')
        plt.show()

        op_RMSPE = np.sqrt(np.mean(op_SPE['target_spe']))
        corrected_op_SPE = op_SPE.sort_values('target_spe',ascending=False)[top_n_instances:].reset_index(drop=True)
        corrected_op_RMSPE = np.sqrt(np.mean(corrected_op_SPE['target_spe']))
        print(f'\nOverprediction RMSPE: {op_RMSPE} on {set_name} set')
        print(f'Corrected Overprediction RMSPE: {corrected_op_RMSPE} on {set_name} set')
        print(f'Percentage improvment in RMSPE of overprediction error on {set_name} set after correcting the top {top_n_instances} instances:')
        print(f'{(op_RMSPE - corrected_op_RMSPE )/op_RMSPE*100}%')


        ovearll_error_idxs = []
        up_error_idxs = []
        op_error_idxs = []
        for i in range(top_n_instances):
            ovearll_error_idxs.append(groundtruth[(groundtruth['stock_id'] == sorted_ovearll_SPE['stock_id'][i]) & (groundtruth['time_id'] == sorted_ovearll_SPE['time_id'][i])].index[0])
            up_error_idxs.append(groundtruth[(groundtruth['stock_id'] == sorted_up_SPE['stock_id'][i]) & (groundtruth['time_id'] == sorted_up_SPE['time_id'][i])].index[0])
            op_error_idxs.append(groundtruth[(groundtruth['stock_id'] == sorted_op_SPE['stock_id'][i]) & (groundtruth['time_id'] == sorted_op_SPE['time_id'][i])].index[0])

        return ovearll_error_idxs, up_error_idxs, op_error_idxs



    def shapley_analysis_of_large_error_instances(self,ovearll_error_idxs, up_error_idxs, op_error_idxs,shap_values_all,set_name):

        print('shap_values_all.feature_names',len(shap_values_all.feature_names))

        # decision plot of the largest overall error instances
        display_n_features = 40
        fig, ax = plt.subplots(figsize=(50,10))
        shap.decision_plot(np.mean(shap_values_all.base_values[ovearll_error_idxs]), shap_values_all.values[ovearll_error_idxs,:], \
                           feature_names=shap_values_all.feature_names, feature_order='importance', feature_display_range= slice(-1,-display_n_features,-1),\
                           title=f'Decision plot of the {display_n_features} largest overall error instances on {set_name} set \n (ignore the WRONG expected value shown)')
        plt.show()

        # decision plot of the largest overprediction error instances
        fig, ax = plt.subplots(figsize=(50,10))
        shap.decision_plot(np.mean(shap_values_all.base_values[op_error_idxs]), shap_values_all.values[op_error_idxs,:], \
                            feature_names=shap_values_all.feature_names, feature_order='importance', feature_display_range= slice(-1,-30,-1),\
                            title=f'Decision plot of the {display_n_features} largest overprediction error instances on {set_name} set \n (ignore the WRONG expected value shown)')
        plt.show()

        # decision plot of the largest underprediction error instances
        fig, ax = plt.subplots(figsize=(50,10))
        shap.decision_plot(np.mean(shap_values_all.base_values[up_error_idxs]), shap_values_all.values[up_error_idxs,:], \
                            feature_names=shap_values_all.feature_names, feature_order='importance', feature_display_range= slice(-1,-30,-1),\
                            title=f'Decision plot of the {display_n_features} largest underprediction error instances on {set_name} set \n (ignore the WRONG expected value shown)')
        plt.show()

        del shap_values_all,ovearll_error_idxs, up_error_idxs, op_error_idxs,set_name
        return




    def compute_shap_interaction_values(self, explainer,X_train,y_train,v1tr):

        w_train = y_train **-2 * v1tr**2
        raw_train_gn = np.log(y_train/v1tr) + self.target_shift
        dtrain = xgb.DMatrix(X_train, label= raw_train_gn ,weight=w_train,enable_categorical=True )

        shap_interaction_values = explainer.shap_interaction_values(dtrain)
        
        # # Sum the absolute interaction values across all samples
        # interaction_sum = np.abs(shap_interaction_values).sum(axis=0)
        # # Create a matrix of feature pairs by summing across rows and columns
        # interaction_matrix = np.sum(interaction_sum, axis=0)
        # # Get indices of top 2 interactions (ignoring diagonal elements)
        # interaction_matrix[np.diag_indices_from(interaction_matrix)] = 0  # remove self-interactions
        # top_2_indices = np.unravel_index(np.argsort(interaction_matrix.ravel())[-2:], interaction_matrix.shape)
        # # Print the top 2 feature pairs
        # top_2_interactions = [(X_train.columns[i], X_train.columns[j]) for i, j in zip(top_2_indices[0], top_2_indices[1])]
        # #print(f"Top 2 interaction features: {top_2_interactions}")

        # Sum the absolute interaction values across all samples
        interaction_sum = np.abs(shap_interaction_values).sum(axis=0)
        # Create a matrix of feature pairs by summing across rows and columns
        interaction_matrix = np.sum(interaction_sum, axis=0)
        # Remove self-interactions (diagonal elements)
        np.fill_diagonal(interaction_matrix, 0)
        # Dictionary to store the top 2 interacting features for each feature
        top_interactions_dict = {}
        # Get feature names from the dataset (assuming X_train is a DataFrame)
        feature_names = X_train.columns

        # Loop over each feature to find its top 2 interacting features
        for i in range(interaction_matrix.shape[0]):
            # Get the top 2 interacting features for feature i
            top_2_interactions = np.argsort(interaction_matrix[i])[-2:][::-1]  # Get indices of top 2 features
            # Store in dictionary
            top_interactions_dict[feature_names[i]] = [feature_names[j] for j in top_2_interactions]

        del explainer,X_train, y_train, v1tr, w_train, raw_train_gn, dtrain, interaction_sum, interaction_matrix
        return shap_interaction_values,top_interactions_dict



    def compute_global_SHAP_values(self,final_reg,X_train,y_train,train_pred,v1tr,set_name):

        print(f'\nGround-Truth RAW Rvol. grand average on {set_name} set: {y_train["target"].values.mean()}')
        print(f'\nModel Prediction RAW Rvol. grand average on {set_name} set: {train_pred["target"].values.mean()}')

        # plot shapley feature importances for all samples
        final_reg.set_param({"device": "cuda"})
        shap.initjs()

        X = X_train

        ###### Explainer #######
        #explainer = shap.Explainer(final_reg,X)
        explainer = shap.TreeExplainer(final_reg, feature_perturbation="tree_path_dependent")
        shap_values_all = explainer(np.array(X),check_additivity=False)
        shap_values_all.feature_names = final_reg.feature_names

        model_base_value = explainer.expected_value

        # ######### Calculate SHAP interaction values #########
        shap_interaction_values,top_2_interactions = self.compute_shap_interaction_values(explainer,X_train,y_train['target'],v1tr['wap1_log_price_ret_vol'].values)
        #shap_interaction_values,top_2_interactions = [],[]

        ####### GLOBAL ALL feature contributions ##############################
        ###### Do manual additivity check because it fails
        print(f'\n -------------------- NOTICE ------------------------')
        print(f'Shapley additivity check is performed on the RAW target values (log of jager transform)')
        print(f'\n -------------------- NOTICE ------------------------')
        self.global_manual_shapley_additivity_check(train_pred,v1tr,shap_values_all,set_name)

        ####### Manually correct the shap values to accomodate v1tr scaling
        # shap_values_all.base_values = np.exp(shap_values_all.base_values - self.target_shift) * v1tr['wap1_log_price_ret_vol'].values  # all_stock_v1tr_df.iloc[view_time_ids_start:view_time_ids_end,stock_id].values
        # shap_values_all.values = np.multiply(np.exp(shap_values_all.values.T - self.target_shift),v1tr['wap1_log_price_ret_vol'].values).T

        ###### Beeswarm plot
        #### ONLY for Explainer
        print(f'\n Global Beeswarm plot for all stock ids and time ids')
        top_n_feat = 25
        self.compute_shapley_beeswarm(shap_values_all,top_n_feat,stock_id=None,view_time_ids_start=None,view_time_ids_end=None,set_name=set_name)

        ###### Bar plot MEAN Absolute value of features
        #### ONLY for Explainer
        print(f'\nMEAN ABSOLUTE of feature bar plot for all stock ids and time ids')
        #clustering = shap.util.hclust(X,y)
        shap.plots.bar(shap_values_all)# clustering=clustering)
        #ax.title(f'MEAN ABSOLUTE of feature bar plot for all stock ids and time ids')

        ###### Bar plot MAXIMUM Absolute value of features
        #### ONLY for Explainer
        print(f'\nMAXIMUM ABSOLUTE of feature bar plot for all stock ids and time ids')
        #clustering = shap.util.hclust(X,y)
        shap.plots.bar(shap_values_all.abs.max(0), )#clustering=clustering)
        #ax.title(f'nMAXIMUM ABSOLUTE of feature bar plot for all stock ids and time ids')

        del final_reg,X_train,y_train,train_pred,v1tr,X, model_base_value, set_name, top_n_feat, shap_interaction_values
        gc.collect()
        return shap_values_all,top_2_interactions


    def global_manual_shapley_additivity_check(self,train_pred,v1tr,shap_values_all, set_name):
        #### ONLY for Explainer
        # shap_pred_all = ( shap_values_all.base_values + shap_values_all.values.sum(axis=1) ) * v1tr['wap1_log_price_ret_vol'] #pd.DataFrame(all_stock_v1tr_df.values.ravel() , columns=['v1tr_all'])['v1tr_all'].values
        # model_shap_rmspe_all = self.rmspe(train_pred['target'], shap_pred_all)

        shap_pred_all = ( shap_values_all.base_values + shap_values_all.values.sum(axis=1) ) 
        model_shap_rmspe_all = self.rmspe(train_pred['target'], shap_pred_all)

        # # line plot of model prediction vs. shap values prediction
        # fig, ax = plt.subplots(figsize=(30,10))
        # ax.plot(np.arange(0,len(train_pred['target'])),train_pred['target'],label='prediction',linestyle='dashed',c='g',marker='*',alpha=0.2)
        # ax.plot(np.arange(0,len(shap_pred_all)),shap_pred_all,label='shap_value',linestyle='dashed',c='b',marker='*',alpha=0.6)
        # ax.set_title(f'Model prediction Vs. Shap_values for additivity check on {set_name} set' )
        # ax.set_ylabel('rvol.')
        # ax.legend()
        # ax.grid(True)
        # plt.show()
        # plt.close()

        # scatter plot of model prediction vs. shap values prediction
        fig, ax = plt.subplots(figsize=(30,10))
        ax.scatter(train_pred['target'],shap_pred_all)
        ax.plot([min(train_pred['target']), max(train_pred['target'])], [min(shap_pred_all),max(shap_pred_all)], color = 'red', linewidth = 1)
        ax.set_xlabel('model prediction')
        ax.set_ylabel('shap prediction')
        ax.grid
        ax.set_title(f'scatter plot of model prediction Vs. shap prediction on {set_name} set')
        plt.show()
        plt.close()

        print(f'\n Check Additivity of shap values in all stock and time ids, model_shap_rmspe_all: {model_shap_rmspe_all} on {set_name} set')
        del train_pred,v1tr,shap_values_all
        gc.collect()
        return




    def analyze_global_SHAP_values(self,shap_values_all,groundtruth, prediction,set_name):

        ###################################### All Error vs. feature Shapley values Analysis #######################################
        ## filter out features that have high negative and positively shapley values and see if they are correlated with the error

        top_largest_shap_val_feat = 100
        error = groundtruth['target'] - prediction['target']
        # select only the features that have high shapley values over all instances/samples
        abs_summed_shap = np.abs(shap_values_all.values).sum(axis=0)
        sorted_summed_shap = np.sort(abs_summed_shap)[::-1]
        labels = np.array(shap_values_all.feature_names)[np.argsort(abs_summed_shap)[::-1]]
        top_n_feat = 50
        print(f'\nTop {top_n_feat} features based on summed absolute SHAP values over ALL instances/samples on {set_name} set')
        print(list(labels[:top_n_feat]))
        bot_n_feat = 25
        print(f'Bottom {bot_n_feat} features based on summed absolute SHAP values over ALL instances/samples on {set_name} set')
        print(list(labels[-bot_n_feat:]))

        # from the features that have high shapley values over all instances/samples
        # find out feature shapley values that are negatively correlated with the error
        feat_error_corr_dict = {}
        for feat in labels[:top_largest_shap_val_feat]:
            # if self.nancorr(shap_values_all.values[:,shap_values_all.feature_names.index(feat)], error) < 0:
            #     feat_error_corr_dict[feat] = self.nancorr(shap_values_all.values[:,shap_values_all.feature_names.index(feat)], error)
            corr, _ = spearmanr(shap_values_all.values[:, shap_values_all.feature_names.index(feat)], error)
            if corr < 0:
                feat_error_corr_dict[feat] = corr

        # sort the features based on the most negative correlation with the error first
        sorted_feat_error_corr_dict = dict(sorted(feat_error_corr_dict.items(), key=lambda item: item[1]))

        # scatter plot of features shapley values vs. error over all instances/samples for the top_n_feat features that have high negative correlation with the error
        top_n_feat = 28
        fig, ax = plt.subplots(int(top_n_feat/4), 4, figsize=(30, 30))
        fig.suptitle(f"{set_name} set", fontsize=16)
        for feat in list(sorted_feat_error_corr_dict.keys())[:top_n_feat]:
            ax.flatten()[list(sorted_feat_error_corr_dict.keys()).index(feat)].scatter(shap_values_all.values[:,shap_values_all.feature_names.index(feat)], error, alpha=0.1)
            ax.flatten()[list(sorted_feat_error_corr_dict.keys()).index(feat)].set_title(f"{feat} shap values vs. error \n, spear. corr: {sorted_feat_error_corr_dict[feat]}")
            ax.flatten()[list(sorted_feat_error_corr_dict.keys()).index(feat)].set_xlabel('error')
        fig.tight_layout()
        plt.show()

        print(f'\nTop {top_n_feat} features that have high negative correlation with the error on {set_name} set')
        print(list(sorted_feat_error_corr_dict.keys())[:top_n_feat])


        ###################################### Underprediction Error Shapley Analysis #######################################
        ## filter out instance where underpreidction is happening, then identfy the features that are causing the underprediction
        ## do not identfiy the features on the entire dataset.

        top_n_feat = 50
        underprediction = prediction['target'] < groundtruth['target']
        bad_feat_up_idx = (prediction['target'] < groundtruth['target']) & (prediction['target'] > shap_values_all.base_values)
        good_feat_up_idx = (prediction['target'] < groundtruth['target']) & (prediction['target'] < shap_values_all.base_values)

        # abs_summed_shap = np.abs(shap_values_all.values[underprediction]).sum(axis=0)
        # sorted_summed_shap = np.sort(abs_summed_shap)[::-1]
        # labels = np.array(shap_values_all.feature_names)[np.argsort(abs_summed_shap)[::-1]]
        # fig = px.bar(x=sorted_summed_shap[:top_n_feat], y=labels[:top_n_feat], orientation='h', title=f"{set_name} set's UNDERprediction instances, Feature importance based on summed absolute SHAP values", height=800)
        # fig.update_layout(yaxis=dict(tickfont=dict(size=10)))  # Reduce the font size of the labels
        # fig.show()
        # print(f'\nTop {top_n_feat} features that have high absolute shapley values over all instances/samples on {set_name} set for UNDERprediction instances')
        # print(list(labels[:top_n_feat]))

        #simple_summed_shap = shap_values_all.values[underprediction].sum(axis=0)
        simple_summed_shap = shap_values_all.values[bad_feat_up_idx].sum(axis=0)        
        sorted_summed_shap = np.sort(simple_summed_shap)
        labels = np.array(shap_values_all.feature_names)[np.argsort(simple_summed_shap)]
        #fig = px.bar(x=sorted_summed_shap[:top_n_feat], y=labels[:top_n_feat], orientation='h', title=f"{set_name} set's UNDERprediction instances, Feature importance based on simply summed SHAP values, Asc. order", height=800)
        fig = px.bar(x=sorted_summed_shap[:top_n_feat], y=labels[:top_n_feat], orientation='h', title=f"{set_name} set's UNDERprediction instances with prediction > base value, Truly BAD Feature importance based on simply summed SHAP values, Asc. order", height=800)
        fig.update_layout(yaxis=dict(tickfont=dict(size=10)))  # Reduce the font size of the labels
        fig.show()
        print(f'\nTop {top_n_feat} features that have high simply summed shapley values over all instances/samples on {set_name} set for UNDERprediction instances, Asc. order')
        print(list(labels[:top_n_feat]))

        #simple_summed_shap = shap_values_all.values[underprediction].sum(axis=0)
        simple_summed_shap = shap_values_all.values[good_feat_up_idx].sum(axis=0)
        sorted_summed_shap = np.sort(simple_summed_shap)[::-1]
        labels = np.array(shap_values_all.feature_names)[np.argsort(simple_summed_shap)[::-1]]
        # fig = px.bar(x=sorted_summed_shap[:top_n_feat], y=labels[:top_n_feat], orientation='h', title=f"{set_name} set's UNDERprediction instances, Feature importance based on simply summed SHAP values, Desc. order", height=800)
        fig = px.bar(x=sorted_summed_shap[:top_n_feat], y=labels[:top_n_feat], orientation='h', title=f"{set_name} set's UNDERprediction instances with prediction < base value, Truly GOOD Feature importance based on simply summed SHAP values, Desc. order", height=800)
        fig.update_layout(yaxis=dict(tickfont=dict(size=10)))  # Reduce the font size of the labels
        fig.show()
        print(f'\nTop {top_n_feat} features that have high simply summed shapley values over all instances/samples on {set_name} set for UNDERprediction instances, Desc. order')
        print(list(labels[:top_n_feat]))


        # min_shap_values = shap_values_all.values[underprediction].min(axis=0)
        # sorted_min_shap = np.sort(min_shap_values)
        # labels = np.array(shap_values_all.feature_names)[np.argsort(min_shap_values)]
        # fig = px.bar(x=sorted_min_shap[:top_n_feat], y=labels[:top_n_feat], orientation='h', title=f"{set_name} set's UNDERprediction instances, Feature importance based on min SHAP values", height=800)
        # fig.update_layout(yaxis=dict(tickfont=dict(size=10)))  # Reduce the font size of the labels
        # fig.show()
        # print(f'\nTop {top_n_feat} features that have high min shapley values over all instances/samples on {set_name} set for UNDERprediction instances')
        # print(list(labels[:top_n_feat]))


        #Examine Individual Feature that lead to most negative shap values Contributions using shap dependence plot
        #self.compute_shapley_PDP_n_Scatter(feature_name,shap_values,stock_id,view_time_ids_start,view_time_ids_end,X,all_stock_y_train_df,all_stock_train_pred_df)




        ###################################### Overprediction Error Shapley Analysis #######################################

        ## filter out instance where underpreidction is happening, then identfy the features that are causing the overprediction
        ## do not identfiy the features on the entire dataset.
        top_n_feat = 50
        overprediction = prediction['target'] > groundtruth['target']
        bad_feat_op_idx = (prediction['target'] > groundtruth['target']) & (prediction['target'] < shap_values_all.base_values)
        good_feat_op_idx = (prediction['target'] > groundtruth['target']) & (prediction['target'] > shap_values_all.base_values)


        # abs_summed_shap = np.abs(shap_values_all.values[overprediction]).sum(axis=0)
        # sorted_summed_shap = np.sort(abs_summed_shap)[::-1]
        # labels = np.array(shap_values_all.feature_names)[np.argsort(abs_summed_shap)[::-1]]
        # fig = px.bar(x=sorted_summed_shap[:top_n_feat], y=labels[:top_n_feat], orientation='h', title=f"{set_name} set's OVERprediction instances, Feature importance based on summed absolute SHAP values", height=800)
        # fig.update_layout(yaxis=dict(tickfont=dict(size=10)))  # Reduce the font size of the labels
        # fig.show()
        # print(f'\nTop {top_n_feat} features that have high absolute shapley values over all instances/samples on {set_name} set for OVERprediction instances')
        # print(list(labels[:top_n_feat]))

        #simple_summed_shap = shap_values_all.values[overprediction].sum(axis=0)
        simple_summed_shap = shap_values_all.values[good_feat_op_idx].sum(axis=0)
        sorted_summed_shap = np.sort(simple_summed_shap)
        labels = np.array(shap_values_all.feature_names)[np.argsort(simple_summed_shap)]
        # fig = px.bar(x=sorted_summed_shap[:top_n_feat], y=labels[:top_n_feat], orientation='h', title=f"{set_name} set's OVERprediction instances, Feature importance based on simply summed SHAP values, Asc. order", height=800)
        fig = px.bar(x=sorted_summed_shap[:top_n_feat], y=labels[:top_n_feat], orientation='h', title=f"{set_name} set's OVERprediction instances with prediction > base value, Truly GOOD Feature importance based on simply summed SHAP values, Asc. order", height=800)
        fig.update_layout(yaxis=dict(tickfont=dict(size=10)))  # Reduce the font size of the labels
        fig.show()
        print(f'\nTop {top_n_feat} features that have high simply summed shapley values over all instances/samples on {set_name} set for OVERprediction instances, Asc. order')
        print(list(labels[:top_n_feat]))

        #simple_summed_shap = shap_values_all.values[overprediction].sum(axis=0)
        simple_summed_shap = shap_values_all.values[bad_feat_op_idx].sum(axis=0)
        sorted_summed_shap = np.sort(simple_summed_shap)[::-1]
        labels = np.array(shap_values_all.feature_names)[np.argsort(simple_summed_shap)[::-1]]
        # fig = px.bar(x=sorted_summed_shap[:top_n_feat], y=labels[:top_n_feat], orientation='h', title=f"{set_name} set's OVERprediction instances, Feature importance based on simply summed SHAP values,  Desc. order", height=800)
        fig = px.bar(x=sorted_summed_shap[:top_n_feat], y=labels[:top_n_feat], orientation='h', title=f"{set_name} set's OVERprediction instances with prediction < base value, Truly BAD Feature importance based on simply summed SHAP values,  Desc. order", height=800)
        fig.update_layout(yaxis=dict(tickfont=dict(size=10)))  # Reduce the font size of the labels
        fig.show()
        print(f'\nTop {top_n_feat} features that have high simply summed shapley values over all instances/samples on {set_name} set for OVERprediction instances, Desc. order')
        print(list(labels[:top_n_feat]))

        # max_shap_values = shap_values_all.values[overprediction].max(axis=0)
        # sorted_max_shap = np.sort(max_shap_values)[::-1]
        # labels = np.array(shap_values_all.feature_names)[np.argsort(max_shap_values)[::-1]]
        # fig = px.bar(x=sorted_max_shap[:top_n_feat], y=labels[:top_n_feat], orientation='h', title=f"{set_name} set's OVERprediction instances, Feature importance based on max SHAP values", height=800)
        # fig.update_layout(yaxis=dict(tickfont=dict(size=10)))  # Reduce the font size of the labels
        # fig.show()
        # print(f'\nTop {top_n_feat} features that have high max shapley values over all instances/samples on {set_name} set for OVERprediction instances')
        # print(list(labels[:top_n_feat]))

        #Examine Individual Feature that lead to most negative shap values Contributions using shap dependence plot
        #self.compute_shapley_PDP_n_Scatter(feature_name,shap_values,stock_id,view_time_ids_start,view_time_ids_end,X,all_stock_y_train_df,all_stock_train_pred_df)

        print(f"\n\nFraction of underprediction instances: {sum(underprediction) / len(prediction)} on {set_name} set")
        print(f"Underprediction RMSPE: {self.rmspe(groundtruth['target'][underprediction], prediction['target'][underprediction])} on {set_name} set")
        print(f"Fraction of overprediction instances: {sum(overprediction) / len(prediction)} on {set_name} set")
        print(f"Overprediction RMSPE: {self.rmspe(groundtruth['target'][overprediction], prediction['target'][overprediction])} on {set_name} set")
        print(f"OVERALL RMSPE: {self.rmspe(groundtruth['target'], prediction['target'])} on {set_name} set")



        # del shap_values_all,groundtruth, prediction, error, underprediction, overprediction, abs_summed_shap, sorted_summed_shap,\
        #     labels, feat_error_corr_dict, sorted_feat_error_corr_dict, min_shap_values, max_shap_values, sorted_min_shap, sorted_max_shap
        del shap_values_all,groundtruth, prediction, error, underprediction, overprediction, abs_summed_shap, sorted_summed_shap,\
            labels, feat_error_corr_dict, sorted_feat_error_corr_dict
        return




    def compute_shapley_heatmap(self,shap_values,stock_id,view_time_ids_start,view_time_ids_end,all_stock_train_pred_df):

        #### ONLY for Explainer
        print(f'\nHEAT MAP \n stock id: {stock_id}, view_time_ids_start:  {view_time_ids_start}, view_time_ids_end: {view_time_ids_end}')
        #print(' NOTE: Heatmap is sorted with f(X) from smallest values to biggest value !! (picture is wrong)')
        # fig,ax = plt.subplots(figsize=(13.5,2))
        # y_asc = np.sort( all_stock_train_pred_df.iloc[ view_time_ids_start : view_time_ids_end ,stock_id].values )
        # ax.plot( range(len(y_asc)), y_asc, color='g')
        # ax.axhline(y_asc.mean(),color='r', linestyle='dashed')
        # ax.set_ylabel('Correct f(x) in Asc. order')
        # ax.set_yticks(np.arange(0,max(y_asc),0.002))
        # fig.show()

        fig,ax = plt.subplots()
        # order = np.argsort(all_stock_train_pred_df.iloc[ view_time_ids_start : view_time_ids_end ,stock_id].values)
        shap.plots.heatmap(shap_values,instance_order=shap_values.sum(1))
        #ax.title(f'\nHEAT MAP \n stock id: {stock_id}, view_time_ids_start:  {view_time_ids_start}, view_time_ids_end: {view_time_ids_end}')
        fig.show()

        del shap_values,all_stock_train_pred_df
        gc.collect()
        return


    def compute_shapley_decision(self,model_base_value,shap_values,feature_names,stock_id,view_time_ids_start,view_time_ids_end,set_name):



        # Create decision plot
        shap.decision_plot(model_base_value, shap_values, X_test, feature_names=feature_names)


        #### ONLY for Explainer
        print(f'\nDECISION PLOT (IGNORE Magnitude, only see relative magnitude) \n stock id: {stock_id}, view_time_ids_start:  {view_time_ids_start}, view_time_ids_end: {view_time_ids_end}')
        fig,ax = plt.subplots()
        shap.plots.decision(model_base_value,shap_values=shap_values.values,features=shap_values.data,feature_names=feature_names, show=True) #matplotlib=True,
        ax.set_title(f'\n DECISION PLOT (IGNORE Magnitude, only see relative magnitude) \n stock id: {stock_id}, view_time_ids_start:  {view_time_ids_start}, view_time_ids_end: {view_time_ids_end}')
        fig.show()

        del shap_values
        gc.collect()
        return


    def compute_shapley_force(self,model_base_value,shap_values,X,feature_names,stock_id,view_time_ids_start,view_time_ids_end):

        # ### ONLY for Explainer
        # print(f'\n FORCE PLOT (IGNORE Magnitude, only see relative magnitude) \n stock id: {stock_id}, view_time_ids_start:  {view_time_ids_start}, view_time_ids_end: {view_time_ids_end}')
        # fig,ax = plt.subplots()
        # shap.plots.force(model_base_value,shap_values=shap_values[0],features=X[0],feature_names=feature_names, show=True) #matplotlib=True,
        # ax.title(f'\n FORCE PLOT (IGNORE Magnitude, only see relative magnitude) \n stock id: {stock_id}, view_time_ids_start:  {view_time_ids_start}, view_time_ids_end: {view_time_ids_end}')
        # fig.show()

        # example_index = 0  # You can change this index to any other example
        # example = X[example_index]
        # # Explain the prediction of the example
        # shap.force_plot(explainer.expected_value, shap_values[example_index], example, feature_names=data.feature_names)

        del shap_values
        gc.collect()
        return



    # def compute_train_avg_target_rvol(self, unique_stock_ids, y_train):
    #     # unique_train_time_ids = self.time_id_order[:self.train_time_id_ind]
    #     unique_train_time_ids = self.time_id_order #[:self.train_time_id_ind]
    #     train_target_df = pd.DataFrame(index=unique_train_time_ids, columns=unique_stock_ids)
    #     for s in unique_stock_ids:
    #         st_index = self.train_stock_id == s
    #         t_index = self.train_time_id[st_index]
    #         train_target_df.loc[t_index, s] = y_train[st_index].values
    #     train_avg_target_rvol = train_target_df.ffill().bfill().mean(axis=1)
    #     return train_avg_target_rvol

    # def compute_test_avg_target_rvol(self, unique_stock_ids, y_test):
    #     #unique_test_time_ids = self.time_id_order[self.train_time_id_ind:]
    #     unique_test_time_ids = self.test_time_id
    #     test_target_df = pd.DataFrame(index=unique_test_time_ids, columns=unique_stock_ids)
    #     for s in unique_stock_ids:
    #         st_index = self.test_stock_id == s
    #         t_index = self.test_time_id[st_index]
    #         test_target_df.loc[t_index, s] = y_test[st_index].values
    #     test_avg_target_rvol = test_target_df.ffill().bfill().mean(axis=1)
    #     return test_avg_target_rvol


    def fraction_above_average(self,signal1, avg):
        # Count the fraction of times when signal1 is above signal2
        fraction_above_avg = (signal1 > avg).mean()
        return fraction_above_avg


    def compute_all_stock_v1tr_df(self, unique_stock_ids, v1tr):
        #unique_train_time_ids = self.time_id_order[:self.train_time_id_ind]
        unique_train_time_ids = self.time_id_order #[:self.train_time_id_ind]
        all_stock_v1tr_df = pd.DataFrame(index=unique_train_time_ids, columns=unique_stock_ids)
        for s in unique_stock_ids:
            st_index = self.train_stock_id == s
            t_index = self.train_time_id[st_index]
            all_stock_v1tr_df.loc[t_index, s] = v1tr[st_index].values
        all_stock_v1tr_df = all_stock_v1tr_df.ffill().bfill()
        return all_stock_v1tr_df

    def compute_all_stock_train_pred_df(self, unique_stock_ids, train_pred):
        #unique_train_time_ids = self.time_id_order[:self.train_time_id_ind]
        unique_train_time_ids = self.time_id_order #[:self.train_time_id_ind]
        all_stock_train_pred_df = pd.DataFrame(index=unique_train_time_ids, columns=unique_stock_ids)
        for s in unique_stock_ids:
            st_index = self.train_stock_id == s
            t_index = self.train_time_id[st_index]
            all_stock_train_pred_df.loc[t_index, s] = train_pred[st_index].values
        all_stock_train_pred_df = all_stock_train_pred_df.ffill().bfill()
        return all_stock_train_pred_df

    def compute_all_stock_test_pred_df(self, unique_stock_ids, test_pred):
        # unique_test_time_ids = self.time_id_order[self.train_time_id_ind:]
        unique_test_time_ids = self.test_time_id
        all_stock_test_pred_df = pd.DataFrame(index=unique_test_time_ids, columns=unique_stock_ids)
        for s in unique_stock_ids:
            st_index = self.test_stock_id == s
            t_index = self.test_time_id[st_index]
            all_stock_test_pred_df.loc[t_index, s] = test_pred[st_index].values
        all_stock_test_pred_df = all_stock_test_pred_df.ffill().bfill()
        return all_stock_test_pred_df


    def compute_all_stock_y_train_df(self, unique_stock_ids, y_train):
        #unique_train_time_ids = self.time_id_order[:self.train_time_id_ind]
        unique_train_time_ids = self.time_id_order #[:self.train_time_id_ind]
        all_stock_y_train_df = pd.DataFrame(index=unique_train_time_ids, columns=unique_stock_ids)
        for s in unique_stock_ids:
            st_index = self.train_stock_id == s
            t_index = self.train_time_id[st_index]
            all_stock_y_train_df.loc[t_index, s] = y_train[st_index].values
        all_stock_y_train_df = all_stock_y_train_df.ffill().bfill()
        return all_stock_y_train_df

    def compute_all_stock_y_test_df(self, unique_stock_ids, y_test):
        #unique_test_time_ids = self.time_id_order[self.train_time_id_ind:]
        unique_test_time_ids = self.test_time_id
        all_stock_y_test_df = pd.DataFrame(index=unique_test_time_ids, columns=unique_stock_ids)
        for s in unique_stock_ids:
            st_index = self.test_stock_id == s
            t_index = self.test_time_id[st_index]
            all_stock_y_test_df.loc[t_index, s] = y_test[st_index].values
        all_stock_y_test_df = all_stock_y_test_df.ffill().bfill()
        return all_stock_y_test_df



    ######## Identify stocks belonging to clusters based on clusterings in dataset
    ######## find stock ids of clusters having same feature values
    ######## This is reverse-engineering cluster labels of already clustered stocks
    def calculate_cluster_fraction(self, column, n_clusters, stock_list):
        """ This function computes the fraction of stock ids in stock_list inside a cluster in the clustering feature.
        The fraction is between 0 - 1. 1 indicates all the stock ids in stock_list are in a particular cluster.
        """

        # self.train_stock_id = df[df['time_id'].isin(train_time_ids)]['stock_id']
        # self.train_time_id = df[df['time_id'].isin(train_time_ids)]['time_id']

        # unique_stock_ids = self.train_stock_id.unique()
        # time_id_order = df2.loc[:3829,'time_id'].values
        # train_time_id_ind = int(len(time_id_order)*0.7)

        # train_time_ids = time_id_order[:train_time_id_ind]
        # train_stock_id = df2[df2['time_id'].isin(train_time_ids)]['stock_id']
        # train_time_id = df2[df2['time_id'].isin(train_time_ids)]['time_id']

        unique_stock_ids = self.train_stock_id.unique()
        #train_time_ids = self.time_id_order[:self.train_time_id_ind]
        train_time_ids = self.time_id_order #[:self.train_time_id_ind]

        train_col_df = self.df[self.df['time_id'].isin(train_time_ids)][column]

        ## reshape the dataframe
        #unique_train_time_ids = self.time_id_order[:self.train_time_id_ind]
        unique_train_time_ids = self.time_id_order #[:self.train_time_id_ind]
        all_stock_column_df = pd.DataFrame(index=unique_train_time_ids, columns=unique_stock_ids)
        for s in unique_stock_ids:
            st_index = self.train_stock_id == s
            t_index = self.train_time_id[st_index]
            all_stock_column_df.loc[t_index, s] = train_col_df[st_index].values
        all_stock_column_df = all_stock_column_df.ffill().bfill()

        features = all_stock_column_df.T.to_numpy()

        ## kmeans
        kmeans = KMeans(n_clusters=n_clusters,n_init=10)
        kmeans.fit(features)
        cluster_labels = kmeans.labels_
        cluster_labels

        clusters_dict = {}
        unique_labels = np.unique(cluster_labels)
        for label in unique_labels:
            indices = np.where(cluster_labels == label)[0]
            stocks_in_cluster = unique_stock_ids[indices]
            clusters_dict[label] = stocks_in_cluster.tolist()

        for c in clusters_dict.keys():
            cnt=0
            for s in stock_list:
                if s in clusters_dict[c]:
                    cnt+=1
            print(f'cluster: {c}, # stock ids in cluster: {cnt}, clustering fraction: {cnt/len(clusters_dict[c])}')

        return



    def check_stock_list_in_all_clustering_features(self, stock_list):

        clustering_features_list = [    "log_target_vol_corr_32_clusters_stnd",
                                        "log_target_vol_sum_stats_16_clusters_stnd",
                                        "sum_stats_4_clusters_labels",
                                        "sum_stats_10_clusters_labels",
                                        "sum_stats_16_clusters_labels",
                                        "sum_stats_30_clusters_labels",
                                        "pear_corr_32_clusters_labels",
                                        "pear_corr_4_clusters_labels",
                                        "pear_corr_49_clusters_labels",
                                        "pear_corr_90_clusters_labels",]

        print('stock_list: ' , stock_list)
        for feature in clustering_features_list:
            n_clusters = int(re.findall(r'\d+', feature)[0])
            print('Feature: ', feature)
            print('Cluster Fractions: ')
            print(self.calculate_cluster_fraction( feature, n_clusters, stock_list))

        return


    def compute_acf_pacf(self,unique_stock_ids,all_stock_train_pred_df,all_stock_y_train_df):
        ##### Autocorrelation and Partial Autocorrelation Plot EVERY individual stock
        plt.close('all')
        for s in unique_stock_ids[0:1]:#[0:40]:
            fig,ax = plt.subplots(2,1,figsize=(30,6))
            stock_residual = all_stock_train_pred_df[s]-all_stock_y_train_df[s]
            plot_acf(stock_residual, lags=200,ax=ax[0])
            plot_pacf(stock_residual, lags=200,ax=ax[1])
            ax[0].set_title(f'Autocorrelation of stock {s} Residuals on train set')
            ax[1].set_title(f'Partial Autocorrelation of stock {s} Residuals on train set')
            ax[0].set_xticks(range(0,200,5))
            ax[1].set_xticks(range(0,200,5))
            ax[0].set_yticks(np.arange(-1, 1, 0.1))
            ax[1].set_yticks(np.arange(-1, 1, 0.1))
            ax[1].set_xlabel('lags')
            ax[0].set_ylabel('ACF')
            ax[1].set_ylabel('PACF')
            ax[0].grid(True)
            ax[1].grid(True)
            fig.show()
        return



    def compute_IFFT(self,unique_stock_ids,all_stock_train_pred_df,all_stock_y_train_df):

        ##### FAST FOURIER TRANSFORM plot of EVERY individual stock
        ##### IFFT plot of reconstructed time series ######
        plt.close('all')
        for s in unique_stock_ids[100:]:#[40:112]:
            stock_residual = all_stock_train_pred_df[s]-all_stock_y_train_df[s]
            x = stock_residual.values
            limit = 0.00001

            n=len(x)
            fhat = np.fft.fft(x,n)
            PSD = fhat*np.conj(fhat) / n
            freq = (1/n)*np.arange(n)
            start=1 #ignore dc component
            L = np.arange(start,np.floor(n/2),dtype='int')
            # fig,ax = plt.subplots(figsize=(30,6))
            # #ax.plot(freq[L],np.array([15]*len(freq[L]))) # line at 15
            # ax.axhline(limit,  color='k', linestyle='-')
            # ax.plot(freq[L],PSD[L])
            # ax.set_xlabel('freq')
            # ax.set_ylabel('mag')
            # ax.set_title(f'mag plot of stock: {s} residual')
            # fig.show()

            indices = PSD > limit
            num_freqs = len(np.where(indices>0)[0])
            print('# of frequencies in residual = ',num_freqs)

            fhat = fhat*indices
            fig,ax = plt.subplots(2,1,figsize=(30,6))
            ffilt = np.fft.ifft(fhat)
            ax[0].plot(np.arange(0,len(x)),ffilt.real,label='top '+str(num_freqs)+' frequencies in residual (train set)',c='g',alpha=1)
            ax[0].plot(np.arange(0,len(x)),x,label='original residual',c='r',alpha=0.2)
            ax[0].legend()
            ax[0].grid()
            ax[0].set_xlabel('time id')
            ax[0].set_ylabel('residual')
            ax[0].set_title(f'IFFT of stock: {s} residual')


            x1 = all_stock_y_train_df[s].values
            limit1 = 0.00001
            n1=len(x1)
            fhat1 = np.fft.fft(x1,n1)
            PSD1 = fhat1*np.conj(fhat1) / n1
            freq1 = (1/n1)*np.arange(n1)
            start1=1 #ignore dc component
            L1 = np.arange(start1,np.floor(n1/2),dtype='int')
            fig1,ax1 = plt.subplots(figsize=(30,6))
            #ax.plot(freq[L],np.array([15]*len(freq[L]))) # line at 15
            ax1.axhline(limit1,  color='k', linestyle='-')
            ax1.plot(freq1[L],PSD1[L])
            ax1.set_xlabel('freq')
            ax1.set_ylabel('mag')
            ax1.set_title(f'mag plot of stock: {s} rvol.')
            fig1.show()

            indices1 = PSD1 > limit1
            num_freqs1 = len(np.where(indices1>0)[0])
            print('# of frequencies in rvol. = ',num_freqs1)
            fhat1 = fhat1*indices1
            ffilt1 = np.fft.ifft(fhat1)
            ax[1].plot(np.arange(0,len(x1)),ffilt1.real,label='top '+str(num_freqs1)+' frequencies in true rvol. (train set)',c='g',alpha=1)
            ax[1].plot(np.arange(0,len(x1)),x1,label='original true rvol.',c='r',alpha=0.2)
            ax[1].legend()
            ax[1].grid()
            ax[1].set_xlabel('time id')
            ax[1].set_ylabel('rvol.')
            ax[1].set_title(f'IFFT of stock: {s} rvol.')
            fig.show()
        return




    def calculate_total_gap(self,data):
        # 1. Order the ground truth in ascending order
        sorted_data = np.sort(data)

        # 2. Take up to the 75th percentile (remove upper outliers)
        percentile_75 = np.percentile(sorted_data, 75)
        filtered_data = sorted_data[sorted_data <= percentile_75]

        # 3. Take first differences
        first_differences = np.diff(filtered_data)

        # 4. Order the first differences in descending order
        sorted_differences = np.sort(first_differences)[::-1]  # Sort in descending order

        # 5. Take the sum of the first differences up to the 50th percentile
        length = len(sorted_differences)
        cutoff_index = int(0.5 * length)  # 50th percentile index
        total_gap = np.sum(sorted_differences[:cutoff_index])

        return total_gap




    def overall_stock_id_analysis(self,unique_stock_ids,all_stock_pred_df,all_stock_y_GroundTruth_df,residuals,train_flag):

        if train_flag:
            set_name = 'train'
        else:
            set_name = 'test'

        all_stock_pred_pivot = all_stock_pred_df.pivot(index='time_id', columns='stock_id', values='target')
        if train_flag:
            all_stock_pred_pivot = all_stock_pred_pivot.reindex(self.time_id_order)

        all_stock_y_GroundTruth_pivot = all_stock_y_GroundTruth_df.pivot(index='time_id', columns='stock_id', values='target')
        if train_flag:
            all_stock_y_GroundTruth_pivot = all_stock_y_GroundTruth_pivot.reindex(self.time_id_order)


        rmspe_per_stock = []
        total_gap_per_stock = []
        for s in unique_stock_ids:
            rmspe_per_stock.append( np.nanmean( ((all_stock_pred_pivot[s]-all_stock_y_GroundTruth_pivot[s])/all_stock_y_GroundTruth_pivot[s])**2 )**0.5  )
            total_gap_per_stock.append( self.calculate_total_gap(all_stock_y_GroundTruth_pivot[s].values) )


        ###### Bar plot of RMSPE for all stocks in the training set
        all_stock_rmspe = pd.Series(rmspe_per_stock,index=unique_stock_ids)
        smallest_10_rmspe_stocks = all_stock_rmspe.sort_values(ascending=True).index.values[:10]
        largest_10_rmspe_stocks = all_stock_rmspe.sort_values(ascending=True).index[::-1].values[:10]
        fig, ax = plt.subplots(figsize=(40,10))
        ax.text(0,0.30,f'10 largest RMSPE stocks: {largest_10_rmspe_stocks}')
        ax.text(0,0.32,f'10 smallest RMSPE stocks: {smallest_10_rmspe_stocks}')
        ax.bar(unique_stock_ids, rmspe_per_stock)
        ax.set_xticks(unique_stock_ids)
        ax.tick_params(axis='x', rotation=45)
        ax.set_yticks(np.arange(0, max(rmspe_per_stock) + 0.01, 0.04))
        ax.grid()
        ax.set_title(f'RMSPE of Real. Vol. for each stock on {set_name} set')
        ax.set_xlabel('Stock ID')
        ax.set_ylabel('RMSPE')
        plt.show()
        plt.close()
        ## check if the largest and smallest fall into a cluster of a clustering feature
        print('\n 10_largest_rmspe_stocks in clustering feature')
        #self.check_stock_list_in_all_clustering_features(stock_list = largest_10_rmspe_stocks)
        print('\n 10_smallest_rmspe_stocks in clustering feature')
        #self.check_stock_list_in_all_clustering_features(stock_list = smallest_10_rmspe_stocks)
        print(f'30_largest_rmspe_stocks in {set_name} set: ',all_stock_rmspe.sort_values(ascending=True).index[::-1].values[:30])


        ####### plot total_gap in descending order for all stocks  ######
        all_stock_total_gap = pd.Series(total_gap_per_stock,index=unique_stock_ids)
        smallest_10_total_gap_stocks = all_stock_total_gap.sort_values(ascending=True).index.values[:10]
        largest_10_total_gap_stocks = all_stock_total_gap.sort_values(ascending=True).index[::-1].values[:10]
        fig, ax = plt.subplots(figsize=(40,10))
        ax.text(4, max(total_gap_per_stock),f'10 largest total gap stocks: {largest_10_total_gap_stocks}')
        ax.text(4, max(total_gap_per_stock)-0.003,f'10 smallest total gap stocks: {smallest_10_total_gap_stocks}')
        sorted_indices = np.argsort(total_gap_per_stock)[::-1]
        sorted_unique_stock_ids = unique_stock_ids[sorted_indices]
        sorted_total_gap_per_stock = np.array(total_gap_per_stock)[sorted_indices]
        ax.bar(range(len(sorted_unique_stock_ids)), sorted_total_gap_per_stock,tick_label=sorted_unique_stock_ids)
        ax.tick_params(axis='x', rotation=45)
        ax.set_yticks(np.arange(0, max(sorted_total_gap_per_stock) + 0.001, 0.001))
        ax.grid()
        ax.set_title(f'Total Gap of Real. Vol. for each stock on GroundTruth {set_name} set')
        ax.set_xlabel('Stock ID')
        ax.set_ylabel('Total Gap')
        plt.show()
        plt.close()
        print(f'30_largest_total_gap_stocks in {set_name} set: ',all_stock_total_gap.sort_values(ascending=True).index[::-1].values[:30])


        groundtruth = all_stock_y_GroundTruth_df['target'].values
        prediction = all_stock_pred_df['target'].values
        residuals = residuals['target'].values # copy is made, its not modified inplace

        ####### scatter plot of True Real. Vol. vs. Pred Real. Vol.
        fig, ax = plt.subplots(figsize=(30,10))
        ax.scatter(groundtruth, prediction, c='b', alpha=0.1)
        ax.plot(groundtruth, groundtruth, c='r',linestyle='solid' )
        ax.set_title(f'Scatter Plot of True vs Predicted Values on {set_name} set')
        ax.set_xlabel(f'True {set_name} rvol. Values')
        ax.set_ylabel(f'Predicted {set_name} rvol. Values')
        plt.show()
        plt.close()

        ## y_train and train_pred Distributions Histogram:
        fig, ax = plt.subplots(figsize=(30,10))
        plt.hist( groundtruth,bins=1000, color='green', alpha=0.9, histtype='bar', rwidth=0.8, label='GroundTruth')
        plt.hist( prediction,bins=1000, color='red', alpha=0.3, ec='r', label='Prediction')
        ax.set_title(f'Distribution of GroundTruth (skew: {stats.skew(groundtruth)} , kurt:{stats.kurtosis(groundtruth)}) and Prediction (skew: {stats.skew(prediction)} , kurt:{stats.kurtosis(prediction)}) on {set_name} set')
        ax.set_xlabel(f' GroundTruth and Prediction on {set_name} set')
        ax.set_ylabel('frequency')
        plt.legend(loc="upper left")
        plt.show()
        plt.close()

        ####### scatter plot of True rvol. Values Plot Vs. Train Residuals
        fig, ax = plt.subplots(figsize=(30,10))
        ax.scatter(groundtruth, residuals, c='c',alpha=0.1 )
        ax.axhline(y=0, color='g', linestyle='-')
        #ax.axhline(y=np.mean(groundtruth), color='r', linestyle='-')
        ax.set_title(f' True R.V. Vs. Residuals Values Plot on {set_name} set')
        ax.set_xlabel(f'True {set_name} Values')
        ax.set_ylabel(f'{set_name} residuals')
        plt.show()
        plt.close()


        ####### scatter plot of Fitted rvol. Values Vs. train residuals Plot:
        fig, ax = plt.subplots(figsize=(30,10))
        ax.scatter(prediction, residuals, c='m',alpha=0.1 )
        ax.axhline(y=0, color='g', linestyle='-')
        #ax.axhline(y=np.mean(groundtruth), color='r', linestyle='-')
        ax.set_title(f' fitted R.V. Vs. Residuals Values Plot on {set_name} set')
        ax.set_xlabel(f'fitted {set_name} Values')
        ax.set_ylabel(f'{set_name} residuals')
        plt.show()
        plt.close()

        ## Normal Q-Q Plot:
        fig, ax = plt.subplots(figsize=(30,10))
        sm.qqplot(residuals, line='q', ax=ax)
        ax.set_title(f'QQ Plot of Residuals on {set_name} set')
        ax.set_xlabel('Theoretical Quantiles')
        ax.set_ylabel('Sample Quantiles')
        plt.show()
        plt.close()

        ## Residuals Distribution Histogram:
        fig, ax = plt.subplots(figsize=(30,10))
        plt.hist( residuals,bins=1000)
        ax.set_title(f'Distribution of Residuals on {set_name} set')
        ax.set_xlabel(f'{set_name} Residuals')
        ax.set_ylabel('frequency')
        plt.show()
        plt.close()

        del all_stock_pred_df,all_stock_y_GroundTruth_df,residuals,all_stock_y_GroundTruth_pivot,all_stock_pred_pivot,groundtruth,prediction
        gc.collect()
        return pd.DataFrame({f'{set_name}_rmspe_per_stock':rmspe_per_stock}, index=unique_stock_ids)




    def individual_stock_id_analysis(self,picked_stock_id,unique_stock_ids,all_stock_y_GroundTruth_df,all_stock_pred_df,residuals,avg_target_rvol,train_flag):

        if train_flag:
            set_name = 'train'
        else:
            set_name = 'test'

        all_stock_pred_pivot = all_stock_pred_df.pivot(index='time_id', columns='stock_id', values='target')
        if train_flag:
            all_stock_pred_pivot = all_stock_pred_pivot.reindex(self.time_id_order)

        all_stock_y_GroundTruth_pivot = all_stock_y_GroundTruth_df.pivot(index='time_id', columns='stock_id', values='target')
        if train_flag:
            all_stock_y_GroundTruth_pivot = all_stock_y_GroundTruth_pivot.reindex(self.time_id_order)

        groundtruth = all_stock_y_GroundTruth_pivot[picked_stock_id].values
        prediction = all_stock_pred_pivot[picked_stock_id].values
        picked_st_residuals = groundtruth - prediction

        ## 1. scatter plot
        plt.figure(figsize=(10,10))
        plt.scatter(all_stock_y_GroundTruth_pivot[picked_stock_id],all_stock_pred_pivot[picked_stock_id], c='blue',label=picked_stock_id, alpha=0.4)
        plt.plot(all_stock_y_GroundTruth_pivot[picked_stock_id],all_stock_y_GroundTruth_pivot[picked_stock_id],linestyle='solid', c='red',label=picked_stock_id, alpha=1 )
        plt.grid()
        plt.xlabel(f'GroundTruth')
        plt.ylabel(f'Prediction')
        plt.legend()
        plt.title(f"stock {picked_stock_id}'s scatter plot of y_{set_name} vs. {set_name}_pred on {set_name} set")
        plt.show()
        plt.close()

        ## 2. Line plot of true vs average real. vol.
        fraction_above_avg = self.fraction_above_average(all_stock_y_GroundTruth_pivot[picked_stock_id], avg_target_rvol)
        plt.figure(figsize=(30,5))
        plt.text(1,all_stock_y_GroundTruth_pivot[picked_stock_id].max(),f"fraction of times this stock's values are above all stocks' avg_target_rvol = {fraction_above_avg}")
        plt.plot(range(len(all_stock_y_GroundTruth_pivot[picked_stock_id])),all_stock_y_GroundTruth_pivot[picked_stock_id],linestyle='solid', c='green',label=f'True stock id: {picked_stock_id}', alpha=0.4 )
        #plt.plot(range(len(all_stock_y_GroundTruth_pivot[picked_stock_id])),[avg_target_rvol]*len(all_stock_y_GroundTruth_pivot[picked_stock_id]),linestyle='solid', c='blue',label=f'{set_name}_avg_target_rvol', alpha=0.4 )
        plt.plot(range(len(avg_target_rvol)),avg_target_rvol,linestyle='solid', c='blue',label=f'{set_name}_avg_target_rvol', alpha=0.4 )
        plt.grid()
        plt.xlabel('index')
        plt.ylabel('train rvol.')
        plt.legend()
        plt.title(f"stock {picked_stock_id}'s line plot of True y_{set_name} vs. {set_name}_avg_target_rvol on {set_name} set")
        plt.show()
        plt.close()

        # ## 3. Line plot of pred vs true real. vol.
        # plt.figure(figsize=(30,5))
        # plt.plot(range(len(all_stock_y_GroundTruth_pivot[picked_stock_id])),all_stock_y_GroundTruth_pivot[picked_stock_id],linestyle='solid', c='green',label='True stock id: '+str(picked_stock_id), alpha=0.7 )
        # plt.plot(range(len(all_stock_pred_pivot[picked_stock_id])),all_stock_pred_pivot[picked_stock_id],linestyle='solid', c='red',label='Pred stock id: '+str(picked_stock_id), alpha=0.4 )
        # plt.grid()
        # plt.xlabel('index')
        # plt.ylabel(f'{set_name} rvol.')
        # plt.legend()
        # plt.title(f"stock {picked_stock_id}'s line plot of True y_{set_name} vs {set_name}_pred on {set_name} set")
        # plt.show()
        # plt.close()

        ## y_train and train_pred Distributions Histogram:
        fig, ax = plt.subplots(2, 1, figsize=(30, 10))
        max_val = max(groundtruth.max(), prediction.max())
        ax[0].hist(groundtruth, bins=1000, color='green', alpha=1, histtype='bar', rwidth=0.8, label='GroundTruth')
        ax[0].set_xlim(-0.001, max_val)
        ax[0].set_title(f"stock {picked_stock_id}'s Distribution of GroundTruth (skew: {stats.skew(groundtruth)} , kurt:{stats.kurtosis(groundtruth)}) and Prediction (skew: {stats.skew(prediction)} , kurt:{stats.kurtosis(prediction)}) on {set_name} set")
        ax[0].set_xlabel(f' GroundTruth on {set_name} set')
        ax[0].set_ylabel('frequency')
        ax[1].hist(prediction, bins=1000, color='red', alpha=1, ec='r', label='Prediction')
        ax[1].set_xlim(-0.001, max_val)
        ax[1].set_xlabel(f' Prediction on {set_name} set')
        ax[1].set_ylabel('frequency')
        plt.legend(loc="upper left")
        plt.show()
        plt.close()

        ####### scatter plot of True rvol. Values Plot Vs. Train Residuals
        fig, ax = plt.subplots(2,1,figsize=(30,10))
        max_val = max(groundtruth.max(), prediction.max())
        ax[0].scatter(groundtruth, picked_st_residuals, c='c',alpha=0.1 )
        ax[0].set_xlim(-0.001, max_val)
        ax[0].axhline(y=0, color='g', linestyle='-')
        #ax.axhline(y=np.mean(groundtruth), color='r', linestyle='-')
        ax[0].set_title(f"stock {picked_stock_id}'s True R.V. Vs. Residuals Values Plot on {set_name} set")
        ax[0].set_xlabel(f'True {set_name} Values')
        ax[0].set_ylabel(f'{set_name} picked_st_residuals')
        ax[1].scatter(prediction, picked_st_residuals, c='m',alpha=0.1 )
        ax[1].set_xlim(-0.001, max_val)
        ax[1].axhline(y=0, color='g', linestyle='-')
        ax[1].set_title(f"stock {picked_stock_id}'s fitted R.V. Vs. Residuals Values Plot on {set_name} set")
        ax[1].set_xlabel(f'fitted {set_name} Values')
        ax[1].set_ylabel(f'{set_name} picked_st_residuals')
        plt.legend(loc="upper left")
        plt.show()
        plt.close()

        ## Normal Q-Q Plot:
        fig, ax = plt.subplots(figsize=(30,10))
        sm.qqplot(picked_st_residuals, line='q', ax=ax)
        ax.set_title(f"stock {picked_stock_id}'s QQ Plot of Residuals on {set_name} set")
        ax.set_xlabel('Theoretical Quantiles')
        ax.set_ylabel('Sample Quantiles')
        plt.show()
        plt.close()

        ## Residuals Distribution Histogram:
        fig, ax = plt.subplots(figsize=(30,10))
        plt.hist( picked_st_residuals,bins=1000)
        ax.set_title(f"stock {picked_stock_id}'s Distribution of Residuals on {set_name} set")
        ax.set_xlabel(f'{set_name} Residuals')
        ax.set_ylabel('frequency')
        plt.show()
        plt.close()

        ## box plot of residuals, groundtruth and prediction
        fig, ax = plt.subplots(figsize=(5,5))
        ax.boxplot([picked_st_residuals,groundtruth,prediction],labels=['residuals','groundtruth','prediction'])
        ax.set_title(f" stock {picked_stock_id}'s Box Plot of Residuals, GroundTruth and Prediction on {set_name} set")
        ax.set_ylabel('value')
        #ax.set_yticks(np.arange(-0.01, 0.025, 0.001))
        ax.grid()
        plt.show()
        plt.close()

        # ###### Autocorrelation Plot
        # fig, ax = plt.subplots(figsize=(10,3))
        # plot_acf(residuals, lags=20, ax=ax)  # You can adjust the number of lags as needed
        # ax.set_xlabel('Lag')
        # ax.set_ylabel('Autocorrelation')
        # ax.set_yticks(np.arange(-1, 1, 0.1))
        # ax.grid()
        # ax.set_title(f'Autocorrelation of {set_name} Residuals')
        # fig.show()

        # ###### Partial Autocorrelation Plot
        # fig, ax = plt.subplots(figsize=(10,3))
        # plot_pacf(residuals, lags=20, ax=ax)  # You can adjust the number of lags as needed
        # ax.set_xlabel('Lag')
        # ax.set_ylabel('Partial Autocorrelation')
        # ax.set_yticks(np.arange(-1, 1, 0.1))
        # ax.grid()
        # plt.title(f'Partial Autocorrelation of {set_name} Residuals')
        # plt.show()

        ##### Autocorrelation and Partial Autocorrelation Plot EVERY individual stock
        #self.compute_acf_pacf(unique_stock_ids,all_stock_pred_df,all_stock_y_GroundTruth_df)


        # #### FAST FOURIER TRANSFORM plot of EVERY individual stock
        # #### IFFT plot of reconstructed time series ######
        # self.compute_IFFT(unique_stock_ids,all_stock_train_pred_df,all_stock_y_train_df)

        del picked_stock_id,unique_stock_ids,all_stock_y_GroundTruth_df,all_stock_pred_df,residuals
        gc.collect()
        return groundtruth



    def overall_time_id_analysis(self, all_stock_y_GroundTruth_df,all_stock_pred_df,avg_target_rvol,train_flag):

        # Precompute variables
        set_name = 'train' if train_flag else 'test'

        # Pivot with 'stock_id' as index and 'time_id' as columns
        if train_flag:
            all_stock_pred_pivot = all_stock_pred_df.pivot(index='stock_id', columns='time_id', values='target').reindex(columns=self.time_id_order)
            all_stock_y_GroundTruth_pivot = all_stock_y_GroundTruth_df.pivot(index='stock_id', columns='time_id', values='target').reindex(columns=self.time_id_order)
        else:
            all_stock_pred_pivot = all_stock_pred_df.pivot(index='stock_id', columns='time_id', values='target')
            all_stock_y_GroundTruth_pivot = all_stock_y_GroundTruth_df.pivot(index='stock_id', columns='time_id', values='target')

        unique_time_ids = all_stock_y_GroundTruth_pivot.columns.values
        # Vectorized RMSPE calculation along the 'stock_id' axis
        rmspe_per_time_id = np.sqrt(
            np.mean(
                ((all_stock_pred_pivot - all_stock_y_GroundTruth_pivot) / all_stock_y_GroundTruth_pivot) ** 2, axis=0
            )
        )

        # print('all_stock_pred_pivot.index',all_stock_pred_pivot)
        # display_n_time_ids = 10
        # sorted_rmspe_per_time_id = rmspe_per_time_id.sort_values(ascending=False).head(display_n_time_ids)
        # for t in sorted_rmspe_per_time_id.index:
        #     plt.figure(figsize=(30,5))
        #     plt.plot(all_stock_pred_pivot.index,all_stock_pred_pivot[t],linestyle='solid', c='red',label='pred', alpha=0.4 )
        #     plt.plot(all_stock_pred_pivot.index,all_stock_y_GroundTruth_pivot[t],linestyle='solid', c='green',label='true', alpha=0.4 )
        #     plt.title(f'time index {t}, rmspe {np.sqrt(np.mean(((all_stock_pred_pivot[t]-all_stock_y_GroundTruth_pivot[t])/all_stock_y_GroundTruth_pivot[t])**2))}')
        #     plt.xticks(all_stock_pred_pivot.index,rotation=90)
        #     plt.grid()
        #     plt.show()
        #     plt.close()

        # plot the RMSPE for all time ids
        display_n_time_ids = 200
        sorted_rmspe_per_time_id = rmspe_per_time_id.sort_values(ascending=False).head(display_n_time_ids)
        fig, ax = plt.subplots(figsize=(40, 10))
        ax.bar(range(len(sorted_rmspe_per_time_id)), sorted_rmspe_per_time_id, tick_label=sorted_rmspe_per_time_id.index)
        ax.set_xticks(range(len(sorted_rmspe_per_time_id)))
        ax.tick_params(axis='x', rotation=90)
        ax.set_yticks(np.arange(0, max(sorted_rmspe_per_time_id) + 0.01, 0.04))
        ax.grid()
        ax.set_title(f'RMSPE of Real. Vol. for each time id on {set_name} set')
        ax.set_xlabel('Time ID')
        plt.show()
        plt.close()

        all_time_id_rmspe = rmspe_per_time_id

        fig, ax = plt.subplots(figsize=(40, 10))
        largest_20_rmspe_time_ids = all_time_id_rmspe.nlargest(20).index.values
        # Precompute the maximum RMSPE value for the largest 20 time IDs
        max_large_val = all_time_id_rmspe.loc[largest_20_rmspe_time_ids].max()
        # Create the second bar plot for the largest 20 RMSPE time IDs
        ax.text(5, max_large_val + 0.001, f'20 largest RMSPE time ids: {largest_20_rmspe_time_ids}', fontsize=12)
        # Convert time ids to string once and plot the data
        ax.bar(largest_20_rmspe_time_ids.astype(str), all_time_id_rmspe.loc[largest_20_rmspe_time_ids])
        # Set y-ticks efficiently, using the precomputed max value
        ax.set_yticks(np.arange(0, max_large_val + 0.001, 0.08))
        ax.set_ylabel('RMSPE')
        ax.set_title(f'20 largest RMSPE time ids on {set_name} set')
        ax.grid()
        plt.show()
        plt.close()


        all_time_id_rmspe = rmspe_per_time_id
        # Use `nsmallest` and `nlargest` to directly get the top 10 smallest and largest RMSPE values without full sorting
        smallest_10_rmspe_time_ids = all_time_id_rmspe.nsmallest(10).index.values
        # Precompute max value for y-ticks range in a single operation
        max_small_val = all_time_id_rmspe.loc[smallest_10_rmspe_time_ids].max()
        # Create the bar plot for RMSPE with precomputed values
        fig, ax = plt.subplots(figsize=(40, 10))
        # Set text once, and avoid converting indices to strings multiple times
        ax.text(5, max_small_val + 0.001, f'10 smallest RMSPE time ids: {smallest_10_rmspe_time_ids}', fontsize=12)
        # Convert time ids to string once and avoid recalculating max_small_val in yticks
        ax.bar(smallest_10_rmspe_time_ids.astype(str), all_time_id_rmspe.loc[smallest_10_rmspe_time_ids])
        # Set y-ticks with precomputed values (adjusting the range once)
        ax.set_yticks(np.arange(0, max_small_val + 0.001, 0.04))
        ax.set_ylabel('RMSPE')
        ax.set_title(f'10 smallest RMSPE time ids on {set_name} set')
        ax.grid()
        plt.show()
        plt.close()


        ###### visualize the time ids with largest and smallest RMSPE on the average rvol. plot on training set
        plt.figure(figsize=(30,5))
        plt.plot(range(len(avg_target_rvol)),avg_target_rvol,linestyle='solid', c='blue',label=f'{set_name}_avg_target_rvol', alpha=0.4 )
        large_idx = np.where(np.isin(unique_time_ids,largest_20_rmspe_time_ids))[0]
        red_colors = colors = ['black', 'maroon', 'darkred', 'firebrick', 'crimson', 'indianred', \
          'tomato', 'lightcoral', 'salmon', 'hotpink', 'palevioletred', \
          'mediumvioletred', 'orchid', 'fuchsia', 'magenta', 'violet', \
          'plum', 'mediumorchid', 'lavender', 'thistle']
        for i,s in enumerate(large_idx):
            plt.axvline(x=s, ymin=0, ymax=1,color=red_colors[i],linestyle='-',label=str(i))
        small_idx = np.where(np.isin(unique_time_ids,smallest_10_rmspe_time_ids))[0]
        green_colors = ['gold', 'yellow', 'lightyellow', 'khaki', 'greenyellow', \
                'chartreuse', 'lime', 'lawngreen', 'darkgreen', 'forestgreen', \
                'seagreen', 'mediumseagreen', 'springgreen', 'aquamarine', \
                'turquoise', 'lightgreen', 'mediumspringgreen', 'cyan', 'skyblue', 'deepskyblue']
        for j,l in enumerate(small_idx):
            plt.axvline(x=l, ymin=0, ymax=1,color=green_colors[j],linestyle='-',label=str(j))
        plt.grid()
        plt.yticks(np.arange(0, 0.04, 0.01))
        plt.xlabel('sequential time id index')
        plt.ylabel(f'{set_name} rvol.')
        plt.title(f'Average {set_name} rvol. Darker reddish lines for 20 largest RMSPE time ids and greenish lines for 10 smallest RMSPE time ids')
        plt.legend()
        plt.show()

        del all_stock_y_GroundTruth_df,all_stock_pred_df,avg_target_rvol
        gc.collect()
        return all_stock_y_GroundTruth_pivot[largest_20_rmspe_time_ids],  all_stock_pred_pivot[largest_20_rmspe_time_ids]



    def compute_model_bias_variance(self,y_test,y_train,X_train,best_mlxtend_xgb_params):

        ## model bias and variance measurement
        # estimate bias and variance
        #train_time_ids = self.time_id_order[:self.train_time_id_ind]
        train_time_ids = self.time_id_order #[:self.train_time_id_ind]

        #full_train_df = self.df[self.df['time_id'].isin(train_time_ids)]

        #X_train = full_train_df[self.feat_cols_list]
        X_train = X_train[self.feat_cols_list]
        #y_train = full_train_df[self.target_name] #target
        y_train = y_train[self.target_name]
        #train_weight = y_train['target']
        X_test = self.test_df[self.feat_cols_list]
        #y_test = self.test_df[self.target_name] #target


        # Assuming best_mlxtend_xgb_params contains the hyperparameters
        max_depth, eta, subsample, colsample_bytree, gamma, reg_alpha, reg_lambda, min_child_weight, num_rounds = best_mlxtend_xgb_params

        # Create XGBRegressor model
        xgb_model = XGBRegressor(
            max_depth=max_depth,
            learning_rate=eta,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            n_estimators=num_rounds,
            objective='reg:squarederror',
            tree_method = "hist",
            device = "cuda"
        )

        #v1tr = np.exp(X_train['log_wap1_log_price_ret_vol']) # double exponential to nullify log
        v1tr = np.array([1]*len(self.train_df['log_wap1_log_price_ret_vol'])) # remove
        #v1ts = np.exp( self.test_df['log_wap1_log_price_ret_vol']) # double exponential to nullify log
        v1ts = np.array([1]*len( self.test_df['log_wap1_log_price_ret_vol'])) # remove

        w_train = y_train **-2 * v1tr**2
        #w_test = y_test **-2 * v1ts**2

        # Train XGBRegressor model
        xgb_model.fit(X_train.values, (np.log(y_train.values/v1tr.values) + self.target_shift), sample_weight=w_train)

        # Now you can use bias_variance_decomp
        mse, bias, var = bias_variance_decomp(xgb_model, X_train.values, (np.log(y_train.values/v1tr.values) + self.target_shift), X_test.values, (np.log(y_test['target'].values/v1ts.values) + self.target_shift), loss='mse', num_rounds=30, random_seed=1)
        print('\nMSE: %.3f' % mse)
        print('Bias: %.3f' % bias)
        print('Variance: %.3f' % var)

        return



    ############################################################################################################
    """  TRAINING and TESTING SET COMPARISON functions START"""
    ############################################################################################################

    def train_n_test_set_st_RMSPE_comparison(self,train_rmspe_per_stock,test_rmspe_per_stock):
        # Combine train and test RMSPE into a single DataFrame for easier plotting
        rmspe_df = pd.DataFrame({
            'Stock ID': train_rmspe_per_stock.index,
            'Train RMSPE': train_rmspe_per_stock['train_rmspe_per_stock'],
            'Test RMSPE': test_rmspe_per_stock['test_rmspe_per_stock']
        })
        fig = px.line(rmspe_df, x='Stock ID', y=['Train RMSPE', 'Test RMSPE'], \
                      title=f'Train vs Test RMSPE per Stock,\n Corrrelation between train and test RMSPE: {rmspe_df["Train RMSPE"].corr(rmspe_df["Test RMSPE"])}')
        fig.update_layout(xaxis=dict(tickvals=rmspe_df['Stock ID'], tickangle=90), yaxis_title='RMSPE', xaxis_title='Stock ID')
        fig.show()
        return



    def train_n_test_set_picked_st_distribution_comparison(self,train_groundtruth, test_groundtruth , picked_stock_id):

        # perform Kolmogorov-Smirnov test
        ks_stat, ks_pval = stats.ks_2samp(train_groundtruth, test_groundtruth)
        print(f"Kolmogorov-Smirnov test: KS Statistic: {ks_stat}, P-Value: {ks_pval}")
        # Check if the distributions are the same
        alpha = 0.05  # Significance level
        if ks_pval < alpha:
            print("The null hypothesis is rejected. The test set does not come from the same distribution as the train set.")
            ks_test_stats = f"K-S TEST: Train and Test sets are DIFFERENT"
        else:
            print("The null hypothesis is accepted. The test set may come from the same distribution as the train set.")
            ks_test_stats = f"K-S TEST: Train and Test sets are SAME"

        # violin plot train_groundtruth and test_groundtruth on the same plot
        fig = go.Figure()
        fig.add_trace(go.Violin(x=['Train']*len(train_groundtruth), y=train_groundtruth, name='Train', box_visible=True, meanline_visible=True))
        fig.add_trace(go.Violin(x=['Test']*len(test_groundtruth), y=test_groundtruth, name='Test', box_visible=True, meanline_visible=True))
        train_stats = f"Train - Mean: {np.mean(train_groundtruth):.4f}, Median: {np.median(train_groundtruth):.4f}, Std: {np.std(train_groundtruth):.4f}, Skew: {stats.skew(train_groundtruth):.4f}, Kurtosis: {stats.kurtosis(train_groundtruth):.4f}, 75th Percentile: {np.percentile(train_groundtruth, 75):.4f}"
        test_stats = f"Test - Mean: {np.mean(test_groundtruth):.4f}, Median: {np.median(test_groundtruth):.4f}, Std: {np.std(test_groundtruth):.4f}, Skew: {stats.skew(test_groundtruth):.4f}, Kurtosis: {stats.kurtosis(test_groundtruth):.4f}, 75th Percentile: {np.percentile(test_groundtruth, 75):.4f}"
        fig.update_layout(
            title=f"Stock {picked_stock_id}'s Train vs Test GroundTruth Distribution<br>{train_stats}<br>{test_stats}<br>   {ks_test_stats}",
            yaxis=dict(tickmode='linear', tick0=0, dtick=0.002)  # Increase y tick resolution
        )
        fig.show()
        return



    def each_time_id_RMSPE_across_stocks_comparison(self,gt_time_id_df, pred_time_id_df,set_name,y_train_df):

        y_train_pivot = y_train_df.pivot(index='time_id', columns='stock_id', values='target')

        for time_id in gt_time_id_df.columns.values:
            gt_df = pd.DataFrame({'gt':gt_time_id_df[time_id],'pred':pred_time_id_df[time_id]}, index=gt_time_id_df.index)
            plt.figure(figsize=(40, 10))
            plt.title(f'Time ID {time_id} GroundTruth vs Prediction on {set_name} set')
            plt.plot(gt_df.index, gt_df['gt'].values, label='GroundTruth', color='blue')
            plt.plot(gt_df.index, gt_df['pred'].values, label='Prediction', color='red')
            # ## Add box plot of y_train_pivot for each stock
            plt.boxplot([y_train_pivot[stock_id].dropna().values for stock_id in gt_df.index], positions=gt_df.index)
            plt.xticks(ticks=gt_df.index, labels=gt_df.index, rotation=90)
            plt.yticks(np.arange(0, 0.04, 0.002))
            plt.ylim(0, 0.04)
            plt.grid()
            plt.legend()
            plt.xlabel('Stock ID')
            plt.ylabel('Real. Vol.')
            plt.show()

        del gt_time_id_df, pred_time_id_df, y_train_df,y_train_pivot
        return


    def val_set_n_test_set_representation_plot(self,mean_val_set_rmspe_error, test_set_rmspe_error, train_set_rmspe_error):
        ## check if average walk-forward validation set error and test set error are positively correlated.
        val_test_corr = np.corrcoef(mean_val_set_rmspe_error,test_set_rmspe_error)[0,1]
        fig, ax1 = plt.subplots()
        # Original scatter plot
        ax1.scatter(test_set_rmspe_error, mean_val_set_rmspe_error, alpha=0.4, color='blue', label='Mean Validation RMSPE')
        ax1.set_xlabel('Test Set RMSPE Error')
        ax1.set_ylabel('Mean Validation Set RMSPE Error', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        # Create twin axis
        ax2 = ax1.twinx()
        # Add train_set_rmspe_error to the twin axis
        ax2.scatter(test_set_rmspe_error, train_set_rmspe_error, alpha=0.4, color='red', label='Train RMSPE')
        ax2.set_ylabel('Train Set RMSPE Error', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        # Add correlation to the title
        train_test_corr = np.corrcoef(test_set_rmspe_error, train_set_rmspe_error)[0, 1]
        plt.title(f'Test Set RMSPE Error vs. Mean Validation/Train Set RMSPE Error \n : val_test_corr: {val_test_corr:.3f} \n train_test_corr {train_test_corr:.3f}')
        # Add legend
        plt.legend(loc='upper left')
        plt.show()

        return


    ############################################################################################################
    """  TRAINING and TESTING SET COMPARISON functions END"""
    ############################################################################################################



    def evaluate_predictions(self,final_reg,test_pred, y_test,train_pred,y_train,X_train,X_test,v1tr,v1ts,w_train,w_test,best_mlxtend_xgb_params,mean_val_set_rmspe_error,raw_train_pred,raw_test_pred,raw_train_gn,raw_test_gn):

        y_true = y_test
        y_pred = test_pred
        test_residuals = pd.DataFrame(y_true['target'] - y_pred['target'])
        test_residuals[['stock_id','time_id']] = y_true[['stock_id','time_id']]
        train_residuals = pd.DataFrame(y_train['target'] - train_pred['target'])
        train_residuals[['stock_id','time_id']] = y_train[['stock_id','time_id']]
        unique_stock_ids = self.train_stock_id.unique()

        #all_stock_train_pred_df = self.compute_all_stock_train_pred_df(unique_stock_ids, train_pred)
        all_stock_train_pred_df = train_pred
        all_stock_raw_train_pred_df = raw_train_pred
        #all_stock_v1tr_df = self.compute_all_stock_v1tr_df(unique_stock_ids, v1tr)
        all_stock_v1tr_df = v1tr
        #all_stock_y_train_df = self.compute_all_stock_y_train_df(unique_stock_ids, y_train)
        all_stock_y_train_df = y_train
        all_stock_raw_train_gn_df = raw_train_gn
        #train_avg_target_rvol = self.compute_train_avg_target_rvol(unique_stock_ids, y_train)
        train_avg_target_rvol = y_train.groupby('time_id')['target'].mean().reindex(self.time_id_order)
        test_avg_target_rvol = y_test.groupby('time_id')['target'].mean() #.reindex(self.time_id_order)
        # train_avg_target_rvol = y_train.groupby('time_id')['target'].median().reindex(self.time_id_order) ## median
        # test_avg_target_rvol = y_test.groupby('time_id')['target'].median() #.reindex(self.time_id_order) ## median

        #all_stock_test_pred_df = self.compute_all_stock_test_pred_df( unique_stock_ids, test_pred)
        all_stock_test_pred_df = test_pred
        all_stock_raw_test_pred_df = raw_test_pred
        #all_stock_y_test_df = self.compute_all_stock_y_test_df( unique_stock_ids, y_test)
        all_stock_y_test_df = y_test
        all_stock_raw_test_gn_df = raw_test_gn
        all_stock_v1ts_df = v1ts

        train_set_rmspe_error = [0.190656,0.190647,0.195972,0.192530]
        test_set_rmspe_error = [0.230367,0.227526,0.232057,0.266616]

        print('\n####################################### PREDICTION #################################################')

        #v1ts = np.exp(np.exp( self.test_df['log_wap1_log_price_ret_vol'])) # double exponential to nullify log
        #v1ts = np.exp( self.test_df['log_wap1_log_price_ret_vol']) # double exponential to nullify log
        print('corr(y_pred/v1ts, y_true/v1ts)',self.nancorr(       y_pred['target'].values/v1ts['wap1_log_price_ret_vol'] ,        y_true['target'].values/v1ts['wap1_log_price_ret_vol'] ))
        print('log(corr( ))',self.nancorr(np.log(y_pred['target'].values/v1ts['wap1_log_price_ret_vol']), np.log(y_true['target'].values/v1ts['wap1_log_price_ret_vol'])))
        print('corr(y_pred, y_true)',self.nancorr(y_pred['target'].values, y_true['target'].values))
        print('log(corr( ))',self.nancorr(np.log(y_pred['target'].values), np.log(y_true['target'].values)))
        train_score = np.mean( ((train_pred['target'].values-y_train['target'].values)/y_train['target'].values)**2 )**0.5
        print(f'RMSPE train score: ',  train_score)
        train_set_rmspe_error.append(train_score)
        test_score = np.mean( ((y_pred['target'].values-y_true['target'].values)/y_true['target'].values)**2 )**0.5
        print(f'RMSPE test score: ',  test_score  )
        test_set_rmspe_error.append(test_score)

        ############################ SET PARAMETERS HERE ##############################
        ##### individual stock id analysis parameters START #####
        high_rmspe_stocks = ['18', '37', '81', '80', '33', '60', '110', '88', '103', '16', '27', '9', '30', '98', '75', '3', '89', '112', '7', '0', '40', '4', '58', '83', '5', '90', '66', '97', '6', '86']
        start_index = 0
        end_index = 1
        good_pdp_feature_name = ['liqvol1_smean',
        'log_liq2_ret_*_wap_eqi_price1_ret_vol_15_ratio_median_stock',
        'pear_corr_90_clusters_labels',
        'robust_sum_stats_60_clusters_labels',
        'root_trade_count_smean',
        'soft_stock_mean_tvpl2_:20',
        'soft_stock_mean_tvpl2_liqf_volf20',
        'texp_wap1_log_price_ret_vol_from_20',
        'tlog_eps523_trade_price_n_wap_eqi_price0_dev',
        'tlog_target_vol_pcorr_3_clusters',
        'tlog_wap1_lprice_ret_vol_ati_*_wap1_lprice_ret_vol_ast_per_wap1_lprice_ret_vol_:20',
        'tvpl2',
        'tvpl2_smean_vol',
        'tvpl3',
        'tvpl3_rmed2v1lf20',
        'v1proj_25',
        'v1proj_29_15_q3',
        'v1proj_29_q3',
        'v1spprojt15f25',
        'v1spprojt15f25_q1',
        'wap1_log_price_ret_vol_from_25',
        'wap1_log_price_ret_volstock_mean_from_20',
        'wap1_log_price_ret_volstock_mean_from_25'][1]

        bad_up_pdp_feature_name = ['wap1_log_price_ret_volstock_mean_from_10',
                                'v1spprojt10f29','pear_corr_90_clusters_labels',\
                                'robust_sum_stats_60_clusters_labels','pear_corr_3_clusters_labels',\
                                'tlog_tlog1p_target_vol_robust_sum_stats_2_clusters',][2]
        bad_op_pdp_feature_name = ['root_book_delta_count',\
                                'soft_stock_mean_tvpl2_:10','soft_stock_mean_tvpl2_:20' ,\
                                'root_trade_count_smean'][0]
        decision_plot_feature_names = ['wap1_log_price_ret_volstock_mean_from_25', 'v1spprojt15f25', 'v1spprojt15f25_q1', 'soft_stock_mean_tvpl2_:20', 'tlog_target_vol_pcorr_3_clusters', 'pear_corr_90_clusters_labels', 'root_trade_count_smean', 'robust_sum_stats_60_clusters_labels', 'tlog_wap1_lprice_ret_vol_ati_*_wap1_lprice_ret_vol_ast_per_wap1_lprice_ret_vol_:20', 'v1proj_29_15_q3', 'log_liq2_ret_*_wap_eqi_price1_ret_vol_15_ratio_median_stock', 'wap1_log_price_ret_volstock_mean_from_20', 'tlog_wap1_lprice_ret_vol_ati_*_wap1_lprice_ret_avg_ast_per_wap1_lprice_ret_vol_:0', 'vol1_mean', 'texp_wap1_log_price_ret_vol_from_20', 'v1spprojt15f29_q3', 'liqvol1_smean', 'soft_stock_mean_tvpl2_liqf_volf20', 'tvpl2', 'v1proj_29_q3', 'v1spprojt15f29', 'tvpl2_rmed2v1', 'tlog_eps523_trade_price_n_wap_eqi_price0_dev', 'root_trade_count', 'tvpl2_smean_vol', 'tlog_wap1_lprice_ret_vol_ati_*_wap1_lprice_ret_avg_ast_per_wap1_lprice_ret_vol_:10', 'v1liq2sprojt5f25', 'wap1_log_price_ret_vol_from_25', 'tlog_target_vol_pcorr_10_clusters', 'v1proj_25_15_std', 'v1proj_29', 'tlog_wap1_log_price_ret_normalized*mean_centered_per_wap1_lprice_ret_vol_:20', 'tlog_tlog1p_target_vol_robust_sum_stats_2_clusters', 'pear_corr_3_clusters_labels', 'tlog_wap1_lprice_ret_vol_ati_*_wap1_lprice_ret_avg_ast_per_wap1_lprice_ret_vol_:20', 'v1liq2sprojt10f25', 'root_book_delta_count', 'wap1_log_price_ret_volstock_mean_from_10', 'wap1_log_price_ret_per_spread_sqr_vol_15_ratio_median_stock', 'pear_corr_10_clusters_labels', 'wap1_log_price_ret_volstock_mean_from_0', 'texp_wap1_log_price_ret_vol_from_10', 'v1spprojt10f29', 'wap1_log_price_ret_per_liq2_vol_15_ratio', 'log_wap1_log_price_ret_vol', 'v1proj_25', 'v1proj_25_std', 'tvpl2_rmed2v1lf25', 'tlog_tlog1p_target_vol_robust_sum_stats_4_clusters', 'tlog_eps523_trade_price_n_wap1_dev']
        ##### individual stock id analysis parameters END #####

        ############################ SET PARAMETERS HERE ##############################

        """ TRAINING SET PREDICTIONS START """

        ###################################################################################################################
        ############################################ TRAINING SET #########################################################
        print('\n########################################################################################################################')
        print('\n####################################### TRAINING SET predictions START #################################################')
        print('\n########################################################################################################################')
        ###################################################################################################################

        ################################################################################################
        ############################## OVERALL STOCK ANALYSIS START ######################################
        print('\n####################################### OVERALL STOCK ANALYSIS START ######################################')
        #train_rmspe_per_stock = self.overall_stock_id_analysis(unique_stock_ids,all_stock_train_pred_df,all_stock_y_train_df,train_residuals,train_flag=True)
        print('\n####################################### OVERALL STOCK ANALYSIS END ######################################')
        ############################## OVERALL STOCK ANALYSIS END ######################################
        ################################################################################################


        ################################################################################################
        ############################## INDIVIDUAL STOCK ANALYSIS START #################################
        print('\n############################## INDIVIDUAL STOCK ANALYSIS START #################################')
        ### #Analyze Single/ INDIVIDUAL stocks with high RMSPE in train set
        # for picked_stock_id in high_rmspe_stocks[start_index:end_index]:
        #     train_groundtruth = self.individual_stock_id_analysis(int(picked_stock_id),unique_stock_ids,all_stock_y_train_df,all_stock_train_pred_df,train_residuals,train_avg_target_rvol,train_flag=True)
        print('\n############################## INDIVIDUAL STOCK ANALYSIS END #################################')
        ############################## INDIVIDUAL STOCK ANALYSIS END #################################
        ################################################################################################




        ################################################################################################
        ############################## OVERALL TIME ID ANALYSIS START ##################################
        ################################################################################################
        print('\n############################## OVERALL TIME ANALYSIS START #################################')
        #train_gt_time_id_df, train_pred_time_id_df = self.overall_time_id_analysis(all_stock_y_train_df,all_stock_train_pred_df,train_avg_target_rvol, train_flag=True)
        print('\n############################## OVERALL TIME ANALYSIS END #################################')
        ################################################################################################
        ############################## OVERALL TIME ID ANALYSIS END #################################
        ################################################################################################




        ###################################################################################################################
        ###################################### Feature importance & SHAPLEY START #########################################
        ###################################################################################################################
        print('\n###################################### Feature importance & SHAPLEY START #########################################')
        print('\n#---------------------------------------------compute_global_SHAP_values----------------------------------------------------------------#')
        train_shap_values_all,train_top_2_interactions = self.compute_global_SHAP_values(final_reg,X_train,raw_train_gn,raw_train_pred,v1tr,set_name='train')
        os.chdir('/home/optimusprime/Desktop/peeterson/optiver/Optiver-Realized-Volatility-Prediction/data/Final_submission_data/Final_submission_data/shapley_values')
        #os.chdir('/content/drive/MyDrive/optiver_real_vol/Final Submission/kaggle_submission_versions')
        with open('train_shap_values_all_Non-st_31_xgb_model.pkl', 'wb') as file:
           pickle.dump(train_shap_values_all, file)
        with open('train_top_2_interactions_Non-st_31_xgb_model.pkl', 'wb') as file:
              pickle.dump(train_top_2_interactions, file)
        print('\n#-----------------------------------------------------------------------------------------------------------------#')

        print('\n#---------------------------------------------analyze_global_SHAP_values----------------------------------------------------------------#')
        # os.chdir('/home/optimusprime/Desktop/peeterson/optiver/Optiver-Realized-Volatility-Prediction/data/Final_submission_data/Final_submission_data/shapley_values')
        # # #os.chdir('/content/drive/MyDrive/optiver_real_vol/Final Submission/kaggle_submission_versions')
        # with open('train_shap_values_all_Non-st_31_xgb_model.pkl', 'rb') as file:
        #     train_shap_values_all = pickle.load( file)
        # with open('train_top_2_interactions_Non-st_31_xgb_model.pkl', 'rb') as file:
        #     train_top_2_interactions = pickle.load( file)
        # self.analyze_global_SHAP_values(train_shap_values_all,all_stock_raw_train_gn_df,all_stock_raw_train_pred_df,set_name='train')
        #top_n_feats = 50
        #self.compute_shapley_barplot(train_shap_values_all,top_n_feats,X_train,y_train['target'],stock_id=None,view_time_ids_start=None,view_time_ids_end=None,set_name='train')
        #self.compute_shapley_PDP_n_Scatter(good_pdp_feature_name,train_shap_values_all,all_stock_raw_train_gn_df,all_stock_raw_train_pred_df,stock_id=None,view_time_ids_start=None,view_time_ids_end=None,set_name='train')
        # self.compute_shapley_decision(train_shap_values_all.base_values,train_shap_values_all,decision_plot_feature_names,stock_id=None,view_time_ids_start=None,view_time_ids_end=None,set_name='train')
        print('\n#-----------------------------------------------------------------------------------------------------------------#')

        print('\n#---------------------------------------------compute_individual_stock_SHAP_values----------------------------------------------------------------#')
        feature_name = "log_first_10_min_vol_stnd" ## see impact of a feature in more detail
        stock_id = 0
        view_time_ids_start = 0
        view_time_ids_end = 500
        #self.compute_individual_stock_SHAP_values(final_reg,X_train,all_stock_train_pred_df,all_stock_v1tr_df,all_stock_y_train_df,feature_name,stock_id,view_time_ids_start,view_time_ids_end)
        print('\n#-----------------------------------------------------------------------------------------------------------------#')

        print('\n#---------------------------------------------compute_individual_instance_SHAP_values----------------------------------------------------------------#')
        #ovearll_error_idxs, up_error_idxs, op_error_idxs = self.identiy_largest_overall_n_under_n_over_prediction_errors(all_stock_y_train_df,all_stock_train_pred_df,set_name='train')
        #self.shapley_analysis_of_large_error_instances(ovearll_error_idxs, up_error_idxs, op_error_idxs,train_shap_values_all,set_name='train')
        print('\n#-----------------------------------------------------------------------------------------------------------------#')
        print('\n###################################### Feature importance & SHAPLEY END #########################################')
        ###################################################################################################################
        ###################################### Feature importance & SHAPLEY END ###########################################
        ###################################################################################################################





        ###################################################################################################################
        ###################################### MODEL BIAS VARINANCE START ################################################
        ###################################################################################################################

         #### Plot top 30 feature importances
        # fig, ax = plt.subplots(figsize=(10, 10))
        # xgb.plot_importance(final_reg, importance_type='gain', max_num_features=30, height=0.8, show_values=False)
        # self.compute_model_bias_variance(y_test,y_train,X_train,best_mlxtend_xgb_params)


        ###################################################################################################################
        ###################################### MODEL BIAS VARINANCE END #################################################
        ##################################################################################################################

        ###################################################################################################################
        ############################################ TRAINING SET PREDICTIONS END ##########################################
        ###################################################################################################################
        print('\n####################################### TRAINING SET predictions END #################################################\n')
        print('\n######################################################################################################################\n')



        """ TESTING SET PREDICTIONS START """

        ###################################################################################################################
        ############################################ TESTING SET PREDICTIONS START ########################################
        print('\n#######################################################################################################################')
        print('\n####################################### TESTING SET predictions START #################################################')
        print('\n#######################################################################################################################')
        ###################################################################################################################

        ################################################################################################
        ############################## OVERALL STOCK ANALYSIS START ######################################
        print('\n####################################### OVERALL STOCK ANALYSIS START ######################################')
        #test_rmspe_per_stock = self.overall_stock_id_analysis(unique_stock_ids,all_stock_test_pred_df,all_stock_y_test_df,test_residuals,train_flag=False)
        print('\n####################################### OVERALL STOCK ANALYSIS END ######################################')
        ############################## OVERALL STOCK ANALYSIS END ######################################
        ################################################################################################



        ################################################################################################
        ############################## INDIVIDUAL STOCK ANALYSIS START #################################
        print('\n############################## INDIVIDUAL STOCK ANALYSIS START #################################')
        #### Analyze Single/ INDIVIDUAL stocks with high RMSPE in train set
        # for picked_stock_id in high_rmspe_stocks[start_index:end_index]:
        #     test_groundtruth = self.individual_stock_id_analysis(int(picked_stock_id),unique_stock_ids,all_stock_y_test_df,all_stock_test_pred_df,test_residuals,test_avg_target_rvol,train_flag=False)
        print('\n############################## INDIVIDUAL STOCK ANALYSIS END #################################')
        ############################## INDIVIDUAL STOCK ANALYSIS END #################################
        ################################################################################################


        ################################################################################################
        ############################## OVERALL TIME ID ANALYSIS START ##################################
        ################################################################################################
        print('\n############################## OVERALL TIME ANALYSIS START #################################')
        #test_gt_time_id_df, test_pred_time_id_df = self.overall_time_id_analysis(all_stock_y_test_df,all_stock_test_pred_df,test_avg_target_rvol, train_flag=False)
        print('\n############################## OVERALL TIME ANALYSIS END #################################')
        ################################################################################################
        ############################## OVERALL TIME ID ANALYSIS END #################################
        ################################################################################################



        ###################################################################################################################
        ###################################### TEST Feature importance & SHAPLEY START #########################################
        ###################################################################################################################

        print('\n###################################### TEST Feature importance & SHAPLEY START #########################################')
        print('\n#---------------------------------------------compute_global_SHAP_values----------------------------------------------------------------#')
        test_shap_values_all,test_top_2_interactions = self.compute_global_SHAP_values(final_reg,X_test,raw_test_gn,raw_test_pred,v1ts,set_name='test')
        os.chdir('/home/optimusprime/Desktop/peeterson/optiver/Optiver-Realized-Volatility-Prediction/data/Final_submission_data/Final_submission_data/shapley_values')
        #os.chdir('/content/drive/MyDrive/optiver_real_vol/Final Submission/kaggle_submission_versions')
        with open('test_shap_values_all_Non-st_31_xgb_model.pkl', 'wb') as file:
            pickle.dump(test_shap_values_all, file)
        with open('test_top_2_interactions_Non-st_31_xgb_model.pkl', 'wb') as file:
            pickle.dump(test_top_2_interactions, file)
        print('\n#-----------------------------------------------------------------------------------------------------------------#')

        print('\n#---------------------------------------------analyze_global_SHAP_values----------------------------------------------------------------#')
        # os.chdir('/home/optimusprime/Desktop/peeterson/optiver/Optiver-Realized-Volatility-Prediction/data/Final_submission_data/Final_submission_data/shapley_values')
        # # #os.chdir('/content/drive/MyDrive/optiver_real_vol/Final Submission/kaggle_submission_versions')
        # with open('test_shap_values_all_Non-st_31_xgb_model.pkl', 'rb') as file:
        #     test_shap_values_all = pickle.load( file)
        # with open('test_top_2_interactions_Non-st_31_xgb_model.pkl', 'rb') as file:
        #     test_top_2_interactions = pickle.load( file)
        # self.analyze_global_SHAP_values(test_shap_values_all,all_stock_raw_test_gn_df,all_stock_raw_test_pred_df,set_name='test')
        # self.compute_shapley_barplot(test_shap_values_all,top_n_feats,X_test,y_test['target'],stock_id=None,view_time_ids_start=None,view_time_ids_end=None,set_name='test')
        #self.compute_shapley_PDP_n_Scatter(good_pdp_feature_name,test_shap_values_all,all_stock_raw_test_gn_df,all_stock_raw_test_pred_df,stock_id=None,view_time_ids_start=None,view_time_ids_end=None,set_name='test')
        # self.compute_shapley_decision(test_shap_values_all.base_values,test_shap_values_all,decision_plot_feature_names,stock_id=None,view_time_ids_start=None,view_time_ids_end=None,set_name='test')
        print('\n#-----------------------------------------------------------------------------------------------------------------#')

        print('\n#---------------------------------------------compute_individual_stock_SHAP_values-------------------------------------------------------#')
        feature_name = "log_first_10_min_vol_stnd" ## see impact of a feature in more detail
        stock_id = 0
        view_time_ids_start = 0
        view_time_ids_end = 500
        #self.compute_individual_stock_SHAP_values(final_reg,X_test,all_stock_test_pred_df,all_stock_v1tr_df,all_stock_y_test_df,feature_name,stock_id,view_time_ids_start,view_time_ids_end)
        print('\n#-----------------------------------------------------------------------------------------------------------------#')


        print('\n#---------------------------------------------compute_individual_instance_SHAP_values-------------------------------------------------------#')
        #ovearll_error_idxs, up_error_idxs, op_error_idxs = self.identiy_largest_overall_n_under_n_over_prediction_errors(all_stock_y_test_df,all_stock_test_pred_df,set_name='test')
        #self.shapley_analysis_of_large_error_instances(ovearll_error_idxs, up_error_idxs, op_error_idxs,test_shap_values_all,set_name='test')
        print('\n#-----------------------------------------------------------------------------------------------------------------#')
        print('\n###################################### TEST Feature importance & SHAPLEY END #########################################')
        ###################################################################################################################
        ###################################### Feature importance & SHAPLEY END #########################################
        ###################################################################################################################


        ###################################################################################################################
        ############################################ TESTING SET PREDICTIONS END ##########################################
        ###################################################################################################################
        print('##################################################################################################')
        print('\n####################################### TESTING SET predictions END #################################################\n')




        """ TRAINING AND TESTING SET PERFORMANCE COMPARISON START """

        ###################################################################################################################
        ############################################ TRAINING AND TESTING SET PERFORMANCE COMPARISON START ##########################################
        ###################################################################################################################
        print('\n####################################### TRAINING AND TESTING SET stocks RMSPE COMPARISON START #################################################')
        #self.train_n_test_set_st_RMSPE_comparison(train_rmspe_per_stock,test_rmspe_per_stock)
        print('##################################################################################################')
        print('\n####################################### TRAINING AND TESTING SET stocks RMSPE COMPARISON END #################################################\n')

        ###################################################################################################################
        print('\n####################################### TRAINING AND TESTING SET stocks DISTRIBUTION COMPARISON START #################################################')
        #self.train_n_test_set_picked_st_distribution_comparison(train_groundtruth, test_groundtruth,high_rmspe_stocks[start_index:end_index])
        print('##################################################################################################')
        print('\n####################################### TRAINING AND TESTING SET stocks DISTRIBUTION COMPARISON END #################################################\n')
        ###################################################################################################################


        ###################################################################################################################
        print('##################################################################################################')
        print('\n################## time_id RMSPE COMPARISON between groundtruth and prediction on train set START ########################')
        #self.each_time_id_RMSPE_across_stocks_comparison(train_gt_time_id_df, train_pred_time_id_df,set_name='train',y_train_df = all_stock_y_train_df)
        print('\n################## time_id RMSPE COMPARISON between groundtruth and prediction on train set END ########################')
        print('\n################## time_id RMSPE COMPARISON between groundtruth and prediction on test set START ##########################\n')
        #self.each_time_id_RMSPE_across_stocks_comparison(test_gt_time_id_df, test_pred_time_id_df,set_name='test',y_train_df = all_stock_y_train_df)
        print('\n################## time_id RMSPE COMPARISON between groundtruth and prediction on test set END ##########################\n')
        print('##################################################################################################')
        ###################################################################################################################


        ###################################################################################################################
        print('\n####################################### Representativeness of validation set in test set START #################################################')
        ###################################################################################################################
        #self.val_set_n_test_set_representation_plot(mean_val_set_rmspe_error, test_set_rmspe_error, train_set_rmspe_error)
        ###################################################################################################################
        print('\n####################################### Representativeness of validation set in test set END #################################################')
        ###################################################################################################################


        ###################################################################################################################
        ############################################ TRAINING AND TESTING SET COMPARISON END ##########################################
        ###################################################################################################################


        del X_train,y_train, all_stock_train_pred_df, all_stock_v1tr_df ,  all_stock_test_pred_df, all_stock_y_train_df,  all_stock_y_test_df
        del y_true, y_pred, test_residuals, train_residuals, unique_stock_ids, all_stock_v1ts_df, train_avg_target_rvol, test_avg_target_rvol
        gc.collect()
        return



    def visualize_tree(self,):
        # feature_importances['average'] = feature_importances[[f'fold_{fold_n + 1}' for fold_n in range(folds.n_splits)]].mean(axis=1)
        # feature_importances.to_csv('feature_importances.csv')
        # plt.figure(figsize=(16, 12))
        # sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(20), x='average', y='feature')
        # plt.title('20 TOP feature importance over {} folds average'.format(folds.n_splits));

        # importances = pd.DataFrame({'Feature': model.feature_name(),
        #                             'Importance': sum( [model.feature_importance(importance_type='gain') for model in models] )})
        # importances2 = importances.nlargest(40,'Importance', keep='first').sort_values(by='Importance', ascending=True)
        # importances2[['Importance', 'Feature']].plot(kind = 'barh', x = 'Feature', figsize = (8,6), color = 'blue', fontsize=11);plt.ylabel('Feature', fontsize=12)

        #TODO: #plot decision tree for interpretability

        return





# %%

# def objective_st31(trial):

#     t_v_t = train_validate_n_test(df_train_reordered_for_stock_31, df_test_for_stock_31)

#     ######  SET Hyperparameter's range for tuning ######
#     # Hyperparameters and algorithm parameters are described here
#     seed1=11
#     missing_value = -np.inf   # Replace with a suitable value
#     early_stopping_rounds = 25
#     num_round= 1000 # num_trees
#     params = {'disable_default_eval_metric': 1,
#               "max_depth": trial.suggest_int('max_depth', 5, 30),
#             "eta": trial.suggest_float(name='eta', low=0.00001, high=0.2,log=True),
#             "subsample" : round(trial.suggest_float(name='subsample', low=0.6, high=1.0,step=0.05),1),
#             "colsample_bytree": round(trial.suggest_float(name='colsample_bytree', low=0.5, high=1,step=0.05),1),
#             'gamma': trial.suggest_int('gamma', 1, 10),
#             'reg_alpha': trial.suggest_int('reg_alpha', 1, 14),
#             'reg_lambda': trial.suggest_int('reg_lambda', 1, 14),
#             'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
#             "tree_method": 'hist',
#             "device": "cuda",
#             "seed":seed1,
#             #'missing': missing_value
#             }

#     # params = {'disable_default_eval_metric': 1,
#     #           "max_depth": trial.suggest_int('max_depth', 25, 35),
#     #         "eta": trial.suggest_float(name='eta', low=0.0001, high=0.01,log=True),
#     #         "subsample" : round(trial.suggest_float(name='subsample', low=0.8, high=1.0,step=0.05),1),
#     #         "colsample_bytree": round(trial.suggest_float(name='colsample_bytree', low=0.5, high=0.8,step=0.05),1),
#     #         'gamma': trial.suggest_int('gamma', 0, 5),
#     #         'reg_alpha': trial.suggest_int('reg_alpha', 0, 5),
#     #         'reg_lambda': trial.suggest_int('reg_lambda', 0, 5),
#     #         'min_child_weight': trial.suggest_int('min_child_weight', 2, 6),
#     #         "tree_method": 'hist',
#     #         "device": "cuda",
#     #         "seed":seed1,
#     #         #'missing': missing_value
#     #         }
#     ######  SET Hyperparameter's range for tuning ######




#     val_avg_error,best_iteration = t_v_t.xgb_train_validate(params,num_round,early_stopping_rounds,trial)
#     print(f"val_avg_error: {val_avg_error}, best_iteration: {best_iteration}")
#     trial.set_user_attr("best_iteration", best_iteration)

#     del t_v_t
#     gc.collect()
#     return val_avg_error



# %%


# #if __name__ == "__main__":

# #optuna.logging.set_verbosity(optuna.logging.WARNING)
# # study_name= 'Correct_residual_autocorrrelation_HAR_n_target_lag_feat_n_target_pred'

# mean_val_set_rmspe_error = [ 0.226999,0.226498,0.226883,0.239605]

# study = optuna.create_study(study_name ='stock 31 model' ,direction="minimize")
# study.optimize(objective_st31, n_trials=50) # 75

# pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
# complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

# print("Study statistics: ")
# print("  Number of finished trials: ", len(study.trials))
# print("  Number of pruned trials: ", len(pruned_trials))
# print("  Number of complete trials: ", len(complete_trials))

# print("Best trial:")
# trial = study.best_trial

# print("Best number of iteration/boosting rounds: ",study.trials[trial.number].user_attrs['best_iteration'])

# print("Trial no.: ",trial.number)
# print("  Value: ", trial.value)
# mean_val_set_rmspe_error.append(trial.value)

# print("  Params: ")
# for key, value in trial.params.items():
#     print("    {}: {}".format(key, value))

# #print("Best hyperparameters:", study.best_params)

# fig = optuna.visualization.plot_parallel_coordinate(study)
# fig.show()

# fig = optuna.visualization.plot_optimization_history(study)
# fig.show()

# fig = optuna.visualization.plot_slice(study)
# fig.show()

# fig = optuna.visualization.plot_param_importances(study)
# fig.show()



# %%

# ############ Best parameters Manual Start ############
# # num_rounds = 1135 #838 #study.trials[trial.number].user_attrs['best_iteration']
# # max_depth = 17
# # eta =  0.012302593098587278
# # subsample =  0.75
# # colsample_bytree =  0.8
# # gamma = 0
# # reg_alpha =  11
# # reg_lambda = 8
# # min_child_weight =  6
# ############ Best parameters Manual End ############

# ############ Best parameters Automatic Start ############
# best_trial = study.best_trial
# num_rounds = study.best_trial.user_attrs['best_iteration']
# print('final best iteration: ',num_rounds )
# seed1 = 11
# missing_value = -np.inf  # Replace with a suitable value
# max_depth = best_trial.params['max_depth']
# eta =  best_trial.params['eta']
# subsample =  best_trial.params['subsample']
# colsample_bytree =  best_trial.params['colsample_bytree']
# gamma =  best_trial.params['gamma']
# reg_alpha =  best_trial.params['reg_alpha']
# reg_lambda = best_trial.params['reg_lambda']
# min_child_weight = best_trial.params['min_child_weight']
# ############ Best parameters Automatic End ############





# best_mlxtend_xgb_params = [max_depth,eta,subsample,colsample_bytree,gamma,reg_alpha,reg_lambda,min_child_weight,num_rounds]

# best_params = { 'disable_default_eval_metric': 1,
#               "max_depth": max_depth,
#             "eta": eta,
#             "subsample" : subsample,
#             "colsample_bytree":  colsample_bytree,
#             'gamma':gamma,
#             'reg_alpha': reg_alpha,
#             'reg_lambda': reg_lambda,
#             'min_child_weight': min_child_weight,
#             "tree_method": 'hist',
#             "device": "cuda",
#             "seed":seed1,
#             #'missing': missing_value
#                }

# t_v_t = train_validate_n_test(df_train_reordered_for_stock_31, df_test_for_stock_31)
# final_reg,train_pred,test_pred,y_train,X_train,X_test,v1tr,v1ts,w_train,target_name = t_v_t.make_predictions(best_params,num_rounds)
# #target_name = 'target'

# train_pred = pd.DataFrame(train_pred).rename(columns={'log_wap1_log_price_ret_vol':'target'})
# train_pred['time_id'] = df_train_reordered['time_id']
# train_pred['stock_id'] = df_train_reordered['stock_id']

# y_train = pd.DataFrame(y_train).rename(columns={'log_wap1_log_price_ret_vol':'target'})
# y_train['time_id'] = df_train_reordered['time_id']
# y_train['stock_id'] = df_train_reordered['stock_id']

# v1tr = pd.DataFrame(v1tr)
# v1tr['time_id'] = df_train_reordered['time_id']
# v1tr['stock_id'] = df_train_reordered['stock_id']
# v1tr['wap1_log_price_ret_vol'] = v1tr['log_wap1_log_price_ret_vol']
# v1tr.drop(columns=['log_wap1_log_price_ret_vol'], inplace=True)

# test_pred = pd.DataFrame(test_pred).rename(columns={'log_wap1_log_price_ret_vol':'target'})
# test_pred['time_id'] = df_test['time_id'].astype(int)
# test_pred['stock_id'] = df_test['stock_id'].astype(int)

# ## # Merge the DataFrames on 'time_id' and 'stock_id' columns
# #os.chdir('/home/optimusprime/Desktop/peeterson/optiver/Optiver-Realized-Volatility-Prediction/data')
# #os.chdir('/content/drive/MyDrive/optiver_real_vol/Final Submission/kaggle_submission_versions')

# v1ts = pd.DataFrame(v1ts)
# v1ts['time_id'] = df_test['time_id'].astype(int)
# v1ts['stock_id'] = df_test['stock_id'].astype(int)
# v1ts['wap1_log_price_ret_vol'] = v1ts['log_wap1_log_price_ret_vol']
# v1ts.drop(columns=['log_wap1_log_price_ret_vol'], inplace=True)

# train = pd.read_csv('train.csv')
# X_test['time_id'] = df_test['time_id']
# X_test['stock_id'] = df_test['stock_id']
# y_test_df = pd.merge(X_test[['time_id', 'stock_id']], train[['time_id', 'stock_id', 'target']], on=['time_id', 'stock_id'], how='left')
# y_test_df['time_id'] = df_test['time_id'].astype(int)
# y_test_df['stock_id'] = df_test['stock_id'].astype(int)
# w_test = y_test_df['target'] **-2 * v1ts['wap1_log_price_ret_vol']**2
# X_test.drop(columns=['time_id'], inplace=True)




# %%
# t_v_t.evaluate_predictions(final_reg,test_pred, y_test_df,train_pred,y_train,X_train,X_test,v1tr,v1ts,w_train,w_test,best_mlxtend_xgb_params,mean_val_set_rmspe_error)

# train_pred_stock_31 = train_pred[train_pred['stock_id'] == 31].copy()
# test_pred_stock_31 = test_pred[test_pred['stock_id'] == 31].copy()

# del t_v_t,final_reg,train_pred,y_test_df,test_pred,y_train,X_train,X_test,v1tr,v1ts,w_train,w_test, df_train_reordered_for_stock_31, df_test_for_stock_31,best_mlxtend_xgb_params


# %%

def objective(trial):


    t_v_t = train_validate_n_test(df_train_reordered, df_test)

    ######  SET Hyperparameter's range for tuning ######
    # Hyperparameters and algorithm parameters are described here
    seed1=11
    missing_value = -np.inf   # Replace with a suitable value
    early_stopping_rounds = 50
    num_round= 7000 # num_trees
    params = {'disable_default_eval_metric': 1,
              "max_depth": trial.suggest_int('max_depth', 20, 80),
            "eta": trial.suggest_float(name='eta', low=0.00001, high=0.2,log=True),
            "subsample" : round(trial.suggest_float(name='subsample', low=0.6, high=1.0,step=0.05),1),
            "colsample_bytree": round(trial.suggest_float(name='colsample_bytree', low=0.5, high=1,step=0.05),1),
            'gamma': trial.suggest_int('gamma', 1, 15),
            'reg_alpha': trial.suggest_int('reg_alpha', 1, 20),
            'reg_lambda': trial.suggest_int('reg_lambda', 1, 20),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            "tree_method": 'hist',
            "device": "cuda",
            "seed":seed1,
            #'missing': missing_value
            }

#     params = {'disable_default_eval_metric': 1,
#               "max_depth": trial.suggest_int('max_depth', 25, 35),
#             "eta": trial.suggest_float(name='eta', low=0.0001, high=0.01,log=True),
#             "subsample" : round(trial.suggest_float(name='subsample', low=0.8, high=1.0,step=0.05),1),
#             "colsample_bytree": round(trial.suggest_float(name='colsample_bytree', low=0.5, high=0.8,step=0.05),1),
#             'gamma': trial.suggest_int('gamma', 0, 5),
#             'reg_alpha': trial.suggest_int('reg_alpha', 0, 5),
#             'reg_lambda': trial.suggest_int('reg_lambda', 0, 5),
#             'min_child_weight': trial.suggest_int('min_child_weight', 2, 6),
#             "tree_method": 'hist',
#             "device": "cuda",
#             "seed":seed1,
#             #'missing': missing_value
#             }
    ######  SET Hyperparameter's range for tuning ######




    val_avg_error,best_iteration = t_v_t.xgb_train_validate(params,num_round,early_stopping_rounds,trial)
    print(f"val_avg_error: {val_avg_error}, best_iteration: {best_iteration}")
    trial.set_user_attr("best_iteration", best_iteration)

    del t_v_t
    gc.collect()
    return val_avg_error



# %%


#if __name__ == "__main__":

#optuna.logging.set_verbosity(optuna.logging.WARNING)
# study_name= 'Correct_residual_autocorrrelation_HAR_n_target_lag_feat_n_target_pred'

mean_val_set_rmspe_error = [ 0.226999,0.226498,0.226883,0.239605]

study = optuna.create_study(study_name ='reduce gaps (variance reduction) in-between target data points, log of jager transform' ,direction="minimize")
study.optimize(objective, n_trials=50) # 75

pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial

print("Best number of iteration/boosting rounds: ",study.trials[trial.number].user_attrs['best_iteration'])

print("Trial no.: ",trial.number)
print("  Value: ", trial.value)
mean_val_set_rmspe_error.append(trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

#print("Best hyperparameters:", study.best_params)

fig = optuna.visualization.plot_parallel_coordinate(study)
fig.show()

fig = optuna.visualization.plot_optimization_history(study)
fig.show()

fig = optuna.visualization.plot_slice(study)
fig.show()

fig = optuna.visualization.plot_param_importances(study)
fig.show()



# %%

############ Best parameters Manual Start ############
# seed1 = 11
# num_rounds = 2060 #838 #study.trials[trial.number].user_attrs['best_iteration']
# max_depth = 36
# eta =  0.0019727795380054263
# subsample =  0.65
# colsample_bytree =  0.55
# gamma = 1
# reg_alpha =  11
# reg_lambda = 3
# min_child_weight =  15
############ Best parameters Manual End ############

############ Best parameters Automatic Start ############
best_trial = study.best_trial
num_rounds = study.best_trial.user_attrs['best_iteration']
print('final best iteration: ',num_rounds )
seed1 = 11
missing_value = -np.inf  # Replace with a suitable value
max_depth = best_trial.params['max_depth']
eta =  best_trial.params['eta']
subsample =  best_trial.params['subsample']
colsample_bytree =  best_trial.params['colsample_bytree']
gamma =  best_trial.params['gamma']
reg_alpha =  best_trial.params['reg_alpha']
reg_lambda = best_trial.params['reg_lambda']
min_child_weight = best_trial.params['min_child_weight']
############ Best parameters Automatic End ############


mean_val_set_rmspe_error = [ 0.226999,0.226498,0.226883,0.239605]


best_mlxtend_xgb_params = [max_depth,eta,subsample,colsample_bytree,gamma,reg_alpha,reg_lambda,min_child_weight,num_rounds]

best_params = { 'disable_default_eval_metric': 1,
              "max_depth": max_depth,
            "eta": eta,
            "subsample" : subsample,
            "colsample_bytree":  colsample_bytree,
            'gamma':gamma,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'min_child_weight': min_child_weight,
            "tree_method": 'hist',
            "device": "cuda",
            "seed":seed1,
            #'missing': missing_value
               }

t_v_t = train_validate_n_test(df_train_reordered, df_test)
final_reg,train_pred,test_pred,y_train,X_train,X_test,v1tr,v1ts,w_train,target_name,raw_train_pred, raw_test_pred,raw_train_gn,target_shift = t_v_t.make_predictions(best_params,num_rounds)
#target_name = 'target'

train_time_id = df_train_reordered['time_id'].astype(int).values
train_stock_id = df_train_reordered['stock_id'].astype(int).values
test_time_id = df_test['time_id'].astype(int).values
test_stock_id = df_test['stock_id'].astype(int).values

train_pred = pd.DataFrame(train_pred).rename(columns={'log_wap1_log_price_ret_vol':'target'})
train_pred['time_id'] = train_time_id
train_pred['stock_id'] = train_stock_id
train_pred.reset_index(drop=True, inplace=True)

raw_train_pred = pd.DataFrame(raw_train_pred).rename(columns={0:'target'})
raw_train_pred['time_id'] = train_time_id
raw_train_pred['stock_id'] = train_stock_id
raw_train_pred.reset_index(drop=True, inplace=True)

y_train = pd.DataFrame(y_train).rename(columns={'log_wap1_log_price_ret_vol':'target'})
y_train['time_id'] = train_time_id
y_train['stock_id'] = train_stock_id
y_train.reset_index(drop=True, inplace=True)

raw_train_gn = pd.DataFrame(raw_train_gn).rename(columns={0:'target'})
raw_train_gn['time_id'] = train_time_id
raw_train_gn['stock_id'] = train_stock_id
raw_train_gn.reset_index(drop=True, inplace=True)


v1tr = pd.DataFrame(v1tr)
v1tr['time_id'] = train_time_id
v1tr['stock_id'] = train_stock_id
v1tr['wap1_log_price_ret_vol'] = v1tr['log_wap1_log_price_ret_vol']
v1tr.drop(columns=['log_wap1_log_price_ret_vol'], inplace=True)
v1tr.reset_index(drop=True, inplace=True)

test_pred = pd.DataFrame(test_pred).rename(columns={'log_wap1_log_price_ret_vol':'target'})
test_pred['time_id'] = test_time_id
test_pred['stock_id'] = test_stock_id
test_pred.reset_index(drop=True, inplace=True)

raw_test_pred = pd.DataFrame(raw_test_pred).rename(columns={0:'target'})
raw_test_pred['time_id'] = test_time_id
raw_test_pred['stock_id'] = test_stock_id
raw_test_pred.reset_index(drop=True, inplace=True)

## # Merge the DataFrames on 'time_id' and 'stock_id' columns
os.chdir('/home/optimusprime/Desktop/peeterson/optiver/Optiver-Realized-Volatility-Prediction/data')
#os.chdir('/content/drive/MyDrive/optiver_real_vol/Final Submission/kaggle_submission_versions')

v1ts = pd.DataFrame(v1ts)
v1ts['time_id'] = test_time_id
v1ts['stock_id'] = test_stock_id
v1ts['wap1_log_price_ret_vol'] = v1ts['log_wap1_log_price_ret_vol']
v1ts.drop(columns=['log_wap1_log_price_ret_vol'], inplace=True)
v1ts.reset_index(drop=True, inplace=True)

train = pd.read_csv('train.csv')
X_test['time_id'] = test_time_id
X_test['stock_id'] = test_stock_id
y_test_df = pd.merge(X_test[['time_id', 'stock_id']], train[['time_id', 'stock_id', 'target']], on=['time_id', 'stock_id'], how='left').reset_index(drop=True)
y_test_df['time_id'] = test_time_id
y_test_df['stock_id'] = test_stock_id
y_test_df.reset_index(drop=True, inplace=True)

w_test = y_test_df['target'] **-2 * v1ts['wap1_log_price_ret_vol']**2
X_test.drop(columns=['time_id'], inplace=True)
#X_test.drop(columns=['stock_id'], inplace=True)
#X_test.reset_index(drop=True, inplace=True)


raw_test_gn = pd.DataFrame()
raw_test_gn['target'] = np.log(y_test_df['target'].values / v1ts['wap1_log_price_ret_vol'].values) + target_shift
raw_test_gn['time_id'] = test_time_id
raw_test_gn['stock_id'] = test_stock_id
raw_test_gn.reset_index(drop=True, inplace=True)

# %%
t_v_t.evaluate_predictions(final_reg,test_pred, y_test_df,train_pred,y_train,X_train,X_test,v1tr,v1ts,w_train,w_test,best_mlxtend_xgb_params,mean_val_set_rmspe_error,raw_train_pred,raw_test_pred,raw_train_gn,raw_test_gn)


# %%
# xgb_n_shap_important_feat_interaction_dict = {'liqvol1_smean': ['tvpl3_rmed2v1lf20', 'tlog_target_vol_pcorr_3_clusters'],
#  'log_liq2_ret_*_wap_eqi_price1_ret_vol_15_ratio_median_stock':['tlog_wap1_lprice_ret_vol_ati_*_wap1_lprice_ret_avg_ast_per_wap1_lprice_ret_vol_:0','tlog_wap1_lprice_ret_vol_ati_*_wap1_lprice_ret_avg_ast_per_wap1_lprice_ret_vol_:10'],
#  #'pear_corr_90_clusters_labels':['tvpl2','tvpl2_liqf10'],
#  #'robust_sum_stats_60_clusters_labels':['soft_stock_mean_tvpl2_liqf','soft_stock_mean_tvpl2_liqf_volf20'],
#  'root_trade_count_smean':['tlog_target_vol_pcorr_3_clusters','tvpl3_rmed2v1lf20'],
#  'soft_stock_mean_tvpl2_:20':['tvpl3_rmed2v1lf20','tlog_target_vol_pcorr_3_clusters'],
#  'soft_stock_mean_tvpl2_liqf_volf20':['tvpl3_rmed2v1lf20','tlog_target_vol_pcorr_3_clusters'],
#  'texp_wap1_log_price_ret_vol_from_20':['wap1_log_price_ret_quart_volstock_mean_from_25','wap1_log_price_ret_quart_volstock_mean_from_20'],
#  'tlog_eps523_trade_price_n_wap_eqi_price0_dev':['wap1_log_price_ret_quart_volstock_mean_from_0','wap1_log_price_ret_volstock_mean_from_0'],
#  'tlog_target_vol_pcorr_3_clusters':['root_trade_count_smean','soft_stock_mean_tvpl2_:0'],
#  'tlog_wap1_lprice_ret_vol_ati_*_wap1_lprice_ret_vol_ast_per_wap1_lprice_ret_vol_:20':['v1spprojt15f25','v1spprojt15f29'],
#  #'tvpl2':['robust_sum_stats_20_clusters_labels','robust_sum_stats_32_clusters_labels'],
#  'tvpl2_smean_vol':['soft_stock_mean_tvpl2_:0','soft_stock_mean_tvpl2_:10'],
#  #'tvpl3':['v1spprojt15f29',],
#  'tvpl3_rmed2v1lf20':['soft_stock_mean_tvpl2_:20','soft_stock_mean_tvpl2_:10'],
#  'v1proj_25':['v1proj_25_15_q1','v1proj_25_q1'],
#  'v1proj_29_15_q3':['root_trade_count_smean','v1proj_29_q1'],
#  'v1proj_29_q3':['v1proj_25_15','v1proj_29_15_q3'],
#  #'v1spprojt15f25':['tvpl3','tvpl3_liqf10'],
#  'v1spprojt15f25_q1':['root_trade_count_smean','v1proj_25'],
#  #'wap1_log_price_ret_vol_from_25':['wap1_log_price_ret_quart_volstock_mean_from_25','wap1_log_price_ret_quart_volstock_mean_from_20'],
#  'wap1_log_price_ret_volstock_mean_from_20':['v1spprojt15f29','v1spprojt15f25'],
#  'wap1_log_price_ret_volstock_mean_from_25':['v1spprojt15f25','v1spprojt15f29']
# }

# interaction_terms_list = []

# %%
# for k in xgb_n_shap_important_feat_interaction_dict.keys():
#     feat = k
#     feat1 = xgb_n_shap_important_feat_interaction_dict[k][0]
#     feat2 = xgb_n_shap_important_feat_interaction_dict[k][1]

#     print(f'cross btw. {feat} and {feat1}')
#     crossed_feat1 = df_train_reordered[feat]*df_train_reordered[feat1]
#     no_outliers1 = crossed_feat1[(crossed_feat1 != 1e8) & (crossed_feat1 != -1e8)]
#     #crossed_feat1 = df_test[feat]*df_test[feat1]
#     print(no_outliers1.describe())
#     no_outliers1.plot(kind='hist', bins=1000, title=f'cross btw. {feat} and {feat1}')
#     plt.show()
#     interaction_terms_list.append((feat,feat1))

#     crossed_feat2 = df_train_reordered[feat]*df_train_reordered[feat2]
#     #crossed_feat2 = df_test[feat]*df_test[feat2]
#     no_outliers2 = crossed_feat2[(crossed_feat2 != 1e8) & (crossed_feat2 != -1e8)]
#     print(f'cross btw. {feat} and {feat2}')
#     print(no_outliers2.describe())
#     no_outliers2.plot(kind='hist', bins=1000, title=f'cross btw. {feat} and {feat2}')
#     plt.show()
#     interaction_terms_list.append((feat,feat2))

# %%



