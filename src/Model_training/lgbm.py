############## IMPORTS #####################
import pickle
import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import RepeatedKFold, cross_val_score, TimeSeriesSplit

import statsmodels as sm
from statsmodels.genmod.generalized_linear_model import GLM

import warnings
#warnings.filterwarnings("ignore")
from sklearn.metrics import confusion_matrix
#from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


from sklearn.utils import class_weight
import optuna
from optuna.trial import TrialState





############## get labels of features and target #####################

os.chdir('/home/optimusprime/Desktop/peeterson/optiver/Optiver-Realized-Volatility-Prediction/data/liquidity_features')

with open('int32_feat_labels.pkl', 'rb') as fp:
    int32_feat_labels = pickle.load(fp)
with open('int64_feat_labels.pkl', 'rb') as fp:
    int64_feat_labels = pickle.load(fp)

with open('float32_feat_labels.pkl', 'rb') as fp:
    float32_feat_labels = pickle.load(fp)
with open('float64_feat_labels.pkl', 'rb') as fp:
    float64_feat_labels = pickle.load(fp)

with open('target_labels.pkl', 'rb') as fp:
    target_labels = pickle.load(fp)

os.chdir('/home/optimusprime/Desktop/peeterson/optiver/Optiver-Realized-Volatility-Prediction/data/liquidity_features')
with open('train_feat_df_reordered.pkl', 'rb') as fp:
    train_feat_df_reordered = pickle.load(fp)

df = train_feat_df_reordered.copy()
del train_feat_df_reordered




############## TRAIN VALIDATION SPLIT #####################

class train_validate_n_test(object):

    def __init__(self,df) -> None:

        self.time_id_order = df.loc[:3829,'time_id'].values # select ordered unique time_ids
        train_time_id_ind = int(len(self.time_id_order)*0.7)
        #train_time_ids = time_id_order[:train_time_id_ind]
        test_time_ids = self.time_id_order[train_time_id_ind:]
        self.test_df = df[df['time_id'].isin(test_time_ids)]

        self.n_folds = 30
        folds = TimeSeriesSplit(n_splits=self.n_folds,)# max_train_size=None, gap=0)
        train_end_ind = int(len(self.time_id_order)*0.7) # index at 70% of time_ids
        self.splits = folds.split( range( train_end_ind ) ) # split 70% train time_ids into n_fold splits

        feature_importances = pd.DataFrame()
        self.feat_cols_list = int32_feat_labels+int64_feat_labels+float32_feat_labels+float64_feat_labels
        feature_importances['feature'] = self.feat_cols_list


    #### RMSPE cost function
    def rmspe(self,y_true, y_pred):
        return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))


    def lgbm_RMSPE(self,preds, train_data):
        labels = train_data.get_label()
        return 'RMSPE', round(self.rmspe(y_true = labels, y_pred = preds),5), False


    def nancorr(self,a, b):
        v = np.isfinite(a)*np.isfinite(b) > 0
        return np.corrcoef(a[v], b[v])[0,1]


    def lgbm_train_validate(self,params_lgbm,n_rounds):
        rmspe_val_score = []
        models= []
        test_y_preds = np.zeros(len(self.test_df))

        for fold_n, (train_index, valid_index) in enumerate(self.splits):
            print('Fold:',fold_n+1)
            print('train_index',train_index)
            print('valid_index',valid_index)
            train_time_ids = self.time_id_order[train_index]
            val_time_ids = self.time_id_order[valid_index]
            train_df = df[df['time_id'].isin(train_time_ids)]
            val_df = df[df['time_id'].isin(val_time_ids)]

            X_train = train_df[self.feat_cols_list]
            y_train = train_df['target']
            X_valid = val_df[self.feat_cols_list]
            y_valid = val_df['target']

            v1tr = np.exp(X_train['log_wap1_log_price_ret_vol'])
            v1v = np.exp(  X_valid['log_wap1_log_price_ret_vol'])
            v1ts = np.exp( self.test_df['log_wap1_log_price_ret_vol'])


            y_train = y_train
            w_train = y_train **-2 * v1tr**2

            y_val = y_valid
            w_val = y_valid **-2 * v1v**2

            dtrain = lgb.Dataset(X_train, label=y_train/v1tr, categorical_feature=int64_feat_labels + int32_feat_labels, weight=w_train)
            dvalid = lgb.Dataset(X_valid,   label=  y_val/v1v,   categorical_feature=int64_feat_labels + int32_feat_labels, weight=w_val  )


            print('model')
            clf = lgb.train(params_lgbm, dtrain, n_rounds, valid_sets = [dtrain, dvalid],feval=self.lgbm_RMSPE,)
                    #   verbose_eval= 250,
                    #   early_stopping_rounds=500,
                      #callbacks=[lgb.early_stopping(stopping_rounds=1), lgb.log_evaluation(period=1)])
            models.append(clf)


            p = clf.predict(X_valid)*v1v
            val_score =  np.mean( ((p-y_val)/y_val)**2 )**0.5
            
            # full_score += y_val.shape[0]*score**2
            
            
            print('SCORE:', val_score)
            print(self.nancorr(       p/v1v ,        y_val/v1v ))
            print(self.nancorr(np.log(p/v1v), np.log(y_val/v1v)))

            print(self.nancorr(p, y_val))
            print(self.nancorr(np.log(p), np.log(y_val)))


            test_pred = clf.predict(self.test_df[self.feat_cols_list] )*v1ts
            test_y_preds += test_pred/self.n_folds

            print(f'split: {fold_n}, val rmspe score is {val_score}')
            rmspe_val_score.append(val_score)

            #del X_train, X_valid, y_train, y_valid,train_df,val_df,dtrain,dvalid, v1tr, v1v, v1ts  


        importances = pd.DataFrame({'Feature': clf.feature_name(), 
                                    'Importance': sum( [model.feature_importance(importance_type='gain') for model in models] )})
    
        importances2 = importances.nlargest(40,'Importance', keep='first').sort_values(by='Importance', ascending=True)
        importances2[['Importance', 'Feature']].plot(kind = 'barh', x = 'Feature', figsize = (8,6), color = 'blue', fontsize=11);plt.ylabel('Feature', fontsize=12)
           
            
        print(f'rmspe score over {self.n_folds} splits is',np.mean(rmspe_val_score))
        
        del X_train, X_valid, y_train, y_valid,train_df,val_df,dtrain,dvalid, v1tr, v1v, v1ts
        gc.collect()
        return np.mean(rmspe_val_score)




    def evaluate_predictions(self,model):

        return



    def make_predictions(self,best_params,num_round, e_s_r):
    
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





################ OPTUNA objective ####################

def objective(trial):
 
    #os.chdir("c:\Work\WORK_PACKAGE\Demand_forecasting\BLUESG_Demand_data\Data-preprocessing_data_generation")
    t_v_t = train_validate_n_test(df)

    ######  SET Hyperparameter's range for tuning ######
    n_rounds = 5
    seed1=11
    # Hyperparameters and algorithm parameters are described here

    params_lgbm = {
            'learning_rate':trial.suggest_float(name='learning_rate', low=0.01, high=0.1,log=True),        
            'lambda_l1': trial.suggest_int('lambda_l1', 2, 3),
            'lambda_l2': trial.suggest_int('lambda_l2', 5, 6),
            'num_leaves': trial.suggest_int(name='lambda_l2', low=80, high=83,step=3),
            'min_sum_hessian_in_leaf':  trial.suggest_int(name='lambda_l2', low=20, high=23,step=3),
            'feature_fraction': round(trial.suggest_float(name='subsample', low=0.8, high=0.9,step=0.1),1),
            'feature_fraction_bynode': round(trial.suggest_float(name='subsample', low=0.8, high=0.80,step=0.1),1),
            'bagging_fraction': round(trial.suggest_float(name='subsample', low=0.8, high=0.8,step=0.1),1),
            'bagging_freq': trial.suggest_int(name='lambda_l2', low=42, high=45,step=3),
            'min_data_in_leaf': trial.suggest_int(name='lambda_l2', low=25, high=28,step=3),
            'max_depth': trial.suggest_int('lambda_l2', 6, 7),
            'objective': 'regression',
            'metric': 'None',
            'device':'gpu',
            'seed': seed1,
            'feature_fraction_seed': seed1,
            'bagging_seed': seed1,
            'drop_seed': seed1,
            'data_random_seed': seed1,
            'boosting': 'gbdt',
            'verbosity': -1,
            'n_jobs':-1,
    }


    ######  SET Hyperparameter's range for tuning ######

    val_avg_error = t_v_t.lgbm_train_validate(params_lgbm,n_rounds)
    del t_v_t
    return val_avg_error




################ MAIN ####################

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, timeout=12000, n_trials=500)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

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

    # best_tweedie_variance_power = study.best_params["tweedie_variance_power"]
    # best_params = {"max_depth": study.best_params["max_depth"],
    #         "eta": study.best_params["eta"],
    #         "subsample" : study.best_params["subsample"],
    #         "colsample_bytree": study.best_params["colsample_bytree"],
    #         'eval_metric':'tweedie-nloglik@'+str(best_tweedie_variance_power), ## try using AUC as well.. 
    #         'tweedie_variance_power': best_tweedie_variance_power,
    #         'gamma': study.best_params["gamma"],
    #         'reg_alpha': study.best_params["reg_alpha"], 
    #         'reg_lambda': study.best_params["reg_lambda"],
    #         'min_child_weight': study.best_params["min_child_weight"],
    #         "objective": 'reg:tweedie',
    #         }
    # early_stopping_rounds = 30
    # eval_metric = 'tweedie-nloglik@'+str(best_tweedie_variance_power)
    # num_round= 1000

    # t_v_t = train_validate_n_test()
    # best_model = t_v_t.make_predictions(best_params,num_round, early_stopping_rounds)
    # t_v_t.evaluate_predictions(best_model)




