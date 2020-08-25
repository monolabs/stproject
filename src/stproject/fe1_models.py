from pathlib import Path
import numpy as np
import pandas as pd
from rdkit.Chem import PandasTools
import matplotlib.pyplot as plt
from utils import *

from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from xgboost.sklearn import XGBRegressor
import xgboost as xgb
from xgboost import plot_tree


import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz 2.44.1/bin/'

pd.set_option('display.width', 500, 'display.max_rows', None, 'display.max_columns', None)
data_dir_raw = Path('../../data')


# Generate dtrain
df_fe1 = pd.read_csv(data_dir_raw / 'df_fe1.csv', index_col=0)

features_unused = ['C#C', 'C-ring']
df_fe1_train = df_fe1.loc[np.sum(df_fe1[features_unused], axis=1) == 0, df_fe1.columns.difference(features_unused)]
dtrain = xgb.DMatrix(df_fe1_train.drop(['measured_st', 'molecule'], axis=1), label=df_fe1_train['measured_st'])

# initial xgb.cv run

params_0 = {'learning_rate': 0.01,
            'n_estimators': 10000,
            'max_depth': 4,
            'min_child_weight': 2,
            'gamma': 0,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'nthread': 4,
            'seed': 42,
            'objective': 'reg:squarederror'}
xgb_ini_results = xgb.cv(params=params_0,
                         dtrain=dtrain,
                         num_boost_round=5000,
                         nfold=5,
                         metrics='rmse',
                         early_stopping_rounds=50,
                         as_pandas=True,
                         seed=42)
xgb_ini_results.to_csv(data_dir_raw / 'xgb_fe1_ini_results.csv')

