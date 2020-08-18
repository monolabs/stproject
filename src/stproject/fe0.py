from pathlib import Path
import numpy as np
import pandas as pd
from rdkit.Chem import PandasTools
from utils import *
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBRegressor
from xgboost import plot_tree
import config
import matplotlib.pyplot as plt

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz 2.44.1/bin/'

pd.set_option('display.width', 500, 'display.max_rows', None, 'display.max_columns', None)
data_dir_raw = Path('../../data')


# -----------------------------FEATURE ENGINEERING 0------------------------------------

df_stliq_clean = pd.read_csv(data_dir_raw / 'df_stliq_clean.csv', index_col=0)

# adding column hosting RDKit molecule object from smiles
PandasTools.AddMoleculeColumnToFrame(df_stliq_clean, 'smiles', 'rdkmol', includeFingerprints=True)

# generating X and Y
features = ['C', 'H', 'O', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'C=C', 'C#C', 'Ar',
            'OH', 'CHO', 'CO', 'COOH', 'COOR']
features_unused = ['R3', 'R4', 'R7', 'R8', 'C#C']
features_used = diff(features, features_unused)

X_raw = np.stack(df_stliq_clean['rdkmol'].apply(count_frags_0))
raw_index = np.array(df_stliq_clean.index)

# column index of unused features in X
col_exclude = [i for (i, ele) in enumerate(features) if ele in features_unused]
col_include = diff(range(len(features)), col_exclude)

# filtering data points with all zero values of feature_unused
row_include = np.where(np.sum(X_raw[:, col_exclude], axis=1) == 0)
X = X_raw[row_include]
X = X[:, col_include]
index = raw_index[row_include]

# X = np.stack(df_stliq_clean['fe0'])
Y_raw = np.array(df_stliq_clean['measured_st'])
Y = Y_raw[row_include]

# constructing new dataframe with preserved index storing features and Y (surface tensions)
df_fe0 = pd.DataFrame(X, index=index, columns=features_used)
df_fe0['measured_st'] = Y
print(df_fe0)

# --------------------------------------------------------------------------------------


# -------------------------------LINEAR MODEL-------------------------------------

lm = LinearRegression(fit_intercept=True)
lm.fit(X,Y)

print(regression_results(Y, lm.predict(X)))

# --------------------------------------------------------------------------------

# --------------------------------XGBoost-----------------------------------------

train, test = train_test_split(df_fe0, test_size=0.1, random_state=42)

xgb1 = XGBRegressor(colsample_bytree=0.5,
                    gamma=0,
                    learning_rate=0.1,
                    max_depth=6,
                    min_child_weight=0,
                    n_estimators=10000,
                    reg_alpha=0,
                    reg_lambda=0,
                    subsample=0.8,
                    seed=42,
                    objective='reg:squarederror')

# eval_set = [(train[features_used], train['measured_st']), (test[features_used], test['measured_st'])]
#
xgb1.fit(train[features_used],
         train['measured_st'])
plot_tree(xgb1)
# feat_imp = pd.Series(xgb1.get_booster().get_fscore()).sort_values(ascending=False)
# feat_imp.plot(kind='bar', title='Feature Importances')
# plt.ylabel('Feature Importance Score')
plt.show()
# --------------------------------------------------------------------------------