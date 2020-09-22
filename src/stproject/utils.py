import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold

import xgboost as xgb

from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt

import seaborn as sns
import matplotlib.pyplot as plt

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


# ================================FEATURE ENGINEERING=======================================

def diff(first, second):
    second = set(second)
    return [item for item in first if item not in second]

def count_rings(mol):
    # return a dictionary of counts of 3-membered (R3) to 8-membered (R8) rings including aromatics

    rings_dict = dict.fromkeys(['R3', 'R4', 'R5', 'R6', 'R7', 'R8'], 0)
    rings_idx_list = mol.GetRingInfo().AtomRings()

    for ring_idx in rings_idx_list:
        rings_dict['R' + str(len(ring_idx))] += 1

    return rings_dict


def count_frags_0(mol):

    frags = dict.fromkeys(['C', 'H', 'C=C', 'C#C', 'Ar', 'O-alc', 'O-eth',
                           'O-ald', 'O-ket', 'O-acid', 'O-ester',
                           'R3', 'R4', 'R5', 'R6', 'R7', 'R8'], 0)
    smiles = Chem.MolToSmiles(mol)
    M = MolWt(mol)

    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'C':
            frags['C'] += 1
        frags['H'] += atom.GetTotalNumHs()

    for bond in mol.GetBonds():
        if str(bond.GetBondType()) == 'DOUBLE':
            frags['C=C'] += 1
        if str(bond.GetBondType()) == 'TRIPLE':
            frags['C#C'] += 1

    ring_dict = count_rings(mol)
    frags.update(ring_dict)

    frags['Ar'] = Chem.Fragments.fr_benzene(mol)
    frags['O-alc'] = Chem.Fragments.fr_Ar_OH(mol) + Chem.Fragments.fr_Al_OH(mol)
    frags['O-eth'] = Chem.Fragments.fr_ether(mol)
    frags['O-ald'] = Chem.Fragments.fr_aldehyde(mol)
    frags['O-ket'] = Chem.Fragments.fr_ketone(mol)
    frags['O-acid'] = Chem.Fragments.fr_COO(mol)
    frags['O-ester'] = Chem.Fragments.fr_ester(mol)

    # catching formate motif (undetected by RDKit 2020.03.2.0)
    frags['O-ester'] += Chem.Fragments.fr_C_O(mol) - frags['O-ald'] - frags['O-ket'] - frags['O-acid'] - frags['O-ester']

    # correcting C=C by subtracting aldehyde, ketone, acid and ester
    frags['C=C'] = frags['C=C'] - frags['O-ald'] - frags['O-ket'] - frags['O-acid'] - frags['O-ester']

    # correcting O-acid and O-ester to double O
    frags['O-acid'] *= 2
    frags['O-ester'] *= 2

    # correcting O-ether by subtracting ester
    frags['O-eth'] -= frags['O-ester']/2

    # Excluding benzene ring from R6
    frags['R6'] -= frags['Ar']

    frags['M'] = MolWt(mol)

    return frags

def count_frags_1(mol):
    # return a list of fragment counts from rdkit's 'mol' object
    # fragments: [C, CH, CH2, CH3, C-ring, CH-ring, CH2-ring, C=C, C#C, Ar, OH, C-O-C, CHO, CO, COOR, COOH]
    # CAUTION: carbonate and anhydride will be incorrectly described

    frags = dict.fromkeys(['C', 'CH', 'CH2', 'CH3', 'C-ring', 'CH-ring', 'CH2-ring', 'C=C', 'C#C', 'Ar',
                           'OH', 'C-O-C', 'CHO', 'CO', 'COOR', 'M'], 0)
    smiles = Chem.MolToSmiles(mol)

    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'C' and atom.GetTotalDegree() == 4: # aromatic atoms not included
            if not atom.IsInRing():
                if atom.GetTotalNumHs() == 0:
                    frags['C'] += 1
                elif atom.GetTotalNumHs() == 1:
                    frags['CH'] += 1
                elif atom.GetTotalNumHs() == 2:
                    frags['CH2'] += 1
                elif atom.GetTotalNumHs() == 3:
                    frags['CH3'] += 1
            else:
                if atom.GetTotalNumHs() == 0:
                    frags['C-ring'] += 1
                elif atom.GetTotalNumHs() == 1:
                    frags['CH-ring'] += 1
                elif atom.GetTotalNumHs() == 2:
                    frags['CH2-ring'] += 1

    for bond in mol.GetBonds():
        if str(bond.GetBondType()) == 'DOUBLE':
            frags['C=C'] += 1
        if str(bond.GetBondType()) == 'TRIPLE':
            frags['C#C'] += 1

    frags['Ar'] = Chem.Fragments.fr_benzene(mol)
    frags['OH'] = Chem.Fragments.fr_Ar_OH(mol) + Chem.Fragments.fr_Al_OH(mol)
    frags['C-O-C'] = Chem.Fragments.fr_ether(mol)
    frags['CHO'] = Chem.Fragments.fr_aldehyde(mol)
    frags['CO'] = Chem.Fragments.fr_ketone(mol)
    frags['COOH'] = Chem.Fragments.fr_COO(mol) # does not detect formic acid
    frags['COOR'] = Chem.Fragments.fr_ester(mol) # does not detect formic acid ester

    # catching formate motif (undetected by RDKit 2020.03.2.0)
    frags['COOR'] += Chem.Fragments.fr_C_O(mol) - frags['CHO'] - frags['CO'] - frags['COOH'] - frags['COOR']

    # correcting ether by substracting ester (ester counts towards ether in RDKit 2020.03.2.0
    frags['C-O-C'] -= frags['COOR']

    # correcting C=C by subtracting aldehyde, ketone, acid and ester
    frags['C=C'] = frags['C=C'] - frags['CHO'] - frags['CO'] - frags['COOH'] - frags['COOR']

    frags['M'] = MolWt(mol)

    return frags


def construct_features(series, func):
    # USED ON LIQUID SURFACE TENSION DATASET (single molecule)
    features_series = series.apply(func)
    df = pd.DataFrame([features_series.iloc[0]])
    for ele in features_series.iloc[1:]:
        df = df.append(ele, ignore_index=True)
    return df

def construct_avg_features (mol_series, molfrac_series, func):
    # USED ON POLYMER MOLAR VOLUME DATASET
    # each element in series is a list containing RDKit mol objects
    # returns pandas dataframe of computed features (weighed by molar fraction) based on func

    df = pd.DataFrame()
    for i in range(len(mol_series)):
        dummy_df = pd.DataFrame()
        for j in range(len(mol_series[i])):
            dummy_df = pd.concat([dummy_df, pd.Series(func(mol_series.iloc[i][j])).to_frame().T * molfrac_series.iloc[i][j]],
                                 ignore_index=True)
        df = pd.concat([df, dummy_df.sum().to_frame().T])
    df.index = mol_series.index

    return df


def avg_monomer_features(df, ref, monomers, features):
    # USED ON POLYMER DATASET TO TEST
    # computes averaged features from ref, weighed by molar ratio in df
    # df = dataframe to be updated (row = polymer samples, columns include monomers molar fraction)
    # ref = reference dataframe (row = monomers, column = features)
    # monomers = monomers to compute the features of (list)
    # features = features to update (list)
    # returns

    # to contain the entire engineered features for all observations
    dummy_arr = np.zeros((len(df), len(features)))

    for i, ind in enumerate(df.index):
        # to contain the total features per observation
        for monomer in monomers:
            dummy_arr[i] += df.loc[ind, monomer] * ref.loc[monomer]

    return pd.DataFrame(dummy_arr, columns=features)

# ================================END OF FEATURE ENGINEERING=================================



# ======================================LINEAR MODEL=========================================

def regression_results(y_true, y_pred):

    # Regression metrics
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred)
    mse=metrics.mean_squared_error(y_true, y_pred)
    mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
    r2=metrics.r2_score(y_true, y_pred)

    print('explained_variance: ', round(explained_variance,4))
    print('mean_squared_log_error: ', round(mean_squared_log_error,4))
    print('r2: ', round(r2,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))


def plot_results(y, y_hat, i_out, title=None, x_label=None, y_label=None, with_line=True):
    # plots data points with specified number of outliers and indices of outliers
    # y and y_hat are pandas dataframe with index

    min_val, max_val = np.min(y.append(y_hat)), np.max(y.append(y_hat))
    sns.scatterplot(y_hat, y)
    sns.scatterplot(y_hat[i_out], y[i_out])
    if with_line:
        sns.lineplot(x=[min_val, max_val], y=[min_val, max_val], alpha=0.2)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title, {'fontsize': 12, 'weight': 'bold'})


def run_linear_regression(df, regressors, features_excluded, target,
                          fit_intercept=True,
                          num_outliers=0,
                          title=None,
                          x_label=None,
                          y_label=None,
                          with_line=True):
    # df is pandas dataframe
    # regressors is a list (or str) of column name to be used as regressors.
    # num_outlier is an integer, denoting number of highest-error data points to be highlighted
    # with_line + True plots y=x line
    # This function automatically exclude data points having features which are in features_excluded list

    df = df[df[features_excluded].sum(axis=1) == 0]
    X, y = df[regressors], df[target]
    lm = LinearRegression(fit_intercept=fit_intercept)
    lm.fit(X, y)
    y_hat = pd.Series(lm.predict(X), index=y.index)

    # the following section searches for the highest value of error metric and indicate the datapoints in a scatterplot
    # number of marked data points is specified with num_outliers
    # y and y_hat must be pandas series (with indices)
    # points with highest error (in descending order)

    i_out = abs(y - y_hat).sort_values(ascending=False).index[:num_outliers]
    # plotting results
    plot_results(y, y_hat, i_out, title=title, x_label=x_label, y_label=y_label, with_line=with_line)

    # evaluation
    rmse = np.sqrt(mean_squared_error(y, y_hat))
    mae = mean_absolute_error(y, y_hat)

    return lm, rmse, mae, i_out


def run_linear_regression_parachor(df, regressors, features_excluded, st_col, mv_col,
                                   n=4,
                                   run_on_st=True,
                                   fit_intercept=True,
                                   num_outliers=0,
                                   title=None,
                                   x_label=None,
                                   y_label=None,
                                   with_line=True):
    # df is pandas dataframe
    # regressors is a list of column name to be used as regressors.
    # if run_on_st = True, results are calculated on surface tension level. Otherwise on
    # num_outlier is an integer, denoting number of highest-error data points to be highlighted
    # with_line + True plots y=x line
    # This function automatically exclude data points having features which are in features_excluded list

    df = df[df[features_excluded].sum(axis=1) == 0]
    X, y_untrans = df[regressors], df[st_col]
    y_trans = y_untrans**(1/n) * df[mv_col]

    lm = LinearRegression(fit_intercept=fit_intercept)
    lm.fit(X, y_trans)
    y_trans_hat = pd.Series(lm.predict(X), index=y_trans.index)

    if run_on_st:
        y_hat = (y_trans_hat/df[mv_col]) ** n
        y = y_untrans
    else:
        y_hat = y_trans_hat
        y = y_trans

    # the following section searches for the highest value of error metric and indicate the datapoints in a scatterplot
    # number of marked data points is specified with num_outliers
    # y and y_hat must be pandas series (with indices)
    # points with highest error (in descending order)

    i_out = abs(y - y_hat).sort_values(ascending=False).index[:num_outliers]
    # plotting results
    plot_results(y, y_hat, i_out, title=title, x_label=x_label, y_label=y_label, with_line=with_line)

    # evaluation
    rmse = np.sqrt(mean_squared_error(y, y_hat))
    mae = mean_absolute_error(y, y_hat)

    return lm, rmse, mae, i_out

def run_linear_regression_scaled(df,
                                    regressors,
                                    features_excluded,
                                    target,
                                    scaler,
                                    fit_intercept=True,
                                    num_outliers=0,
                                    title=None,
                                    x_label=None,
                                    y_label=None,
                                    with_line=True):
    # df is pandas dataframe
    # regressors is a list (or str) of column name to be used as regressors.
    # num_outlier is an integer, denoting number of highest-error data points to be highlighted
    # with_line + True plots y=x line
    # This function automatically exclude data points having features which are in features_excluded list
    # regressors will be divided by scaler to make new features used in regression

    df = df[df[features_excluded].sum(axis=1) == 0]
    X, y = df[regressors].divide(df[scaler], axis=0), df[target]
    lm = LinearRegression(fit_intercept=fit_intercept)
    lm.fit(X, y)
    y_hat = pd.Series(lm.predict(X), index=y.index)

    # the following section searches for the highest value of error metric and indicate the datapoints in a scatterplot
    # number of marked data points is specified with num_outliers
    # y and y_hat must be pandas series (with indices)
    # points with highest error (in descending order)

    i_out = abs(y - y_hat).sort_values(ascending=False).index[:num_outliers]
    # plotting results
    plot_results(y, y_hat, i_out, title=title, x_label=x_label, y_label=y_label, with_line=with_line)

    # evaluation
    rmse = np.sqrt(mean_squared_error(y, y_hat))
    mae = mean_absolute_error(y, y_hat)

    return lm, rmse, mae, i_out

def evaluate_model(model, X, y,
                   num_outliers=0,
                   title=None,
                   x_label=None,
                   y_label=None,
                   with_line=True):
    # predict new X and plot results with specified number of outliers

    y_hat = pd.Series(model.predict(X), index=y.index)

    # the following section searches for the highest value of error metric and indicate the datapoints in a scatterplot
    # number of marked data points is specified with num_outliers
    # y and y_hat must be pandas series (with indices)
    # points with highest error (in descending order)

    i_out = abs(y - y_hat).sort_values(ascending=False).index[:num_outliers]
    # plotting results
    plot_results(y, y_hat, i_out, title=title, x_label=x_label, y_label=y_label, with_line=with_line)

    # evaluation
    rmse = np.sqrt(mean_squared_error(y, y_hat))
    mae = mean_absolute_error(y, y_hat)

    return rmse, mae, i_out

# ====================================END OF LINEAR MODEL========================================


# ===========================================XGBOOST==============================================

def optimize_xgb(trials, score_func, space, algo=tpe.suggest, max_evals=250):

    # this function optimizes xgb parameters
    # space = defined feature space to be optimized
    # trials = hyperopt trials object
    # score_xgb = score function
    # algo = algorithm use to suggest next point
    # prints best parameters

    best = fmin(score_func, space, algo=algo, trials=trials, max_evals=max_evals)
    print(best)

def run_xgb(X, y,
            space,
            k=5,
            max_evals=250,
            random_state=42):
    '''
     1. randomly partitions the data into train and validation set (p = portion going to test)
     2. run hyperopt in sets of params with max iteration = max_evals
     3. return list of models trained on the entire train data

     X and y = full train set (entire data - test set)
     space = space of parameters for hyperopt optimization. Template:
         space = {
                 'n_estimators' : hp.quniform('n_estimators', 100, 1000, 1),
                 'eta' : hp.quniform('eta', 0.01, 0.10, 0.02),
                 'max_depth' : hp.choice('max_depth', np.arange(1, 13, 1)),
                 'min_child_weight' : hp.quniform('min_child_weight', 1, 6, 1),
                 'subsample' : hp.quniform('subsample', 0.5, 1, 0.05),
                 'gamma' : hp.quniform('gamma', 0.5, 1, 0.05),
                 'colsample_bytree' : hp.quniform('colsample_bytree', 0.5, 1, 0.05),
                 'eval_metric': 'rmse',
                 'objective': 'reg:squarederror',
                 'nthread' : 6,
                 'silent' : 1,
                 'seed': 42
                 }
    '''

    # Defining score function
    def score_xgb(params):
        # must take only params (space to be optimized) as argument
        # train xgb model, evaluate, returns the score, status and model

        print(f"iteration: {kfold_iteration}. Training with params:")
        print(params)
        num_round = int(params['n_estimators'])
        del params['n_estimators']
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_valid, label=y_valid)
        # watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
        model = xgb.train(params, dtrain, num_round)
        predictions = model.predict(dvalid)
        score = np.sqrt(mean_squared_error(y_valid, predictions))
        print("\tScore {0}\n\n".format(score))
        return {'loss': score, 'status': STATUS_OK, 'trained_model': model}

    # list to capture k best models
    models = []
    params = []

    # splitting into k-folds
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    kf.get_n_splits(X)

    # Looping through k splits
    kfold_iteration=1
    for train_index, valid_index in kf.split(X):
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        # performing hyperopt optimization
        # the function score_func takes X_train, X_valid, y_train and y_valid to train xgbmodel
        trials = Trials()
        best = fmin(score_xgb, space, algo=tpe.suggest, trials=trials, max_evals=max_evals)
        params.append(best)

        # training and recording best model
        best_n_estimators = int(best['n_estimators'])
        del best['n_estimators']
        dtrain_full = xgb.DMatrix(X, label=y)
        models.append(xgb.train(best, dtrain_full, best_n_estimators))

        kfold_iteration += 1

    return models, params


def evaluate_xgb_models(models,
                        X,
                        y,
                        weights=None,
                        num_outliers=0,
                        title=None,
                        x_label=None,
                        y_label=None,
                        with_line=True):
    # returns weighted predictions of several models
    # weights = list of weights (length = length of models)
    # y is pandas series

    predictions = np.zeros((len(X), len(models)))
    if weights == None:
        weights = [1/len(models)] * len(models)
    weights = np.array(weights).reshape((1, -1)) # into 1xlen(models) numpy array
    for i in range(len(models)):
        predictions[:, i] = models[i].predict(xgb.DMatrix(X))
    y_hat = pd.Series(np.sum(predictions * weights, axis=1), index=X.index)

    i_out = abs(y - y_hat).sort_values(ascending=False).index[:num_outliers]

    plot_results(y, y_hat, i_out, title=title, x_label=x_label, y_label=y_label, with_line=with_line)

    # evaluation
    rmse = np.sqrt(mean_squared_error(y, y_hat))
    mae = mean_absolute_error(y, y_hat)

    return rmse, mae, i_out


# ========================================END OF XGBOOST=========================================


# =======================================LOADING DATA========================================

def load_data_fe0(file_path, features_excluded, target):
    # function automatically exclude datapoints with non-zero entries in features_excluded
    # e.g. if aldehyde 'CHO' is excluded, any substance identified as aldehyde will be removed

    df = pd.read_csv(file_path, index_col=0)
    df = df[df[features_excluded].sum(axis=1) == 0]
    X = df[df.columns.difference(features_excluded + [target])]
    y = df[target]
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_valid, y_train, y_valid

# ==================================END OF LOADING DATA========================================
