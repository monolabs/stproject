import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from sklearn.model_selection import GridSearchCV
import re
from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt
import seaborn as sns
import matplotlib.pyplot as plt


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


def plot_results(y, y_hat, name_series, num_outliers, title, x_label, y_label, with_line=True):
    # this function searches for the highest value of error metric and indicate the datapoints in a scatterplot
    # number of marked data points is specified with num_outliers
    # y_true, y_pred and name_series must be pandas series (with indices)
    # name_series is a series containing names used to make sense of analysis

    # points with highest error (in descending order)
    i_out = abs(y - y_hat).sort_values(ascending=False).index[:num_outliers]
    y_out = y[i_out]
    y_hat_out = y_hat[i_out]
    names_out = name_series[i_out]
    print(pd.DataFrame({'name': names_out,
                        'y': y_out,
                        'y_hat': y_hat_out}))

    # plotting data points
    min_val,  max_val = np.min(y.append(y_hat)), np.max(y.append(y_hat))
    sns.scatterplot(y_hat, y)
    sns.scatterplot(y_hat_out, y_out)
    if with_line:
        sns.lineplot(x=[min_val, max_val], y=[min_val, max_val], alpha=0.2)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title, {'fontsize': 12, 'weight': 'bold'})




