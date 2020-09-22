from pathlib import Path
import numpy as np
import pandas as pd
from rdkit.Chem import PandasTools
from utils import *


import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz 2.44.1/bin/'

pd.set_option('display.width', 500, 'display.max_rows', None, 'display.max_columns', None)
data_dir_raw = Path('../../data')


# -----------------------------FEATURE ENGINEERING 0------------------------------------

df_stliq_clean = pd.read_csv(data_dir_raw / 'df_stliq_clean.csv', index_col=0)

# adding column hosting RDKit molecule object from smiles
PandasTools.AddMoleculeColumnToFrame(df_stliq_clean, 'smiles', 'rdkmol', includeFingerprints=True)

# feature engineering + exporting data
df_fe0 = construct_features(df_stliq_clean['rdkmol'], count_frags_0)
df_fe0.set_index(df_stliq_clean.index, inplace=True) # recovering original index

# appending measured_st and molecule name from df_stliq_clean
df_fe0['measured_st'] = df_stliq_clean['measured_st']
df_fe0['molecule'] = df_stliq_clean['molecule']
df_fe0['density'] = df_stliq_clean['density']

df_fe0 = df_fe0[~df_fe0['molecule'].str.contains('anhydride|carbonate|Formic acid|Furan')]
df_fe0.to_csv(data_dir_raw / 'df_fe0.csv')
# --------------------------------------------------------------------------------------


# -----------------------------FEATURE ENGINEERING 0------------------------------------
# -------------------------------------POLYMER------------------------------------------
# TODO
# --------------------------------------------------------------------------------------


# -----------------------------FEATURE ENGINEERING 0------------------------------------
# -------------------------------------DIOLS--------------------------------------------

df_diols = pd.read_csv(data_dir_raw / 'diols_ref.csv', index_col=0)

# adding column hosting RDKit molecule object from smiles
PandasTools.AddMoleculeColumnToFrame(df_diols, 'smiles', 'rdkmol', includeFingerprints=True)

# adding features from feature engineering 0
df_diols_fe0 = construct_features(df_diols['rdkmol'], count_frags_0)
df_diols_fe0.set_index(df_diols.index, inplace=True) # recovering original index

df_diols_fe0.to_csv(data_dir_raw / 'df_diols_fe0.csv')
# --------------------------------------------------------------------------------------


# -----------------------------FEATURE ENGINEERING 0------------------------------------
# -------------------------------------ACIDS--------------------------------------------

df_acids = pd.read_csv(data_dir_raw / 'acids_ref.csv', index_col=0)

# adding column hosting RDKit molecule object from smiles
PandasTools.AddMoleculeColumnToFrame(df_acids, 'smiles', 'rdkmol', includeFingerprints=True)

# adding features from feature engineering 0
df_acids_fe0 = construct_features(df_acids['rdkmol'], count_frags_0)
df_acids_fe0.set_index(df_acids.index, inplace=True) # recovering original index

df_acids_fe0.to_csv(data_dir_raw / 'df_acids_fe0.csv')
# --------------------------------------------------------------------------------------


# -----------------------------FEATURE ENGINEERING 0------------------------------------
# -------------------------------------TEST--------------------------------------------

df_test = pd.read_csv(data_dir_raw / 'polyesters.csv')
df_diols_fe0 = pd.read_csv(data_dir_raw / 'df_diols_fe0.csv', index_col=0)
df_acids_fe0 = pd.read_csv(data_dir_raw / 'df_acids_fe0.csv', index_col=0)

# correcting 'H', O-alc', 'O-acid' and 'O-ester' by subtracting atom/functional group loss from condensation
# TODO: make a function for this procedure
df_diols_fe0['H'] -= 2
df_acids_fe0['H'] -= 2
df_diols_fe0['O-alc'] -= 2
df_acids_fe0['O-acid'] -= 4
df_acids_fe0['O-ester'] += 4

df_monomers_fe0 = pd.concat([df_diols_fe0, df_acids_fe0])
monomers = df_test.columns[df_test.columns.str.contains('diol|acid')]
features = df_monomers_fe0.columns

df_test_fe0 = avg_monomer_features(df_test, df_monomers_fe0, monomers, features)

# appending mn, tg, measured_st, st_polar, st_disperse, commercial
to_append = ['mn', 'tg', 'measured_st', 'st_polar', 'st_disperse', 'commercial']
for col in to_append:
    df_test_fe0[col] = df_test[col]

df_test_fe0.to_csv(data_dir_raw / 'df_test_fe0.csv')
# --------------------------------------------------------------------------------------