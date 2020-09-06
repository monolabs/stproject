from pathlib import Path
import numpy as np
import pandas as pd
from rdkit.Chem import PandasTools
from utils import *


import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz 2.44.1/bin/'

pd.set_option('display.width', 500, 'display.max_rows', None, 'display.max_columns', None)
data_dir_raw = Path('../../data')


# -----------------------------FEATURE ENGINEERING 1------------------------------------

df_stliq_clean = pd.read_csv(data_dir_raw / 'df_stliq_clean.csv', index_col=0)

# adding column hosting RDKit molecule object from smiles
PandasTools.AddMoleculeColumnToFrame(df_stliq_clean, 'smiles', 'rdkmol', includeFingerprints=True)

# feature engineering + exporting data
df_fe1 = construct_features(df_stliq_clean['rdkmol'], count_frags_1)
df_fe1.set_index(df_stliq_clean.index, inplace=True) # recovering original index

# appending measured_st, density and molecule name from df_stliq_clean
df_fe1['measured_st'] = df_stliq_clean['measured_st']
df_fe1['molecule'] = df_stliq_clean['molecule']
df_fe1['density'] = df_stliq_clean['density']

df_fe1 = df_fe1[~df_fe1['molecule'].str.contains('anhydride|carbonate|Formic acid|Furan')]
df_fe1.to_csv(data_dir_raw / 'df_fe1.csv')
# --------------------------------------------------------------------------------------


# -----------------------------FEATURE ENGINEERING 1------------------------------------
# -------------------------------------DIOLS--------------------------------------------

df_diols = pd.read_csv(data_dir_raw / 'diols_ref.csv', index_col=0)

# adding column hosting RDKit molecule object from smiles
PandasTools.AddMoleculeColumnToFrame(df_diols, 'smiles', 'rdkmol', includeFingerprints=True)

# adding features from feature engineering 0
df_diols_fe1 = construct_features(df_diols['rdkmol'], count_frags_1)
df_diols_fe1.set_index(df_diols.index, inplace=True) # recovering original index

df_diols_fe1.to_csv(data_dir_raw / 'df_diols_fe1.csv')
# --------------------------------------------------------------------------------------


# -----------------------------FEATURE ENGINEERING 0------------------------------------
# -------------------------------------ACIDS--------------------------------------------

df_acids = pd.read_csv(data_dir_raw / 'acids_ref.csv', index_col=0)

# adding column hosting RDKit molecule object from smiles
PandasTools.AddMoleculeColumnToFrame(df_acids, 'smiles', 'rdkmol', includeFingerprints=True)

# adding features from feature engineering 0
df_acids_fe1 = construct_features(df_acids['rdkmol'], count_frags_1)
df_acids_fe1.set_index(df_acids.index, inplace=True) # recovering original index

df_acids_fe1.to_csv(data_dir_raw / 'df_acids_fe1.csv')
# --------------------------------------------------------------------------------------