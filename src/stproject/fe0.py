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

df_fe0 = df_fe0[~df_fe0['molecule'].str.contains('anhydride|carbonate|Formic acid')]
df_fe0.to_csv(data_dir_raw / 'df_fe0.csv')
# --------------------------------------------------------------------------------------