from pathlib import Path
import numpy as np
import pandas as pd
from rdkit.Chem import PandasTools, Fragments
from utils import *
import config
import re

pd.set_option('display.width', 500, 'display.max_rows', None, 'display.max_columns', None)
data_dir_raw = Path('../../data')

"""
-----------------------------FEATURE ENGINEERING 1------------------------------------
"""
df_stliq_clean = pd.read_csv(data_dir_raw / 'df_stliq_clean.csv', index_col=0)

# adding column hosting RDKit molecule object from smiles
PandasTools.AddMoleculeColumnToFrame(df_stliq_clean, 'smiles', 'rdkmol', includeFingerprints=True)

# generating X and Y
X = df_stliq_clean['rdkmol'].apply(countfrags1)
print(X)

"""
--------------------------------------------------------------------------------------
"""