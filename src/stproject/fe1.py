from pathlib import Path
import numpy as np
import pandas as pd
from tabula import read_pdf
from rdkit import Chem
from rdkit import RDConfig
import utils
import config
import re

pd.set_option('display.width', 500, 'display.max_rows', None, 'display.max_columns', None)
data_dir_raw = Path('../../data')

"""
-----------------------------FEATURE ENGINEERING 1------------------------------------
"""
df_stliq_clean = pd.read_csv(data_dir_raw / 'df_stliq_clean.csv')

# adding column hosting RDKit molecule object from smiles
Chem.PandasTools.AddMoleculeColumnToFrame(df_stliq_clean, 'smiles', 'rdkmol', includeFingerprints=True)

# generating canonical smiles
df_stliq_clean['csmiles'] = Chem.MolToSmiles(x, True)

# adding features

c_regex = re.compile('C\(C.*\)\(C')
ch_regex = re.compile('C\(C.*\)(?!\()')
ch2_regex = re.compile('C')



print(df_stliq_clean)
"""
--------------------------------------------------------------------------------------
"""