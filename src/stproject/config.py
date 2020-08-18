from pathlib import Path
import numpy as np
import pandas as pd
from tabula import read_pdf

from rdkit import RDConfig
import utils
import re

pd.set_option('display.width', 500, 'display.max_rows', None, 'display.max_columns', None)
data_dir_raw = Path('../../data')

# example
# df = pd.read_csv(data_dir_raw / 'example.csv')

"""
--------------------------------DATA PREPARATION------------------------------------

Extracts and cleans full dataframe of surface tensions of liquid
rows = chemicals
raw data columns = No, Molecule, SMILES notation, Measured ST, MF, CAS RN
Index from dataframe before exclusion of silicon-containing sample are preserved
"""

# df_stliq_import = read_pdf(data_dir_raw / 'st_liq.pdf', pages='3-11')
# df_stliq = pd.DataFrame()
#
# for df in df_stliq_import[0: len(df_stliq_import)]:
#     df_stliq = pd.concat([df_stliq, df[~df.index.isin([0, 1])]], axis=0)
#
# df_stliq.reset_index(drop=True, inplace=True)
#
# # exclusion of row 262, 263, 264 and 293 (belonging to 1 multirow data point) due to error by read_pdf
# df_stliq.drop([262, 263, 264, 293], inplace=True)
#
# df_stliq.columns = ['original_id', 'molecule', 'smiles', 'measured_st', 'mol_formula', 'cas', 'excess_column']
#
# # fixing empty measured_st column and dropping 'excess_column'
# df_stliq.loc[df_stliq['measured_st'].isnull(), ['measured_st']] = (
#     df_stliq.loc[df_stliq['measured_st'].isnull(), ['excess_column']].values)
# df_stliq.drop('excess_column', axis=1, inplace=True)
#
# # manually adding Octadecamethyloctasiloxane data previously excluded due to multirow error
# df_stliq.loc[-1] = [263, 'Octadecamethyloctasiloxane',
#                         'C[Si](C)(C)O[Si](C)(C)O[Si](C)(C)O[Si](C)(C)O[Si](C)(C)O[Si](C)(C)O[Si](C)(C)O[Si](C)(C)C',
#                         '18.9', 'C18H54O7Si8', '556-69-4']
# df_stliq.index += 1
# df_stliq = df_stliq.sort_index()
#
# # dropping standard deviations in measured_st
# df_stliq['measured_st'] = df_stliq['measured_st'].str.replace('\s\(.*\).*$', '', regex=True)
#
# #converting measured_st to float type & remove silicon-containing samples
# df_stliq['measured_st'] = df_stliq['measured_st'].astype(float)
# df_stliq.drop(df_stliq.index[df_stliq['smiles'].str.contains('Si')], axis=0, inplace=True)
#
# #exporting
# df_stliq.to_csv(data_dir_raw / 'df_stliq_clean.csv')

"""
--------------------------------------------------------------------------------------
"""