from pathlib import Path
import numpy as np
import pandas as pd
from tabula import read_pdf
from rdkit import Chem
from rdkit import RDConfig
from utils import *
import re

smiles = ['C(=O)O', 'CC(=O)O', 'COC=O', 'CC(=O)OC']
names = ['formic acid', 'acetic acid', 'methyl formate', 'methyl acetate']
mols = [Chem.MolFromSmiles(smile) for smile in smiles]
csmiles = [Chem.MolToSmiles(mol) for mol in mols]
fr_COO = [Chem.Fragments.fr_COO(mol) for mol in mols]
fr_ether = [Chem.Fragments.fr_ether(mol) for mol in mols]
fr_ester = [Chem.Fragments.fr_ester(mol) for mol in mols]
print(pd.DataFrame({'name': names, 'csmile': csmiles, 'fr_COO': fr_COO, 'fr_ether': fr_ether, 'fr_ester': fr_ester}))
