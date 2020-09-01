from pathlib import Path
import numpy as np
import pandas as pd
from tabula import read_pdf
from rdkit import Chem
from rdkit import RDConfig
from utils import *
import re

smiles = 'COC=O'
mol = Chem.MolFromSmiles(smiles)
csmiles = Chem.MolToSmiles(mol)
print(Chem.Fragments.fr_C_O(mol))
