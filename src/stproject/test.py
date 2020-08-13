from pathlib import Path
import numpy as np
import pandas as pd
from tabula import read_pdf
from rdkit.Chem import Fragments
from rdkit import RDConfig
from utils import *
import re

smiles = 'C1=CC=C(C(=C1)C(=O)O)C(=O)O'
mol = Chem.MolFromSmiles(smiles)

print(countfrags1(mol))
