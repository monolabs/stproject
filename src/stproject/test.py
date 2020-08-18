from pathlib import Path
import numpy as np
import pandas as pd
from tabula import read_pdf
from rdkit import Chem
from rdkit import RDConfig
from utils import *
import re

smiles = 'C1=CC=CC=C1'
mol = Chem.MolFromSmiles(smiles)
print(Chem.MolToSmiles(mol))