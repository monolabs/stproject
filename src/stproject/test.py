from pathlib import Path
import numpy as np
import pandas as pd
from tabula import read_pdf
from rdkit import Chem
from rdkit import RDConfig
from utils import *
import re

smile = 'C1=CC=C(C=C1)C(=O)O'
print(count_frags_0(Chem.MolFromSmiles(smile)))
