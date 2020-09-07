from pathlib import Path
import numpy as np
import pandas as pd
from tabula import read_pdf
from rdkit import Chem
from rdkit import RDConfig
from utils import *
import re

s = pd.Series(range(15))
s.append(s)
print(s.append(s))
