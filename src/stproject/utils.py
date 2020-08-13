import numpy as np
from rdkit.Chem import Fragments

def countfrags1(mol):

# return a list of fragment counts from rdkit's 'mol' object
# fragments: [C, CH, CH2, CH3, C-ring, CH-ring, CH2-ring, OH, CHO, CO, COOR, COOH, Ar]

    frags = [0]*12
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'C' and atom.GetTotalDegree() == 4:
            if not atom.IsInRing():
                frags[atom.GetTotalNumHs()] += 1 # update for C, CH, CH2, CH3
            else:
                frags[atom.GetTotalNumHs()+4] += 1 # update for C-ring, CH-ring, CH2-ring

    # update for OH, CHO, CO, COOR, COOH
    frags[7:] = [Fragments.fr_Al_OH(mol) + Fragments.fr_Ar_OH(mol),
                 Fragments.fr_aldehyde(mol),
                 Fragments.fr_ketone(mol),
                 Fragments.fr_ester(mol),
                 Fragments.fr_COO(mol),
                 Fragments.fr_benzene(mol)]

    return frags





