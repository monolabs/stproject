import numpy as np
import sklearn.metrics as metrics
from sklearn.model_selection import GridSearchCV
import re
from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt


def diff(first, second):
    second = set(second)
    return [item for item in first if item not in second]

def count_rings(mol):
    # return a dictionary of counts of 3-membered (R3) to 8-membered (R8) rings including aromatics

    rings_dict = dict.fromkeys(['R3', 'R4', 'R5', 'R6', 'R7', 'R8'], 0)
    rings_idx_list = mol.GetRingInfo().AtomRings()

    for ring_idx in rings_idx_list:
        rings_dict['R' + str(len(ring_idx))] += 1

    return rings_dict


def count_frags_0(mol):
    # fragments: [C, H, O, C5, C6, double bond, triple bond, Ar, OH, ]

    frags = dict.fromkeys(['C', 'H', 'O', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'C=C', 'C#C', 'Ar',
                           'OH', 'CHO', 'CO', 'COOH', 'COOR'], 0)
    smiles = Chem.MolToSmiles(mol)
    M = MolWt(mol)

    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'C':
            frags['C'] += 1
        if atom.GetSymbol() == 'O':
            frags['O'] += 1
        frags['H'] += atom.GetTotalNumHs()

    frags['C=C'] = len(re.findall('[0-9C\)]=[C\(]', smiles))
    frags['C#C'] = len(re.findall('[0-9C\)]#[C\(]', smiles))

    ring_dict = count_rings(mol)
    frags.update(ring_dict)

    frags['Ar'] = Chem.Fragments.fr_benzene(mol)
    frags['OH'] = Chem.Fragments.fr_Ar_OH(mol) + Chem.Fragments.fr_Al_OH(mol)
    frags['CHO'] = Chem.Fragments.fr_aldehyde(mol)
    frags['CO'] = Chem.Fragments.fr_ketone(mol)
    frags['COOH'] = Chem.Fragments.fr_COO(mol)
    frags['COOR'] = Chem.Fragments.fr_ester(mol)

    # scaling by molecular weight
    for (key,value) in frags.items():
        frags[key] /= M

    return list(frags.values())

def count_frags_1(mol):
    # return a list of fragment counts from rdkit's 'mol' object
    # fragments: [C, CH, CH2, CH3, C-ring, CH-ring, CH2-ring, Al-OH, Ar-OH, C-O-C, CHO, CO, COOR, COOH, Ar]

    frags = [0] * 12
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'C' and atom.GetTotalDegree() == 4:
            if not atom.IsInRing():
                frags[atom.GetTotalNumHs()] += 1 # update for C, CH, CH2, CH3
            else:
                frags[atom.GetTotalNumHs()+4] += 1 # update for C-ring, CH-ring, CH2-ring

    # update for OH, CHO, CO, COOR, COOH
    frags[7:] = [Chem.Fragments.fr_Al_OH(mol),
                 Chem.Fragments.fr_Ar_OH(mol),
                 Chem.Fragments.fr_aldehyde(mol),
                 Chem.Fragments.fr_ketone(mol),
                 Chem.Fragments.fr_ester(mol),
                 Chem.Fragments.fr_COO(mol),
                 Chem.Fragments.fr_benzene(mol)]

    return frags


def regression_results(y_true, y_pred):

    # Regression metrics
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred)
    mse=metrics.mean_squared_error(y_true, y_pred)
    mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
    r2=metrics.r2_score(y_true, y_pred)

    print('explained_variance: ', round(explained_variance,4))
    print('mean_squared_log_error: ', round(mean_squared_log_error,4))
    print('r2: ', round(r2,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))


def xgb_fit(alg,
             dtrain,
             predictors,
             target,
             useTrainCV=True,
             cv_folds=5,
             early_stopping_rounds=50):

    if useTrainCV:
        xgb_params = alg.get_xgb_params



