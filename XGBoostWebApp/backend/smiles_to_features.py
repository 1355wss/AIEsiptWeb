from rdkit import Chem
from rdkit.Chem.AtomPairs import Pairs
import numpy as np
from rdkit.Chem import AllChem
import pandas as pd

def smiles_to_vector(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    fpgen = AllChem.GetAtomPairGenerator()
    fp = fpgen.GetCountFingerprint(mol)
    fp = fp.ToList()
    fp = pd.DataFrame(fp)
    fp = fp.to_numpy().flatten()
    return fp


