import os
from copy import deepcopy

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, AllChem

from utils import replace_slash, imgdir, plt, remove_axes, make_rectangular

Chem.Draw.rdMolDraw2D.MolDrawOptions().annotationFontScale = 5.0


def display_atom_index(mol, index=True):
    for atom in mol.GetAtoms():
        if index:
            val = atom.GetIdx()
        else:
            val = 0
        atom.SetAtomMapNum(val)
    return


def draw_reaction(rxn, filename):
    rxn = AllChem.ReactionFromSmarts(rxn, useSmiles=True)
    d = Draw.MolDraw2DCairo(800, 300)
    d.DrawReaction(rxn, highlightByReactant=True)
    png = d.GetDrawingText()
    open(os.path.join(imgdir, filename), 'wb+').write(png)
    return
  

def draw(mols, titles=None, index=False, size=500, show=True, debug=True): 
    mols = deepcopy(mols)
    if debug:
        print(type(mols))
    if type(mols) is pd.Series:
        mols = np.array(mols.values)
    elif type(mols) in [str, Chem.rdchem.Mol]:
        mols = [mols]
        
    for i, mol in enumerate(mols):
        if type(mol) is str:
            mols[i] = Chem.MolFromSmiles(mol)
    
    N = len(mols)
    if N > 25:
        M = int(N/2)
        if titles is None:
            titles = [None]*N
        draw(mols[:M], titles=titles[:M], show=False, index=index, size=size)
        draw(mols[M:], titles=titles[M:], show=True, index=index, size=size)
        return
    
    mols, m, n = make_rectangular(mols)
    if debug:
        print('(m,n)=({0},{1})'.format(m, n))
        
    fig, axes = plt.subplots(m, n, figsize=(n, m))
    axes = np.atleast_2d(axes)
    if m*n > 10:
        size = 500
    k = 0
    for i in range(m):
        for j in range(n):
            mol = mols[i,j]
            if not mol:
                remove_axes(axes[i,j])
                continue
            if type(mol) == str:
                mol = Chem.MolFromSmiles(mol)
            
            display_atom_index(mol, index=index)
            image = Draw.MolsToImage([mol], subImgSize=(size, size), useSVG=True)
            axes[i,j].imshow(image)
            if titles:
                axes[i,j].set_title(titles[k], fontsize=6, y=0.9)
            remove_axes(axes[i,j])
            
            k += 1

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0.2)
    if show:
        plt.show()
    return axes


def save_molecules(mols, folder=None, debug=False):
    mols = deepcopy(mols)
    if debug:
        print(type(mols))
    if type(mols) is pd.Series:
        mols = np.array(mols.values)
    elif type(mols) in [str, Chem.rdchem.Mol]:
        mols = [mols]
        
    for i, mol in enumerate(mols):
        if type(mol) is str:
            mols[i] = Chem.MolFromSmiles(mol)
    
    if not folder:
        folder = os.path.join(imgdir, timestamp())
        os.mkdir(folder)
    
    for mol in mols:
        smiles = Chem.MolToSmiles(mol)
        smiles = replace_slash(smiles)
        filedir = os.path.join(folder, smiles + '.svg')
        Draw.MolToFile(mol, filedir)
    return


if __name__ == '__main__':
    
    smiles = [
        'CNc1ccccc1',
        'Cc1ccccc1N',
        'CCCCCCCCN',
        'NCCN',
        'CCNCCO',
        'NCCNCCN',
        'CNCCNC',
        'C1CCNCC1',
        'CCN(CC)CCCN',
        'COCCNCCOC',
        'CCN(CC)CCCN',
        'C1=CC=C(C=C1)CNCC2=CC=CC=C2',
        'CC(C)NCC1=CC=CC=C1',
        'CN1CCNCC1',
        'C(COCCO)N',
        'C(CN)COCCOCCOCCCN',
        'CN(C)CCCNCCCN',
        'CNCC1=CC=CC=C1',
        'C(CO)NCCO',
        'C1CNC2=CC=CC=C21',
    ]
    
    titles = [
        88.6,
        86.8,
        101.8,
        98.9,
        94.8,
        100.0,
        94.9,
        92.5,
        101.2,
        98.1,
        101.2,
        98.4,
        98.5,
        93.7,
        101.9,
        99.9,
        100.5,
        98.7,
        95.3,
        84.6,
        ]
    
    draw(smiles, titles=titles)