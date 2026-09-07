import os
import pickle
import itertools
from copy import deepcopy
from collections import defaultdict, Counter

import pandas as pd
import numpy as np
import chemicals
from scipy.optimize import fmin_slsqp
from rdkit import Chem
from rdkit.Chem import rdFMCS, AllChem, Fragments, Descriptors, Descriptors3D, SanitizeFlags
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger
1
from drawer import draw
from utils import cprint, markrow, isarray
from putils import ray

RDLogger.DisableLog('rdApp.*')  # to suppress (WARNING: not removing hydrogen atom without neighbors)


FUNCTIONAL_GROUPS = {  # retrieved from https://www.daylight.com/dayhtml_tutorials/languages/smarts/smarts_examples.html and revised by DP
    'Alkyl Carbon': '[CX4]',
    'Alkene': 'C=C',  # added
    'Allenic Carbon': '[$([CX2](=C)=C)]',
    'Vinylic Carbon': '[$([CX3]=[CX3])]',
    'Acetylenic Carbon': '[$([CX2]#C)]',
    'Arene': 'c',
    'Carbonyl': '[CX3]=[OX1]',
    'Carbonyl with Carbon': '[CX3](=[OX1])C',
    'Carbonyl with Nitrogen': '[OX1]=CN',
    'Carbonyl with Oxygen': '[CX3](=[OX1])O',
    'Acyl Halide': '[CX3](=[OX1])[F,Cl,Br,I]',
    'Aldehyde': '[CX3H1](=O)[#6]',
    'Anhydride': '[CX3](=[OX1])[OX2][CX3](=[OX1])',
    'Amide': '[#6X3](=[O])[#7]',  # was '[NX3][CX3](=[OX1])[#6]'
    'Amidinium': '[NX3][CX3]=[NX3+]',
    'Carbamate': '[NX3,NX4+][CX3](=[OX1])[OX2,OX1-]',
    'Carbamic ester': '[NX3][CX3](=[OX1])[OX2H0]',
    'Carbamic acid': '[NX3,NX4+][CX3](=[OX1])[OX2H,OX1-]',
    'Carbonic acid': '[CX3](=[OX1])(O)O',
    'Carbonic ester': 'C[OX2][CX3](=[OX1])[OX2]C',
    'Carboxylic acid': '[CX3](=O)[OX1H0-,OX2H1]',
    'Cyanamide': '[NX3][CX2]#[NX1]',
    'Isocynate': '[NX2]=[C]=O',
    'Ester': '[#6][CX3](=O)[OX2H0][#6]',
    'Ketone': '[#6][CX3](=O)[#6]',
    'Ether': '[OD2]([#6])[#6]',
    'Enamine': '[NX3][CX3]=[CX3]',
    'Primary amine': '[#6][NX3;H2]',  # added
    'Secondary amine': '[NX3H1]([#6])[#6]',  # added
    'Primary or secondary amine, not amide': '[NX3;H2;!$(NC=[!#6]);!$(NC#[!#6])][#6]',
    # 'Two primary or secondary amines': '[NX3;H2,H1;!$(NC=O)].[NX3;H2,H1;!$(NC=O)]',  # can't interpret
    'Enamine or Aniline': '[NX3][$(C=C),$(cc)]',
    'Azide': '[$(*-[NX2-]-[NX2+]#[NX1]),$(*-[NX2]=[NX2+]=[NX1-])]',
    'Azide ion': '[$([NX1-]=[NX2+]=[NX1-]),$([NX1]#[NX2+]-[NX1-2])]',
    'Nitrogen': '[#7]',
    'Azo': '[NX2]=[NX2]',
    'Azoxy': '[$([NX2]=[NX3+]([O-])[#6]),$([NX2]=[NX3+0](=[O])[#6])]',
    'Diazo': '[$([#6]=[N+]=[N-]),$([#6-]-[N+]#[N])]',
    'Azole': '[$([nr5]:[nr5,or5,sr5]),$([nr5]:[cr5]:[nr5,or5,sr5])]',
    'Phosphorus': 'P(=O)(O)(O)O',  # added
    'Phosphonite': 'P(=O)(O)(O)C',  # added
    'Chlorosilane': '[Si;X4](Cl)',  # added
    'Boronic acid': '[B](O)(O)',  # added
    'Hydrazine': '[NX3][NX3]',
    'Hydrazone ': '[NX3][NX2]=[*]',
    'Metal oxide': '[!#1;!#6;!#7;!#8;!#9;!#15;!#16;!#17;!#35;!#53]~[#8]',  # added
    'Imine': '[#6;X3]=[#7;X2]',  # added
    'Substituted imine': '[CX3;$([C]([#6])[#6]),$([CH][#6])]=[NX2][#6]',
    'Substituted or un-substituted imine': '[$([CX3]([#6])[#6]),$([CX3H][#6])]=[$([NX2][#6]),$([NX2H])]',
    'Iminium': '[NX3+]=[CX3]',
    'Unsubstituted dicarboximide': '[CX3](=[OX1])[NX3H][CX3](=[OX1])',
    'Substituted dicarboximide': '[CX3](=[OX1])[NX3H0]([#6])[CX3](=[OX1])',
    'Dicarboxdiimide': '[CX3](=[OX1])[NX3H0]([NX3H0]([CX3](=[OX1]))[CX3](=[OX1]))[CX3](=[OX1])',
    'Nitrate': '[$([NX3](=[OX1])(=[OX1])O),$([NX3+]([OX1-])(=[OX1])O)]',
    'Nitrile': '[NX1]#[CX2]',
    'Isonitrile': '[CX1-]#[NX2+]',
    'Nitro': '[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]',
    'Nitroso': '[NX2]=[OX1]',
    'Hydroxyl': '[OX2H]',
    'Hydroxyl in Alcohol': '[#6][OX2H]',
    'Hydroxyl in Carboxylic Acid': '[OX2H][CX3]=[OX1]',
    'Enol': '[OX2H][#6X3]=[#6]',
    'Phenol': '[OX2H][cX3]:[c]',
    'Peroxide': '[OX2,OX1-][OX2,OX1-]',
    'Thiol': '[#16X2H]',
    'Thiol on carbon': '[#6][SX2H]',
    'Thioamide': '[NX3][CX3]=[SX1]',
    'Thiosulfinate': '[S:1](=O)[S:2]',
    'Thiocarbonyl': '[C]=[S]',  # added
    
    'Sulfide': '[#16X2H0]',
    'Sulfinate': '[$([#16X3](=[OX1])[OX2H0]),$([#16X3+]([OX1-])[OX2H0])]',
    'Sulfinic Acid': '[$([#16X3](=[OX1])[OX2H,OX1H0-]),$([#16X3+]([OX1-])[OX2H,OX1H0-])]',
    'Sulfone': '[$([#16X4](=[OX1])=[OX1]),$([#16X4+2]([OX1-])[OX1-])]',
    'Sulfonate': '[$([#16X4](=[OX1])(=[OX1])([#6])[OX2H0]),$([#16X4+2]([OX1-])([OX1-])([#6])[OX2H0])]',
    'Sulfonamide': '[$([#16X4]([NX3])(=[OX1])(=[OX1])[#6]),$([#16X4+2]([NX3])([OX1-])([OX1-])[#6])]',
    'Sulfoxide': '[$([#16X3]=[OX1]),$([#16X3+][OX1-])]',
    'Sulfate': '[$([#16X4](=[OX1])(=[OX1])([OX2H,OX1H0-])[OX2][#6]),$([#16X4+2]([OX1-])([OX1-])([OX2H,OX1H0-])[OX2][#6])]',
    'Sulfuric acid ester': '[$([SX4](=O)(=O)(O)O),$([SX4+2]([O-])([O-])(O)O)]',
    'Sulfamate': '[$([#16X4]([NX3])(=[OX1])(=[OX1])[OX2][#6]),$([#16X4+2]([NX3])([OX1-])([OX1-])[OX2][#6])]',
    'Sulfamic Acid': '[$([#16X4]([NX3])(=[OX1])(=[OX1])[OX2H,OX1H0-]),$([#16X4+2]([NX3])([OX1-])([OX1-])[OX2H,OX1H0-])]',
    'Sulfenic acid': '[#16X2][OX2H,OX1H0-]',
    'Sulfenate': '[#16X2][OX2H0]',
    'Sulfonyl': '[SX4](=O)(=O)',  # added
    'Sulfonate salt': '[S](=O)(=O)([O][Na,K,Li])',  # added
    
    'Oxime': '[C]=[N][OH]',  # added
    
    'Halogen on carbon': '[#6][F,Cl,Br,I]',
    'Halogen': '[F,Cl,Br,I]',
    'Boryl': '[*][B;H2;D1]',  # added
    'Ethynyl': '[*][C]#[C;D1]',  # added
    'Acyl halide': '[CX3](=[OX1])[F,Cl,Br,I]',
    'sp2 cationic carbon': '[$([cX2+](:*):*)]',
    'Aromatic sp2 carbon': '[$([cX3](:*):*),$([cX2+](:*):*)]',
    'Any sp2 carbon': '[$([cX3](:*):*),$([cX2+](:*):*),$([CX3]=*),$([CX2+]=*)]',
    'Any sp2 nitrogen': '[$([nX3](:*):*),$([nX2](:*):*),$([#7X2]=*),$([NX3](=*)=*),$([#7X3+](-*)=*),$([#7X3+H]=*)]',
    'Explicit Hydrogen on sp2-Nitrogen': '[$([#1X1][$([nX3](:*):*),$([nX2](:*):*),$([#7X2]=*),$([NX3](=*)=*),$([#7X3+](-*)=*),$([#7X3+H]=*)])]',
    'sp3 nitrogen': '[$([NX4+]),$([NX3]);!$(*=*)&!$(*:*)]',
    'Explicit Hydrogen on an sp3 N': '[$([#1X1][$([NX4+]),$([NX3]);!$(*=*)&!$(*:*)])]',
    'sp2 N in N-Oxide': '[$([$([NX3]=O),$([NX3+][O-])])]',
    'sp3 N in N-Oxide Exclusive': '[$([$([NX4]=O),$([NX4+][O-])])]',
    'sp3 N in N-Oxide Inclusive': '[$([$([NX4]=O),$([NX4+][O-,#0])])]',
    'Quaternary Nitrogen': '[$([NX4+]),$([NX4]=*)]',
    'Tricoordinate S double bonded to N': '[$([SX3]=N)]',
    'S double-bonded to Carbon': '[$([SX1]=[#6])]',
    'Triply bonded N': '[$([NX1]#*)]',
    'Divalent Oxygen': '[$([OX2])]',
    'Unbranched alkane': '[R0;D2][R0;D2][R0;D2][R0;D2]',
    'Unbranched chain': '[R0;D2]~[R0;D2]~[R0;D2]~[R0;D2]',
    'Long chain': '[AR0]~[AR0]~[AR0]~[AR0]~[AR0]~[AR0]~[AR0]~[AR0]',
    'Terminal S bonded to P': '[$([SX1]~P)]',
    'Nitrogen on -N-C=N-': '[$([NX3]C=N)]',
    'Nitrogen on -N-N=C-': '[$([NX3]N=C)]',
    'Nitrogen on -N-N=N-': '[$([NX3]N=N)]',
    'Oxygen in -O-C=N-': '[$([OX2]C=N)]',
    'S in aromatic 5-ring with lone pair': '[sX2r5]',
    'Aromatic 5-Ring O with Lone Pair': '[oX2r5]',
    'N in 5-sided aromatic ring': '[nX2r5]',
    'N in 5-ring arom': '[$([nX2r5]:[a-]),$([nX2r5]:[a]:[a-])]',
    'CIS or TRANS double bond in a ring': '*/,\[R]=;@[R]/,\*',
    'CIS or TRANS double or aromatic bond in a ring': '*/,\[R]=,:;@[R]/,\*',
    'Unfused benzene ring': '[cR1]1[cR1][cR1][cR1][cR1][cR1]1',
    'Anionic divalent Nitrogen': '[NX2-]',
    'Oxenium Oxygen': '[OX2H+]=*',
    'Oxonium Oxygen': '[OX3H2+]',
    'Carbocation': '[#6+]',
    'Hydrogen-bond donor': '[!$([#6,H0,-,-2,-3])]',
    'Hydrogen-bond acceptor': '[!$([#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]',
    'Possible intramolecular H-bond': '[O,N;!H0]-*~*-*=[$([C,N;R0]=O)]',
}


def enumerate_fragments_deprecated(smiles, skip_warnings=False, skip_rings=False):
    smiles = get_sanitized_smiles(smiles)
    
    mol_stereo = enumerate_stereocenters(smiles)
    if (mol_stereo['atom_unassigned'] != 0) or (mol_stereo['bond_unassigned'] != 0):
        cprint(smiles, 'has undefined stereochemistry', color='y')
        if skip_warnings:
            return

    mol = Chem.MolFromSmiles(smiles)
    mol = add_hydrogens(mol)
    
    fragments = []
    for bond in mol.GetBonds():
        bondidx = bond.GetIdx()
        
        if skip_rings and bond.IsInRing():
            continue

        if bond.GetBondTypeAsDouble() > 1.9999:
            continue

        try:
            # Use RDkit to break the given bond
            mh = Chem.RWMol(mol)
            a1 = bond.GetBeginAtomIdx()
            a2 = bond.GetEndAtomIdx()
                
            mh.RemoveBond(a1, a2)

            mh.GetAtomWithIdx(a1).SetNoImplicit(True)
            mh.GetAtomWithIdx(a2).SetNoImplicit(True)

            # Call SanitizeMol to update radicals
            Chem.SanitizeMol(mh)

            # Convert the two molecules into a SMILES string
            fragmented_smiles = Chem.MolToSmiles(mh)

            # Split fragment and canonicalize
            split_smiles = fragmented_smiles.split(".")
            if len(split_smiles) == 2:
                frag1, frag2 = sorted(split_smiles)
            elif len(split_smiles) == 1:  # ring break case
                frag1, frag2 = split_smiles[0], ''
            else:
                raise RuntimeError('Number of fragments are more than two. Skip.')
            
            # Stoichiometry check
            assert (count_atom_types(frag1) + count_atom_types(frag2)) \
                == count_atom_types(smiles), "Error with {}; {}; {}".format(frag1, frag2, smiles)

            # Check introduction of new stereocenters
            is_valid_stereo = check_stereocenters(frag1) and check_stereocenters(frag2)

            ds = pd.Series({'molecule': get_sanitized_smiles(smiles),
                            'bond_index': bondidx,
                            'bond_type': get_bond_type(bond),
                            'fragment1': get_sanitized_smiles(frag1),
                            'fragment2': get_sanitized_smiles(frag2),
                            'is_valid_stereo': is_valid_stereo,
                            })
            fragments.append(ds)
        except:
            continue
        
    return pd.DataFrame(fragments)


def enumerate_fragments(smiles, skip_warnings=False, skip_rings=False, debug=False):
    smiles = get_sanitized_smiles(smiles)
    
    mol_stereo = enumerate_stereocenters(smiles)
    if (mol_stereo['atom_unassigned'] != 0) or (mol_stereo['bond_unassigned'] != 0):
        if debug:
            cprint(smiles, 'has undefined stereochemistry', color='y')
        if skip_warnings:
            return

    mol = Chem.MolFromSmiles(smiles)
    mol = add_hydrogens(mol)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    
    fragments = []
    for bond in mol.GetBonds():
        bondidx = bond.GetIdx()
        
        if skip_rings and bond.IsInRing():
            continue

        if bond.GetBondTypeAsDouble() > 1.9999:
            continue

        try:
            # Use RDkit to break the given bond
            mh = Chem.RWMol(mol)
            a1 = bond.GetBeginAtomIdx()
            a2 = bond.GetEndAtomIdx()
            
            if (a1 == 4 and a2 == 5) | (a1 == 5 and a2 == 4):
                a = 1
                
            mh.RemoveBond(a1, a2)
            mh.GetAtomWithIdx(a1).SetNoImplicit(True)
            mh.GetAtomWithIdx(a2).SetNoImplicit(True)
            
            # Sanitization
            Chem.SanitizeMol(mh)
            
            # Split fragment and canonicalize
            fragmented_smiles = Chem.MolToSmiles(mh)
            split_smiles = fragmented_smiles.split('.')
            if len(split_smiles) == 2:
                frag1, frag2 = sorted(split_smiles)
            elif len(split_smiles) == 1:  # ring break case
                frag1, frag2 = split_smiles[0], ''
            else:
                raise RuntimeError('Number of fragments are more than two. Skip.')
            
            # Stoichiometry check
            assert (count_atom_types(frag1) + count_atom_types(frag2)) \
                == count_atom_types(smiles), "Error with {}; {}; {}".format(frag1, frag2, smiles)

            # Check introduction of new stereocenters
            is_valid_stereo = check_stereocenters(frag1) and check_stereocenters(frag2)

            ds = pd.Series({'molecule': get_sanitized_smiles(smiles),
                            'bond_index': bondidx,
                            'bond_type': get_bond_type(bond),
                            'fragment1': get_sanitized_smiles(frag1),
                            'fragment2': get_sanitized_smiles(frag2),
                            'is_valid_stereo': is_valid_stereo,
                            })
            fragments.append(ds)
        except:
            continue
        
    return pd.DataFrame(fragments)


def tabulate_fragments(smiles, dataset, prop):
    ds = enumerate_fragments(smiles)
    
    df = []
    for data in dataset:
        smiles_f1 = data.smiles_f1
        smiles_f2 = data.smiles_f2
        y = data.y
        
        indices = (ds['molecule'] == smiles) & (ds['fragment1'] == smiles_f1) & (ds['fragment2'] == smiles_f2)
        bond_index = ds[indices]['bond_index'].values[0]
        bond_type = ds[indices]['bond_type'].values[0]
        
        df.append({'molecule': smiles, 'fragment1': smiles_f1, 'fragment2': smiles_f2, 
                   'bond_index': bond_index, 'bond_type': bond_type, prop: y})
    
    df = pd.DataFrame(df)
    return df


def CAS_to_smiles(cas):
    try:
        mol = chemicals.search_chemical(cas)
        smiles = mol.smiles
        return smiles
    except:  # try NIH
        pass
    
    try:
        cprint('Failed in retrieving SMILES from CAS using chemicals library. Try searching NIH.', color='w')
        url = 'http://cactus.nci.nih.gov/chemical/structure/' + cas + '/smiles'
        smiles = urlopen(url).read().decode('utf8')
        return smiles
    except:
        cprint('Failed in retrieveing SMILES for CAS No.', cas, color='y')
        return None
    

def count_atom_types(smiles):
    """ Return a dictionary of each atom type in the given fragment or molecule
    """
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    mol = Chem.rdmolops.AddHs(mol)
    return Counter([atom.GetSymbol() for atom in mol.GetAtoms()])


def enumerate_stereocenters(smiles):
    """ Returns a count of both assigned and unassigned stereocenters in the
    given molecule """

    mol = Chem.MolFromSmiles(smiles)
    Chem.FindPotentialStereoBonds(mol)

    stereocenters = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    stereobonds = [
        bond
        for bond in mol.GetBonds()
        if bond.GetStereo() is not Chem.rdchem.BondStereo.STEREONONE
    ]

    atom_assigned = len([center for center in stereocenters if center[1] != "?"])
    atom_unassigned = len([center for center in stereocenters if center[1] == "?"])

    bond_assigned = len(
        [
            bond
            for bond in stereobonds
            if bond.GetStereo() is not Chem.rdchem.BondStereo.STEREOANY
        ]
    )
    bond_unassigned = len(
        [
            bond
            for bond in stereobonds
            if bond.GetStereo() is Chem.rdchem.BondStereo.STEREOANY
        ]
    )

    return pd.Series(
        {
            "atom_assigned": atom_assigned,
            "atom_unassigned": atom_unassigned,
            "bond_assigned": bond_assigned,
            "bond_unassigned": bond_unassigned,
        }
    )


def check_stereocenters(smiles):
    """Check the given SMILES string to determine whether accurate
    enthalpies can be calculated with the given stereochem information
    """
    stereocenters = enumerate_stereocenters(smiles)
    if stereocenters["bond_unassigned"] > 0:
        return False

    max_unassigned = 1 if stereocenters["atom_assigned"] == 0 else 1
    if stereocenters["atom_unassigned"] <= max_unassigned:
        return True
    else:
        return False


def get_bond_type(bond):
    return "{}-{}".format(
        *tuple(sorted((bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol())))
    )


def find_farthest_coordinate(neighbors, center, verbose=False):
    coords_neighbors = list(map(lambda neighbor: neighbor.coords, neighbors))
    vectors = [tuple(map(lambda x, y, center=center: x - y, coord, center)) for coord in coords_neighbors]
    avg_length = np.mean(list(map(lambda vector: np.linalg.norm(vector, 2), vectors)))
    X = fmin_slsqp(minimize_sum_inner_products, x0=center, eqcons=[const_bond_length], 
                   args=(vectors, avg_length), iprint=verbose)
    coords = tuple(map(lambda x, y: x + y, X, center))
    return coords


def minimize_sum_inner_products(X, *args):
    vectors = args[0]
    inner_products = list()
    for vector in vectors:
        inner_product = np.dot(X, vector)
        inner_products.append(inner_product)
    return np.sum(inner_products)


def const_bond_length(X, *args):
    avg_length = args[1]
    return np.linalg.norm(X, 2) - avg_length


def convert_degF_to_degC(degF):
    degC = (degF - 32)*5/9
    return degC


def hasring(mol):
    for atom in mol.GetAtoms():
        if atom.IsInRing():
            return False
    return True


def remove_isotope(mol):
    pairs = [(atom, atom.GetIsotope()) for atom in mol.GetAtoms()]
    for atom, isotope in pairs:
        if isotope:
            atom.SetIsotope(0)
    return mol


def MarkLocalCharge(mol):
    out = list()
    for atom in mol.GetAtoms():
        local_charge = atom.GetDoubleProp("_GasteigerCharge")
        atom.SetProp('atomNote', '%.2f'%(local_charge))
    return


def GetLocalCharge(mol):
    out = list()
    for atom in mol.GetAtoms():
        charge = atom.GetFormalCharge()
        out.append(charge)
    return out


def classify_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    
    is_ionic = any(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms())
    is_radical = any(atom.GetNumRadicalElectrons() > 0 for atom in mol.GetAtoms())
    
    if is_ionic:
        return "ionic"
    elif is_radical:
        return "radical"
    else:
        return "neutral"
    

def isFeasibleCharge(mol):
    charges = GetLocalCharge(mol)
    non_neutral_charges = [charge for charge in charges if charge != 0]
    if non_neutral_charges:
        if len(np.unique(np.abs(non_neutral_charges))) == 1:
            return True
        else:
            return False
    return True


def isNeutralCharge(mol):
    charges = GetLocalCharge(mol)
    non_neutral_charges = [charge for charge in charges if charge != 0]
    if non_neutral_charges:
        if sum(non_neutral_charges) == 0:
            return True
        else:
            return False
    return True


def GetFormalChargeList(mol_list):
    out = list()
    for mol in mol_list:
        out.append(Chem.GetFormalCharge(mol))
    return out


def get_squared_formal_charge(mols):
    if type(mols) is list:
        scharges = list(map(lambda mol: Chem.GetFormalCharge(mol)**2, mols))
    elif type(mols) is Chem.rdchem.Mol:
        scharges = Chem.GetFormalCharge(mols)**2
    else:
        raise NotImplementedError
    return scharges


def eval_sum_of_squared_local_charge(mol):
    scharge = 0
    for atom in mol.GetAtoms():
        charge = atom.GetFormalCharge()
        scharge += charge**2
    return scharge


def get_sum_of_squared_local_charges(mols):
    if type(mols) is list:
        scharges = list(map(lambda mol: eval_sum_of_squared_local_charge(mol), mols))
    elif type(mols) is Chem.rdchem.Mol:
        scharges = eval_sum_of_squared_local_charge(mols)
    else:
        raise NotImplementedError
    return scharges


def check_zwitter(mol):
    charges = []
    for atom in mol.GetAtoms():
        charge = atom.GetFormalCharge()
        charges.append(charge)
    
    if any(x > 0 for x in charges) and any(x < 0 for x in charges):
        return True
    else:
        return False


def eval_amine_portion_among_heavy_atoms(molecule):
    mol, smiles = parse_molecule(molecule)
    n_heavy = mol.GetNumHeavyAtoms()
    n_amine = smiles.count('n') + smiles.count('N')
    r_amine = n_amine/n_heavy
    # print(smiles + ':', r_amine)
    return r_amine


def find_min_E_conf(confs):
    idx = 0
    energy = int(1e10)
    for _idx, conf in enumerate(confs):
        _status = conf[0]
        _energy = conf[1]
        if _status != 0:
            continue
        if _energy < energy:
            idx = _idx
            energy = _energy
    return idx, energy


def get_conformers(mol, nconfs=30, mode='ETKDG'):
    mol = deepcopy(mol)
    
    if mode == 'ETKDG':
        status = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if status == 0:
            return mol, 0
        else:
            smiles = Chem.MolToSmiles(mol)
            cprint('Failed in getting conformer of', smiles, color='y')
            return mol, None
        
    elif mode == 'UFF':
        AllChem.EmbedMultipleConfs(mol, nconfs, numThreads=0)
        confs = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=1000, numThreads=0)
        if len(confs) >= 1:
            idx, energy = find_min_E_conf(confs)
            if energy == int(1e10):
                smiles = Chem.MolToSmiles(mol)
                cprint('Failed in getting conformer of', smiles, color='y')
                return mol, None
            else:
                return mol, idx
        else:
            smiles = Chem.MolToSmiles(mol)
            cprint('Failed in getting conformer of', smiles, color='y')
            return mol, None
    else:
        raise NotImplementedError
    

def eval_sphericity(molecule):
    mol, smiles = parse_molecule(molecule)
    try:
        mol = Chem.AddHs(mol)
        mol_, _ = get_conformers(mol)
        return Descriptors3D.SpherocityIndex(mol_)
    except Exception as e:
        cprint('Failed in evaluating sphericity of', smiles, color='y')
        cprint(e)
        return np.nan


def eval_num_rings(molecule):
    mol, smiles = parse_molecule(molecule)
    return Chem.rdMolDescriptors.CalcNumRings(mol)


def eval_num_aromatic_rings(molecule):
    mol, smiles = parse_molecule(molecule)
    return Chem.rdMolDescriptors.CalcNumAromaticRings(mol)


def average_E_amide(pairs):
    if not pairs:
        return None
    else:
        return np.mean([pair[1] for pair in pairs])


def check_E_amide_validity(pairs):
    if not pairs:
        return False
    else:
        return np.any([pair[2] for pair in pairs])


FA = Chem.MolFromSmiles('C(=O)O')
amidation = AllChem.ReactionFromSmarts('[!h0#7:1].[C:2](=[O:3])[O:4]>>[#7:1][C:2](=[O:3]).[H][O:4][H]')
def enumerate_amides(mol):
    # React molecule with FA
    products = amidation.RunReactants([mol, FA])  # this changes the atom indices
    
    # Collect amides
    amides = []
    for product in products:
        assert len(product) == 2
        if Chem.MolToSmiles(product[1]) == '[H]O[H]':  # exclude byproduct H2O
            amides.append(product[0])
        else:
            amides.append(product[1])
    return amides


H2 = Chem.MolFromSmiles('[H][H]')
hemiaminal_formation = AllChem.ReactionFromSmarts('[#7:1][C:2](=[O:3])(-[H:4]).[H:5][H:6]>>[#7:1][C:2](-[O:3][H:5])(-[H:4])(-[H:6])')
def enumerate_hemiaminals(pairs):
    _pairs = []
    for (idx, amide) in pairs:
        # React amide with H2
        amide.UpdatePropertyCache()
        _amide = Chem.AddHs(amide)  # explicit hydrogen
        products = hemiaminal_formation.RunReactants([_amide, H2])
        unique_hemiaminals = np.unique(list(map(lambda x: Chem.MolToSmiles(x[0]), products)))
        assert len(unique_hemiaminals) == 1
        
        # Collect hemiaminals
        hemiaminal = Chem.RemoveHs(products[0][0])
        _pairs.append((idx, hemiaminal))
    return _pairs


def find_formyl_group_matching_index(mol, amides):
    pairs = []
    for amide in amides:
        idx = get_ioncenter(mol, amide, index='acidic')[0]
        pairs.append((idx, amide))
    
    return pairs


def evaluate_CN_BDE(pairs, BDE_predictor, target='amide', verbose=False):
    """
    Using in-house model
    """
    if target == 'amide':
        fragment = '[CH]=O'
    elif target == 'hemiaminal':
        fragment = '[CH2]O'
    else:
        raise RuntimeError("unexpected behavior")
    
    energies = list()
    for idx, amide in pairs:
        smiles = get_sanitized_smiles(amide)
        cleavages = enumerate_fragments(smiles)
        cleavages = cleavages[cleavages['fragment1'].apply(lambda x: x != '') \
                            & cleavages['fragment2'].apply(lambda x: x != '')]
        cleavages['bde'] = BDE_predictor(smiles)
        
        CN_cleavages = cleavages[cleavages['bond_type'] == 'C-N']
        target_cleavage = CN_cleavages[(CN_cleavages['fragment1'] == fragment) \
                                     | (CN_cleavages['fragment2'] == fragment)]
        if len(target_cleavage) == 0:
            cprint('There is no feasieble fragment for', smiles, color='y')
            continue
        else:
            assert len(target_cleavage) == 1
        
        energy = float(target_cleavage['bde'].values)
        energies.append((idx, energy))
    return energies


def evaluate_CN_BDE_deprecated(pairs, BDE_predictor, target='amide', verbose=False):
    """
    Using Alfabet
    """
    if target == 'amide':
        fragment = '[CH]=O'
    elif target == 'hemiaminal':
        fragment = '[CH2]O'
    else:
        raise RuntimeError("unexpected behavior")
    
    energies = list()
    for idx, amide in pairs:
        smiles = Chem.CanonSmiles(Chem.MolToSmiles(amide))
        cleavages = BDE_predictor.predict([smiles], verbose=verbose)
        
        CN_cleavages = cleavages[cleavages['bond_type'] == 'C-N']
        target_cleavage = CN_cleavages[(CN_cleavages['fragment1'] == fragment) \
                                     | (CN_cleavages['fragment2'] == fragment)]
        if len(target_cleavage) == 0:
            cprint('There is no feasieble fragment for', smiles, color='y')
            continue
        else:
            assert len(target_cleavage) == 1
        valid = target_cleavage['is_valid'].values[0]
        energy = target_cleavage['bdfe_pred'].values[0]
        energies.append((idx, energy, valid))
    return energies


def eval_E_amide(molecule, BDE_predictor, value_only=True, verbose=False):
    mol, smiles = parse_molecule(molecule)
    mol = deepcopy(mol)
    # mol = Chem.AddHs(mol)  # explicit hydrogens
    indices = enumerate_formylable_N_indices(mol)
    
    if not indices:  # maybe only tertiary
        cprint('Cannot find any bindable N site for', smiles, color='y')
        return None
    
    # Generate possible amides
    amides = enumerate_amides(mol)
    
    # Find amides matching the given index
    pairs = find_formyl_group_matching_index(mol, amides)
    if not pairs:
        cprint('Failed in generating amides for', smiles, color='y')
        return None
    
    # Evaluate CN-bond dissociation energy
    indices_and_energies = evaluate_CN_BDE(pairs, BDE_predictor, target='amide', verbose=verbose)
    
    if len(indices_and_energies) > 0:
        if value_only:
            return [value for (idx, value) in indices_and_energies]
        else:
            return indices_and_energies
    else:
        return None


def eval_E_hemiaminal(molecule, BDE_predictor, value_only=True, verbose=False):
    mol, smiles = parse_molecule(molecule)
    mol = deepcopy(mol)
    # mol = Chem.AddHs(mol)  # explicit hydrogens
    indices = enumerate_formylable_N_indices(mol)
    
    if not indices:  # maybe only tertiary
        cprint('Cannot find any bindable N site for', smiles, color='y')
        return None
    
    # Generate possible amides
    amides = enumerate_amides(mol)
    
    # Find amides matching the given index
    pairs = find_formyl_group_matching_index(mol, amides)
    if not pairs:
        cprint('Failed in generating amides for', smiles, color='y')
        return None
    
    # Generate hemiaminals corresponding to the pairs
    _pairs = enumerate_hemiaminals(pairs)
    
    # Evaluate CN-bond dissociation energy
    indices_and_energies = evaluate_CN_BDE(_pairs, BDE_predictor, target='hemiaminal', verbose=verbose)
    
    if len(indices_and_energies) > 0:
        if value_only:
            return [value for (idx, value) in indices_and_energies]
        else:
            return indices_and_energies
    else:
        return None


def enumerate_formylable_N_indices(mol):
    indices = list()
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        charge = atom.GetFormalCharge()
        num_H = atom.GetTotalNumHs(includeNeighbors=True)
        if atomic_num == 7 and charge == 0 and num_H > 0:
            idx = atom.GetIdx()
            indices.append(idx)
    return indices


diNpattern = Chem.MolFromSmarts('[NX3;H2,H1][CX4][CX4][NX3;H2,H1]')
def eval_descriptors(molecule, params, i=None, verbose=True):
    if type(params) is ray._raylet.ObjectRef:
        params = ray.get(params)
    BDE_predictor = params[0]
    
    if isinstance(molecule, pd.Series):
        smiles = molecule.smiles
        mol = Chem.MolFromSmiles(smiles)
    else:
        smiles = molecule
        mol = Chem.MolFromSmiles(smiles)
    if verbose:
        print(markrow(i), 'Evaluating descriptors of ' + smiles, flush=True)
    
    # n_ring = eval_num_rings(mol)
    # n_aring = eval_num_aromatic_rings(mol)
    pKa_avg = np.mean(molecule.pKa_mu)
    pKa_max = np.max(molecule.pKa_mu)
    sphericity = eval_sphericity(mol)
    E_amide = eval_E_amide(mol, BDE_predictor)
    E_amide_avg = average_E_amide(E_amide)
    E_amide_validity = check_E_amide_validity(E_amide)
    n_primaryN = Fragments.fr_NH2(mol)
    n_secondaryN = Fragments.fr_NH1(mol)
    n_tertiaryN = Fragments.fr_NH0(mol)
    n_aromaticN = Fragments.fr_Ar_NH(mol)
    n_hydroxylN = Fragments.fr_N_O(mol)
    n_diNpattern = len(mol.GetSubstructMatches(diNpattern))
    r_diNpattern = n_diNpattern/mol.GetNumHeavyAtoms()
    return (pKa_avg, pKa_max, sphericity, E_amide, E_amide_avg, E_amide_validity, \
            n_primaryN, n_secondaryN, n_tertiaryN, n_aromaticN, n_hydroxylN, \
            n_diNpattern, r_diNpattern)  # columns_additionals


CCC_ring_group = Chem.MolFromSmarts('C1CC1')
COC_ring_group = Chem.MolFromSmarts('C1OC1')
CNC_ring_group = Chem.MolFromSmarts('C1NC1')
CCCC_ring_group = Chem.MolFromSmarts('C1CCC1')
CCOC_ring_group = Chem.MolFromSmarts('C1COC1')
CCNC_ring_group = Chem.MolFromSmarts('C1CNC1')


def contains_triangle_square(molecule):
    mol, smiles = parse_molecule(molecule)
    for group in [CCC_ring_group, COC_ring_group, CNC_ring_group,
                   CCCC_ring_group, CCOC_ring_group, CCNC_ring_group]:
        indices = mol.GetSubstructMatches(group)
        indices = list(itertools.chain(*indices))  # flatten
        if len(indices) != 0:
            return True
    return False


def contains_ring_non_neighboring_nitrogen(molecule):
    mol, smiles = parse_molecule(molecule)
    rings = mol.GetRingInfo().AtomRings()
    for ring in rings:
        found = False
        for idx in ring:
            if found:
                break
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetSymbol() in ['n', 'N']:
                found = True
                break
            neighbors = atom.GetNeighbors()
            for neighbor in neighbors:
                if neighbor.GetSymbol() in ['n', 'N']:
                    found = True
                    break
        if not found:
            return True
    return False
    

methyl_group = Chem.MolFromSmarts('[CH3]')
def contains_methyl_group_non_neighboring_nitrogen(molecule):
    mol, smiles = parse_molecule(molecule)
    mol = remove_isotope(mol)
    mol = Chem.RemoveHs(mol)
    
    indices = mol.GetSubstructMatches(methyl_group)
    indices = list(itertools.chain(*indices))  # flatten
    if len(indices) == 0:
        return False
    else:
        for idx in indices:
            neighbors = mol.GetAtomWithIdx(idx).GetNeighbors()
            assert len(neighbors) == 1
            if not neighbors[0].GetSymbol() in ['n', 'N']:
                return True
    return False


def is_simple(molecule):
    mol, smiles = parse_molecule(molecule)
    mol = Chem.RemoveHs(mol)
    preterminals = list()
    indices = list()
    in_symmetry = list()
    
    # Find pre-terminal atoms
    for atom in mol.GetAtoms():
        if len(atom.GetBonds()) != 1:
            continue  # consider only terminal atom
        if atom.GetSymbol() == 'H':
            continue  # consider only heavy atom
        neighbors = [neighbor for neighbor in atom.GetNeighbors()]
        assert len(neighbors) == 1
        idx = neighbors[0].GetIdx()
        if idx in indices:
            continue
        preterminals.append(neighbors[0])
        indices.append(idx)
    
    for preterminal in preterminals:
        idx = preterminal.GetIdx()
        atype = preterminal.GetSymbol()
        if idx in in_symmetry:
            continue
        neighbors = [neighbor.GetSymbol() for neighbor in preterminal.GetNeighbors()]
        
        found = False
        for _preterminal in preterminals:
            _idx = _preterminal.GetIdx()
            _atype = _preterminal.GetSymbol()
            if idx == _idx:
                continue
            _neighbors = [_neighbor.GetSymbol() for _neighbor in _preterminal.GetNeighbors()]
            
            # Same pattern exists
            if Counter(neighbors) == Counter(_neighbors):
                in_symmetry.append(idx)
                in_symmetry.append(_idx)
                found = True
                break
            
            # Has symmetry
            distances = [len(Chem.rdmolops.GetShortestPath(mol, idx, idx_)) for idx_ in indices if idx != idx_]
            if len(set(distances)) <= 1:
                in_symmetry.append(idx)
                found = True
                break
            
        if not found:
            return False
    return True


def is_symmetric(molecule):
    mol, smiles = parse_molecule(molecule)
    if not '1' in smiles:
        return True  # skip linear molecule
    
    patterns = list()  # two step patterns
    for atom in mol.GetAtoms():
        s0 = atom.GetSymbol()
        for neighbor0 in atom.GetNeighbors():
            s1 = neighbor0.GetSymbol()
            for neighbor1 in neighbor0.GetNeighbors():
                if neighbor1 == atom:
                    continue
                s2 = neighbor1.GetSymbol()
                patterns.append(s0 + s1 + s2)
    patterns_unique = set(patterns)
    
    for pattern_unique in patterns_unique:
        counts = patterns.count(pattern_unique)
        if counts < 2:
            return False
    return True


def has_symmetry(smiles):
    if not '(' in smiles:
        return True
    mol = Chem.MolFromSmiles(smiles)
    symmetry_list = []
    counter = 0
    for atom in mol.GetAtoms():
        previousCounter = counter
        idx = atom.GetIdx()
        neighbors = [neighbor.GetIdx() for neighbor in atom.GetNeighbors()]
        if len(neighbors) < 2:
            continue
        pairs = list(itertools.combinations(neighbors, 2))

        for pair in pairs:
            _mol = Chem.RWMol(mol)
            _mol.RemoveBond(pair[0], idx)
            _mol.GetAtomWithIdx(pair[0]).SetNoImplicit(True)
            _mol.RemoveBond(pair[1], idx)
            _mol.GetAtomWithIdx(pair[1]).SetNoImplicit(True)
            _mol.GetAtomWithIdx(idx).SetNoImplicit(True)
            Chem.SanitizeMol(_mol)

            smiles_fragmented = str(Chem.MolToSmiles(_mol))
            FragA, FragB, FragC = sorted(smiles_fragmented.split('.'))
            if (FragA == FragB and FragA != '[H]') or \
                (FragA == FragC and FragA != '[H]') or \
                (FragB == FragC and FragB != '[H]'):
                if previousCounter == counter:
                    symmetry_list.append([idx])
                    symmetry_list[counter].append(pair)
                    counter = counter + 1
                else:
                    symmetry_list[counter - 1].append(pair)
    if symmetry_list:
        return True
    else:
        return False


def has_ring(smiles):
    mol = Chem.MolFromSmiles(str(smiles) or '')
    if mol is None:
        return False
    return any(a.IsInRing() for a in mol.GetAtoms())


def has_hydroxyl(smiles):
    mol = Chem.MolFromSmiles(str(smiles) or '')
    if mol is None:
        return False
    patt = Chem.MolFromSmarts('[OX2H]')
    return mol.HasSubstructMatch(patt)


def isSameMolecule(mol1, mol2):
    smiles1 = get_sanitized_smiles(mol1)
    smiles2 = get_sanitized_smiles(mol2)
    if smiles1 == smiles2:
        return True
    else:
        return False


def MolFromSmilesList(smileses):
    assert type(smileses) in [list, tuple, set, np.ndarray]
    mols = list()
    for smiles in smileses:
        try:
            mol = Chem.MolFromSmiles(smiles)
            Chem.SanitizeMol(mol)
        except:
            cprint('Some molecule failed in conversion', color='y')
        mols.append(mol)
    return mols


def MolToSmilesList(mols, sanitize=True):
    assert type(mols) in [list, tuple, set]
    if sanitize:
        for mol in mols:
            try:
                Chem.SanitizeMol(mol)
            except:
                cprint('Failed in sanitization of', smiles, color='y')
                try:
                    smiles = Chem.CanonSmiles(Chem.MolToSmiles(mol))
                except:
                    smiles = Chem.MolToSmiles(mol)
    smileses = list()
    for mol in mols:
        try:
            smiles = Chem.CanonSmiles(Chem.MolToSmiles(mol))
        except:
            cprint('Some molecule failed in conversion', color='y')
        smileses.append(smiles)
    return smileses


def contains_atomicnum_gt16(mol):
    return any(atom.GetAtomicNum() > 16 for atom in mol.GetAtoms())

    
def has_isotopes(molecule):
    mol = get_sanitized_mol(molecule)
    for atom in mol.GetAtoms():
        if atom.GetIsotope() > 0:
            return True
    return False


def get_atomic_composition(molecule, permit_charge=True, use_digit=False, base_elements=None):
    if type(molecule) == str:
        mol = Chem.MolFromSmiles(molecule)
    else:
        mol = molecule
    
    mol_with_H = Chem.AddHs(mol)
    if base_elements:
        comp = {}
        for elem in base_elements:
            comp[elem] = 0
    else:
        comp = defaultdict(lambda: 0)

    for atom in mol_with_H.GetAtoms():
        if use_digit:
            comp[atom.GetAtomicNum()] += 1
        else:
            comp[atom.GetSymbol()] += 1

    if not permit_charge:
        charge = Chem.GetFormalCharge(mol_with_H)  # if charged, add charge as "atomic number" 0
        if charge != 0:
            raise NotImplementedError
    return dict(comp)
    
    
def get_atomic_energy(smiles, dictionary):
    composition = get_atomic_composition(smiles)
    energy = 0
    for element, count in composition.items():
        energy += dictionary[element] * count 
    return energy
    
    
def get_energy_dictionary(df, dataset, etype, unit_conversion=1):
    ds = df[df['dataset'] == dataset]
    
    dictionary = {}
    for idx, row in ds.iterrows():
        element = row['element']
        energy = row[etype]
        dictionary[element] = energy * unit_conversion
    return dictionary


def eval_molecular_weight(smiles):
    Mw = Descriptors.ExactMolWt(Chem.MolFromSmiles(smiles))
    return Mw


def eval_molecular_weights(df):
    df['Mw'] = [Descriptors.ExactMolWt(Chem.MolFromSmiles(smiles)) for smiles in df['smiles']]
    return df


def correct_valence(mol):
    _mol = deepcopy(mol)
    atoms = list(_mol.GetAtoms())
    for atom in atoms:
        chg = atom.GetFormalCharge()
        hcount = atom.GetTotalNumHs(includeNeighbors=True)
        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(hcount - chg)
        atom.UpdatePropertyCache()
    return _mol


def plot_charge_profile(y, ax=None):
    y = y.flatten()
    if not ax:
        fig, ax = plt.subplots(1,1)
    ax.plot(x, y.reshape(-1))
    plt.show()
    return


def find_radical_atom_idx(mol):
    for atom in mol.GetAtoms():
        if atom.GetNumRadicalElectrons() != 0:
            return atom.GetIdx()


def check_fission(smiles, bond_type, bond_index, element_index):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.rdmolops.AddHs(mol)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    
    bond = mol.GetBondWithIdx(bond_index)
    symbols = bond_type.split('-')

    atom_idx1 = bond.GetBeginAtom().GetAtomicNum()
    atom_idx2 = bond.GetEndAtom().GetAtomicNum()
    symbol1 = element_index[atom_idx1].symbol
    symbol2 = element_index[atom_idx2].symbol
    feasible = (symbol1 == symbols[0] and symbol2 == symbols[1]) or \
               (symbol2 == symbols[0] and symbol1 == symbols[1])
    if not feasible:
        cprint('Fission is not correct for', i, 'th entry!', color='y')
    return feasible


dx = 0.005
x = np.linspace(-0.25, 0.25, 101)
def eval_peak_area(y, x_peak=-0.1, plot=False):
    y = y.flatten()
    x_min = x_peak - dx*10
    x_max = x_peak + dx*10
    indices = [idx for idx, x_ in enumerate(x) if (x_min < x_) and (x_ < x_max)]
    x_peak_ = [x_ for idx, x_ in enumerate(x) if idx in indices]
    y_peak_ = [y_ for idx, y_ in enumerate(y) if idx in indices]
    S = sum(np.array(y_peak_)*dx)
    if plot:
        fig, ax = plt.subplots(1,1)
        plot_charge_profile(y, ax=ax)
        ax.fill_between(x_peak_, y_peak_)
    return S


def neutralize_molecule(molecule):
    if isinstance(molecule, str):
        mol = Chem.MolFromSmiles(molecule)
    else:
        mol = deepcopy(molecule)
    
    mol = Chem.RemoveHs(mol)
    
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    matches = mol.GetSubstructMatches(pattern)
    matches_list = [match[0] for match in matches]
    
    if len(matches_list) > 0:
        for idx in matches_list:
            atom = mol.GetAtomWithIdx(idx)
            chg = atom.GetFormalCharge()
            hcount = atom.GetTotalNumHs(includeNeighbors=True)
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(hcount - chg)
            atom.UpdatePropertyCache()
    
    Chem.SanitizeMol(mol)  # do not apply canonical ordering of atoms to keep original atom indices
    return mol


def get_sanitized_mol(molecule, **kwargs):
    return parse_molecule(molecule, returns='mol', **kwargs)


def get_sanitized_smiles(molecule, **kwargs):
    return parse_molecule(molecule, returns='smiles', **kwargs)


def parse_molecule(molecule, returns='both', canonical_ordering=True, canonical_smiles=True):
    if isarray(molecule):
        return list(map(lambda x: parse_molecule(x, returns=returns, 
                                                    canonical_ordering=canonical_ordering,
                                                    canonical_smiles=canonical_smiles), 
                                                molecule))
    
    # Get RDKit molecule object
    if molecule in ['', None]:
        return molecule
    elif isinstance(molecule, str):
        mol = Chem.MolFromSmiles(molecule)
    else:
        mol = molecule
    
    # Sanitization
    try:
        Chem.SanitizeMol(mol)
    except:
        cprint('Failed in sanitization', color='y')
        return None
    
    # Canonicalization of atom index
    if canonical_ordering:
        mol = canonical_atom_index_ordering(mol)
    
    smiles = Chem.MolToSmiles(mol)
    
    # Canonicalization of smiles
    if canonical_smiles:
        smiles = Chem.CanonSmiles(smiles)
    
    if returns == 'both':
        return mol, smiles
    elif returns == 'mol':
        return mol
    elif returns == 'smiles':
        return smiles


def canonical_atom_index_ordering(mol):
    order = tuple(zip(*sorted([(j, i) for i, j in enumerate(Chem.CanonicalRankAtoms(mol))])))[1]
    mol_ = Chem.RenumberAtoms(mol, order)
    return mol


def add_hydrogens(mol):
    mol = Chem.rdmolops.AddHs(mol)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    mol = get_sanitized_mol(mol)
    return mol


def get_ioncenter(acidic, basic, index='both'):
    if type(acidic) is str:
        acidic = Chem.MolFromSmiles(acidic)
    if type(basic) is str:
        basic = Chem.MolFromSmiles(basic)
        
    if isSameMolecule(acidic, basic):  # assert given acidic and basic components are not the same
        # smileses = MolToSmilesList([acidic, basic])
        smileses = get_sanitized_smiles([acidic, basic], canonical_ordering=False)
        cprint('Check', smileses)
        raise()
    
    mcs = rdFMCS.FindMCS([acidic, basic])  # maximum common substructure
    mcsp = Chem.MolFromSmarts(mcs.smartsString)
    s1 = acidic.GetSubstructMatch(mcsp)
    s2 = basic.GetSubstructMatch(mcsp)
    
    ioncenters = list()
    for i, j in zip(s1, s2):
        acidic_atom = acidic.GetAtomWithIdx(i)
        basic_atom = basic.GetAtomWithIdx(j)
        if acidic_atom.GetTotalNumHs() != basic_atom.GetTotalNumHs():
            if index == 'acidic':
                ioncenters.append(i)
            elif index == 'basic':
                ioncenters.append(j)
            else:
                ioncenters.append((i,j)) # both
    
    return ioncenters

