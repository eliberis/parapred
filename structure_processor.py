from Bio.PDB import *
from Bio.PDB.Model import Model
from Bio.PDB.Structure import Structure
import numpy as np

# Config constants
NUM_EXTRA_RESIDUES = 2 # The number of extra residues to include on the either side of a CDR
CONTACT_DISTANCE = 4.5 # Contact distance between atoms in Angstroms

chothia_cdr_def = { "L1" : (24, 34), "L2" : (50, 56), "L3" : (89, 97),
                    "H1" : (26, 32), "H2" : (52, 56), "H3" : (95, 102) }

aa_s = "CSTPAGNDEQHRKMILVFYWU" # U for unknown

NUM_FEATURES = len(aa_s) + 7 # one-hot + extra features


# TODO: Could optimise a bit, but not important
def extract_cdrs(chain, cdr_names):
    cdrs = { name : [] for name in cdr_names }
    for res in chain.get_unpacked_list():
        # Does this residue belong to any of the CDRs?
        for cdr_name in cdrs:
            cdr_low, cdr_hi = chothia_cdr_def[cdr_name]
            cdr_range = range(-NUM_EXTRA_RESIDUES + cdr_low, cdr_hi +
                              NUM_EXTRA_RESIDUES + 1)
            if res.id[1] in cdr_range:
                cdrs[cdr_name].append(res)
    return cdrs


def residue_seq_to_one(seq):
    three_to_one = lambda r: Polypeptide.three_to_one(r.resname)\
        if r.resname in Polypeptide.standard_aa_names else 'U'
    return list(map(three_to_one, seq))


def one_to_number(res_str):
    return [aa_s.index(r) for r in res_str]


def print_cdrs(cdrs):
    for cdr_name in cdrs:
        residues = residue_seq_to_one(cdrs[cdr_name])
        print(cdr_name, ":", ''.join(residues))


def aa_features():
    # Meiler's features
    prop1 = [[1.77, 0.13, 2.43,  1.54,  6.35, 0.17, 0.41],
             [1.31, 0.06, 1.60, -0.04,  5.70, 0.20, 0.28],
             [3.03, 0.11, 2.60,  0.26,  5.60, 0.21, 0.36],
             [2.67, 0.00, 2.72,  0.72,  6.80, 0.13, 0.34],
             [1.28, 0.05, 1.00,  0.31,  6.11, 0.42, 0.23],
             [0.00, 0.00, 0.00,  0.00,  6.07, 0.13, 0.15],
             [1.60, 0.13, 2.95, -0.60,  6.52, 0.21, 0.22],
             [1.60, 0.11, 2.78, -0.77,  2.95, 0.25, 0.20],
             [1.56, 0.15, 3.78, -0.64,  3.09, 0.42, 0.21],
             [1.56, 0.18, 3.95, -0.22,  5.65, 0.36, 0.25],
             [2.99, 0.23, 4.66,  0.13,  7.69, 0.27, 0.30],
             [2.34, 0.29, 6.13, -1.01, 10.74, 0.36, 0.25],
             [1.89, 0.22, 4.77, -0.99,  9.99, 0.32, 0.27],
             [2.35, 0.22, 4.43,  1.23,  5.71, 0.38, 0.32],
             [4.19, 0.19, 4.00,  1.80,  6.04, 0.30, 0.45],
             [2.59, 0.19, 4.00,  1.70,  6.04, 0.39, 0.31],
             [3.67, 0.14, 3.00,  1.22,  6.02, 0.27, 0.49],
             [2.94, 0.29, 5.89,  1.79,  5.67, 0.30, 0.38],
             [2.94, 0.30, 6.47,  0.96,  5.66, 0.25, 0.41],
             [3.21, 0.41, 8.08,  2.25,  5.94, 0.32, 0.42],
             [0.00, 0.00, 0.00,  0.00,  0.00, 0.00, 0.00]]

    # Kidera's features aren't that useful
    # prop2 = [[-0.75,  0.06,  0.63,  1.50,  0.60,  1.14, -0.53,  1.18, -0.19],
    #          [-1.21, -1.19, -0.33, -0.46, -0.54,  0.22, -0.99,  0.74,  1.02],
    #          [-0.67, -0.97,  0.01, -0.36,  0.57,  0.86, -0.68,  0.11,  0.14],
    #          [-0.71,  0.90,  0.21, -0.72, -1.26,  0.86, -1.72,  1.03,  1.98],
    #          [-1.44, -0.47,  0.11,  0.32, -0.51, -0.86,  1.35, -1.29, -0.60],
    #          [-2.16, -1.02, -0.19, -0.03, -0.84, -0.99, -1.72,  1.43,  1.73],
    #          [-0.34, -1.25, -0.60, -0.96, -1.00, -1.19, -0.97,  1.19,  1.27],
    #          [-0.54, -0.75, -1.74, -1.07, -1.17, -1.72, -0.06,  0.74,  1.39],
    #          [ 0.17, -0.62, -1.65, -1.03, -1.74, -1.78,  1.96, -1.21, -0.27],
    #          [ 0.22, -1.24, -0.46, -1.05,  0.19, -0.42,  0.57, -0.14, -0.12],
    #          [ 0.52, -0.46, -0.18, -0.13, -0.56, -0.10,  0.59, -0.27, -0.27],
    #          [ 1.16, -0.57, -1.52, -1.07, -0.28, -0.13, -0.16,  0.28, -0.03],
    #          [ 0.68, -0.16, -1.62, -1.76, -0.86, -1.19,  0.71,  0.40,  0.15],
    #          [ 0.44,  0.20,  0.72,  1.00,  0.45,  0.24,  1.39, -1.24, -1.29],
    #          [ 0.21,  1.37,  0.97,  1.52,  1.91,  1.27,  0.06, -1.30, -1.49],
    #          [ 0.25,  1.06,  1.01,  1.14,  0.69,  0.02,  0.93, -1.36, -1.14],
    #          [-0.34,  0.42,  0.77,  1.38,  1.84,  1.66, -0.09, -1.63, -1.32],
    #          [ 1.09,  1.46,  1.24,  1.16,  0.88,  0.48,  0.37, -0.46, -0.75],
    #          [ 1.34,  1.16,  1.04, -0.07,  1.02,  1.21, -1.25,  0.94,  0.30],
    #          [ 2.08,  2.06,  1.55,  0.67,  0.61,  0.42,  0.23,  0.83, -0.52],
    #          [ 0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00]]
    # return np.concatenate((np.array(prop1), np.array(prop2)), axis=1)
    return np.array(prop1)


def seq_to_one_hot(res_seq_one):
    from keras.utils.np_utils import to_categorical
    ints = one_to_number(res_seq_one)
    feats = aa_features()[ints]
    onehot = to_categorical(ints, num_classes=len(aa_s))
    return np.concatenate((onehot, feats), axis=1)


def atom_in_contact_with_chain(a, c):
    for c_res in c.get_unpacked_list():
        for c_a in c_res.get_unpacked_list():
            if a - c_a < CONTACT_DISTANCE:
                return True
    return False


def residue_in_contact_with(res, c_search, dist=CONTACT_DISTANCE):
    return any(len(c_search.search(a.coord, dist)) > 0
               for a in res.get_unpacked_list())


def annotate_chain_with_prob(c, cdr_names, probs):
    for a in c.get_atoms():
        a.set_bfactor(0)

    for i, cdr_name in enumerate(cdr_names):
        cdr_low, cdr_hi = chothia_cdr_def[cdr_name]
        cdr_range = range(-NUM_EXTRA_RESIDUES + cdr_low,
                          cdr_hi + NUM_EXTRA_RESIDUES + 1)

        j = 0
        for res in c.get_residues():
            if not res.id[1] in cdr_range:
                continue

            p = probs[i, j][0]
            for a in res.get_atom():
                a.set_bfactor(p * 100)

            j += 1
    return c


def produce_annotated_ab_structure(ab_h_chain, ab_l_chain, probs):
    ab_h_chain = annotate_chain_with_prob(ab_h_chain,
                                          ["H1", "H2", "H3"], probs[0:3, :])
    ab_l_chain = annotate_chain_with_prob(ab_l_chain,
                                          ["L1", "L2", "L3"], probs[3:6, :])

    # Create a structure with annotated AB chains
    new_model = Model(0)
    new_model.add(ab_h_chain)
    new_model.add(ab_l_chain)

    structure = Structure(0)
    structure.add(new_model)
    return structure


def save_structure(structure, file_name):
    io = PDBIO()
    io.set_structure(structure)
    io.save(file_name)


def save_chain(chain, file_name):
    model = Model(0)
    model.add(chain)
    struct = Structure(0)
    struct.add(model)

    save_structure(struct, file_name)


def get_structure_from_pdb(pdb_file):
    parser = PDBParser()
    return parser.get_structure("", pdb_file)


def extended_epitope(ag_chain, ab_h_chain, ab_l_chain, cutoff=10.0):
    ab_model = Model(0)
    ab_model.add(ab_h_chain)
    ab_model.add(ab_l_chain)

    ab_search = NeighborSearch(Selection.unfold_entities(ab_model, 'A'))
    epitope = filter(lambda r: residue_in_contact_with(r, ab_search), ag_chain)

    epi_search = NeighborSearch(Selection.unfold_entities(list(epitope), 'A'))
    ext_epi = filter(lambda r: residue_in_contact_with(r, epi_search, dist=cutoff), ag_chain)
    return list(ext_epi)
