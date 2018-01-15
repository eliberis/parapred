from Bio.PDB import *
from Bio.PDB.Model import Model
from Bio.PDB.Structure import Structure
import numpy as np

# Config constants
NUM_EXTRA_RESIDUES = 2 # The number of extra residues to include on the either side of a CDR
CONTACT_DISTANCE = 4.5 # Contact distance between atoms in Angstroms

chothia_cdr_def = { "L1" : (24, 34), "L2" : (50, 56), "L3" : (89, 97),
                    "H1" : (26, 32), "H2" : (52, 56), "H3" : (95, 102) }

aa_s = "CSTPAGNDEQHRKMILVFYWX" # X for unknown

NUM_FEATURES = len(aa_s) + 7 # one-hot + extra features


def residue_in_cdr(res_id, chain_type):
    cdr_names = [chain_type + str(e) for e in [1, 2, 3]]  # L1-3 or H1-3
    # Loop over all CDR definitions to see if the residue is in one.
    # Inefficient but easier to implement.
    for cdr_name in cdr_names:
        cdr_low, cdr_hi = chothia_cdr_def[cdr_name]
        range_low, range_hi = -NUM_EXTRA_RESIDUES + cdr_low, cdr_hi + NUM_EXTRA_RESIDUES
        if range_low <= res_id[0] <= range_hi:
            return cdr_name
    return None


def find_pdb_residue(pdb_residues, residue_id):
    for pdb_res in pdb_residues:
        if (pdb_res.id[1], pdb_res.id[2].strip()) == residue_id:
            return pdb_res
    return None


def extract_cdrs(chain, sequence, chain_type):
    cdrs = {}
    pdb_residues = chain.get_unpacked_list()
    seq_residues = sorted(sequence)

    for res_id in seq_residues:
        cdr = residue_in_cdr(res_id, chain_type)
        if cdr is not None:
            pdb_res = find_pdb_residue(pdb_residues, res_id)
            cdr_seq = cdrs.get(cdr, [])
            cdr_seq.append((sequence[res_id], pdb_res, res_id))
            cdrs[cdr] = cdr_seq
    return cdrs


def extract_cdrs_from_structure(chain, chain_type):
    cdrs = {}
    pdb_residues = chain.get_unpacked_list()

    for r in pdb_residues:
        cdr = residue_in_cdr(r.get_id()[1:], chain_type)
        if cdr is not None:
            cdr_seq = cdrs.get(cdr, [])
            cdr_seq.append(r)
            cdrs[cdr] = cdr_seq

    return cdrs


def residue_seq_to_one(seq):
    three_to_one = lambda r: Polypeptide.three_to_one(r.resname)\
        if r.resname in Polypeptide.standard_aa_names else 'X'
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


def annotate_chain_with_prob(c, seq, chain_type, probs):
    for a in c.get_atoms():
        a.set_bfactor(0)

    offsets = [0, 0, 0]

    def annotate(res, cdr):
        cdr_id = int(cdr[1]) - 1
        p = probs[cdr_id, offsets[cdr_id]][0]
        if res is not None:
            for a in res.get_atom():
                a.set_bfactor(p * 100)

        offsets[cdr_id] += 1

    if seq is not None:
        for res_id in seq:
            cdr = residue_in_cdr(res_id, chain_type)
            if cdr is not None:
                res = find_pdb_residue(c, res_id)
                annotate(res, cdr)

    else:
        for res in c.get_unpacked_list():
            cdr = residue_in_cdr(res.get_id()[1:], chain_type)
            if cdr is not None:
                annotate(res, cdr)

    return c


def produce_annotated_ab_structure(ab_h_chain, ab_l_chain, ab_seq, probs):
    ab_h_chain = annotate_chain_with_prob(ab_h_chain, ab_seq["H"],
                                          "H", probs[0:3, :])
    ab_l_chain = annotate_chain_with_prob(ab_l_chain, ab_seq["L"],
                                          "L", probs[3:6, :])

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
