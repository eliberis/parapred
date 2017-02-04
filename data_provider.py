from Bio.PDB import *
import pickle
import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical
from os.path import isfile

# Config constants
NUM_EXTRA_RESIDUES = 2 # The number of extra residues to include on the either side of a CDR
CONTACT_DISTANCE = 4.5 # Contact distance between atoms in Angstroms
PDBS = "data/pdbs/{0}.pdb"
DATASET_DESC_FILE = "data/dataset_desc.csv"
DATASET_PICKLE = "data.p"

chothia_cdr_def = { "L1" : (24, 34), "L2" : (50, 56), "L3" : (89, 97),
                    "H1" : (26, 32), "H2" : (52, 56), "H3" : (95, 102) }
aa_s = "CSTPAGNDEQHRKMILVFYWU" # U for unknown


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
    return list(map(lambda r: aa_s.index(r), res_str))


def print_cdrs(cdrs):
    for cdr_name in cdrs:
        residues = residue_seq_to_one(cdrs[cdr_name])
        print(cdr_name, ":", ''.join(residues))

def one_residue_seq_to_one_hot(res_seq_one):
    ints = one_to_number(res_seq_one)
    return to_categorical(ints, nb_classes=len(aa_s))


def atom_in_contact_with_chain(a, c):
    for c_res in c.get_unpacked_list():
        for c_a in c_res.get_unpacked_list():
            if a - c_a < CONTACT_DISTANCE:
                return True
    return False

def residue_in_contact_with_chain(res, c):
    return any(map(lambda atom: atom_in_contact_with_chain(atom, c),
                   res.get_unpacked_list()))

def compute_entries():
    train_df = pd.read_csv(DATASET_DESC_FILE)

    max_cdr_len = 0
    max_ag_len = 0
    dataset = []
    for _, entry in train_df.iterrows():
        print("Processing PDB: ", entry['PDB'])

        parser = PDBParser()
        struct = parser.get_structure("", PDBS.format(entry['PDB']))

        cdrs = {}
        contact = {}
        ag = None

        # Extract CDRs and the antigen chain
        for c in struct.get_chains():
            if c.id == entry['Ab Heavy Chain']:
                cdrs.update(extract_cdrs(c, ["H1", "H2", "H3"]))
            elif c.id == entry['Ab Light Chain']:
                cdrs.update(extract_cdrs(c, ["L1", "L2", "L3"]))
            elif c.id == entry['Ag']:
                ag = c

        # Compute ground truth -- contact information
        for cdr_name, cdr_chain in cdrs.items():
            contact[cdr_name] = \
                list(map(lambda res: residue_in_contact_with_chain(res, ag),
                         cdr_chain))
            max_cdr_len = max(max_cdr_len, len(contact[cdr_name]))

        # Biopython Entities can't be pickled, convert to AA strings (???)
        # TODO investigate?
        cdrs = {k: residue_seq_to_one(v) for k, v in cdrs.items()}
        ag = residue_seq_to_one(ag)
        max_ag_len = max(max_ag_len, len(ag))

        dataset.append({ "cdrs" : cdrs,
                         "contact_truth": contact,
                         "antigen_chain": ag })

    return { "max_cdr_len": max_cdr_len,
             "max_ag_len": max_ag_len,
             "entries": dataset }

def open_dataset():
    dataset = []

    if isfile(DATASET_PICKLE):
        print("Precomputed dataset found, loading...")
        with open(DATASET_PICKLE, "rb") as f:
            dataset = pickle.load(f)
    else:
        print("Computing and storing the dataset...")
        dataset = compute_entries()
        with open(DATASET_PICKLE, "wb") as f:
            pickle.dump(dataset, f)

    return dataset

def load_data_matrices():
    dataset = open_dataset()
    max_cdr_len = dataset["max_cdr_len"]
    max_ag_len = dataset["max_ag_len"]
    num_aas = len(aa_s)

    all_cdrs = []
    all_lbls = []
    all_ags = []
    for entry in dataset["entries"]:
        cdr_mats = []
        cont_mats = []
        for cdr_name in entry["cdrs"]:
            cdr_mat = one_residue_seq_to_one_hot(entry["cdrs"][cdr_name])
            cdr_mat_pad = np.zeros((max_cdr_len, num_aas))
            cdr_mat_pad[:cdr_mat.shape[0], :] = cdr_mat
            cdr_mats.append(cdr_mat_pad)

            cont_mat = np.array(entry["contact_truth"][cdr_name], dtype=float)
            cont_mat_pad = np.zeros((max_cdr_len, 1))
            cont_mat_pad[:cont_mat.shape[0], 0] = cont_mat
            cont_mats.append(cont_mat_pad)

        cdrs = np.stack(cdr_mats, axis=0)
        lbls = np.stack(cont_mats, axis=0)

        ag_mat = one_residue_seq_to_one_hot(entry["antigen_chain"])
        ag_mat_pad = np.zeros((max_ag_len, num_aas))
        ag_mat_pad[:ag_mat.shape[0], :] = ag_mat
        ag_repl = np.resize(ag_mat_pad,
                            (6, ag_mat_pad.shape[0], ag_mat_pad.shape[1]))

        all_cdrs.append(cdrs)
        all_lbls.append(lbls)
        all_ags.append(ag_repl)

    examples = np.concatenate(all_cdrs, axis=0)
    labels = np.concatenate(all_lbls, axis=0)
    ags = np.concatenate(all_ags, axis=0)

    return (examples, labels, ags,
            {"max_cdr_len": max_cdr_len, "max_ag_len": max_ag_len})

