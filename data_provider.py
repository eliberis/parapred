import pickle
import pandas as pd
from os.path import isfile
from structure_processor import *

PDBS = "data/pdbs/{0}.pdb"
DATASET_DESC_FILE = "data/dataset_desc.csv"
DATASET_MAX_CDR_LEN = 31
DATASET_MAX_AG_LEN = 1269
DATASET_PICKLE = "data.p"


def compute_entries():
    train_df = pd.read_csv(DATASET_DESC_FILE)
    max_cdr_len = DATASET_MAX_CDR_LEN
    max_ag_len = DATASET_MAX_AG_LEN

    all_cdrs = []
    all_lbls = []
    all_ags = []
    for _, entry in train_df.iterrows():
        print("Processing PDB: ", entry['PDB'])

        pdb_file = entry['PDB']
        ab_h_chain = entry['Ab Heavy Chain']
        ab_l_chain = entry['Ab Light Chain']
        ag_chain = entry['Ag']

        ag_repl, cdrs, lbls, _ = open_single_pdb(pdb_file, ab_h_chain,
                                                 ab_l_chain, ag_chain,
                                                 max_ag_len=max_ag_len,
                                                 max_cdr_len=max_cdr_len)

        all_cdrs.append(cdrs)
        all_lbls.append(lbls)
        all_ags.append(ag_repl)

    cdrs = np.concatenate(all_cdrs, axis=0)
    lbls = np.concatenate(all_lbls, axis=0)
    ags = np.concatenate(all_ags, axis=0)

    return (ags, cdrs, lbls,
            {"max_cdr_len": max_cdr_len, "max_ag_len": max_ag_len})


def open_dataset():
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


def open_single_pdb(pdb_file, ab_h_chain_id, ab_l_chain_id, ag_chain_id,
                    max_cdr_len, max_ag_len):
    parser = PDBParser()
    structure = parser.get_structure("", PDBS.format(pdb_file))

    # Extract CDRs and the antigen chain
    model = structure[0]

    cdrs = {}
    cdrs.update(extract_cdrs(model[ab_h_chain_id], ["H1", "H2", "H3"]))
    cdrs.update(extract_cdrs(model[ab_l_chain_id], ["L1", "L2", "L3"]))

    ag = model[ag_chain_id]

    # Compute ground truth -- contact information
    contact = {}
    for cdr_name, cdr_chain in cdrs.items():
        contact[cdr_name] = \
            [residue_in_contact_with_chain(res, ag) for res in cdr_chain]

    # Convert Residue entities to amino acid sequences
    # (TODO replace with tree building later)
    cdrs = {k: residue_seq_to_one(v) for k, v in cdrs.items()}
    ag = residue_seq_to_one(ag)

    # Convert to matrices
    cdr_mats = []
    cont_mats = []
    for cdr_name in ["H1", "H2", "H3", "L1", "L2", "L3"]:
        cdr_chain = cdrs[cdr_name]
        cdr_mat = seq_to_one_hot(cdr_chain)
        cdr_mat_pad = np.zeros((max_cdr_len, NUM_FEATURES))
        cdr_mat_pad[:cdr_mat.shape[0], :] = cdr_mat
        cdr_mats.append(cdr_mat_pad)

        cont_mat = np.array(contact[cdr_name], dtype=float)
        cont_mat_pad = np.zeros((max_cdr_len, 1))
        cont_mat_pad[:cont_mat.shape[0], 0] = cont_mat
        cont_mats.append(cont_mat_pad)

    cdrs = np.stack(cdr_mats)
    lbls = np.stack(cont_mats)

    ag_mat = seq_to_one_hot(ag)
    ag_mat_pad = np.zeros((max_ag_len, NUM_FEATURES))
    ag_mat_pad[:ag_mat.shape[0], :] = ag_mat

    # Replicate AG chain 6 times
    ag_repl = np.resize(ag_mat_pad,
                        (6, ag_mat_pad.shape[0], ag_mat_pad.shape[1]))

    return ag_repl, cdrs, lbls, structure
