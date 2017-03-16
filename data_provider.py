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

    for _, entry in train_df.iterrows():
        print("Processing PDB: ", entry['PDB'])

        pdb_file = entry['PDB']
        ab_h_chain = entry['Ab Heavy Chain']
        ab_l_chain = entry['Ab Light Chain']
        ag_chain = entry['Ag']

        ag, cdrs, lbls, _ = open_single_pdb(pdb_file, ab_h_chain,
                                            ab_l_chain, ag_chain,
                                            max_ag_len=max_ag_len,
                                            max_cdr_len=max_cdr_len)

        for i in range(6):
            yield {"ag": ag, "ab": cdrs[i], "lb": lbls[i]}


def open_dataset():
    if isfile(DATASET_PICKLE):
        print("Precomputed dataset found, loading...")
        with open(DATASET_PICKLE, "rb") as f:
            dataset = pickle.load(f)
    else:
        print("Computing and storing the dataset...")
        dataset = list(compute_entries())
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
    cdr_lists = []
    cont_lists = []
    for cdr_name in ["H1", "H2", "H3", "L1", "L2", "L3"]:
        cdr_lists.append([aa_encoded(r) for r in cdrs[cdr_name]])
        cont_lists.append(contact[cdr_name])

    ag_list = [aa_encoded(r) for r in ag]

    return ag_list, cdr_lists, cont_lists, structure
