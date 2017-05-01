import pickle
import pandas as pd
from os.path import isfile
from structure_processor import *

PDBS = "data/pdbs/{0}.pdb"
TRAIN_DATASET_DESC_FILE = "data/abip_train.csv"
TEST_DATASET_DESC_FILE = "data/abip_test.csv"
DATASET_MAX_CDR_LEN = 31  # For padding
DATASET_MAX_AG_LEN = 1269
DATASET_MAX_ATOMS_PER_RESIDUE = 24
DATASET_MAX_CDR_ATOMS = DATASET_MAX_CDR_LEN * DATASET_MAX_ATOMS_PER_RESIDUE
DATASET_MAX_AG_ATOMS = DATASET_MAX_AG_LEN * DATASET_MAX_ATOMS_PER_RESIDUE
DATASET_PICKLE = "data.p"


def load_chains(dataset_desc_filename):
    df = pd.read_csv(dataset_desc_filename)
    for _, entry in df.iterrows():
        print("Processing PDB: ", entry['PDB'])

        pdb_name = entry['PDB']
        ab_h_chain = entry['Ab Heavy Chain']
        ab_l_chain = entry['Ab Light Chain']
        ag_chain = entry['Ag']

        structure = get_structure_from_pdb(PDBS.format(pdb_name))
        model = structure[0] # Structure only has one model

        yield model[ag_chain], model[ab_h_chain], model[ab_l_chain], pdb_name


def process_dataset(desc_file):
    num_in_contact = 0
    num_residues = 0

    all_ag_atoms = []
    all_cdr_atoms = []
    all_lbls = []
    all_cont_masks = []

    for ag_chain, ab_h_chain, ab_l_chain, _ in load_chains(desc_file):
        # Sadly, Biopython structures can't be pickled, it seems
        ag_atoms, cdr_atoms, lbls, cont_mask, (nic, nr) =\
            process_chains(ag_chain, ab_h_chain, ab_l_chain)

        num_in_contact += nic
        num_residues += nr

        all_ag_atoms.append(ag_atoms)
        all_cdr_atoms.append(cdr_atoms)

        all_lbls.append(lbls)
        all_cont_masks.append(cont_mask)

    ag_atoms = np.concatenate(all_ag_atoms, axis=0)
    cdr_atoms = np.concatenate(all_cdr_atoms, axis=0)
    lbls = np.concatenate(all_lbls, axis=0)
    cont_masks = np.concatenate(all_cont_masks, axis=0)

    return ag_atoms, cdr_atoms, lbls, cont_masks, num_residues / num_in_contact


def compute_entries():
    train_set = process_dataset(TRAIN_DATASET_DESC_FILE)
    test_set = process_dataset(TEST_DATASET_DESC_FILE)
    param_dict = {
        "max_ag_len": DATASET_MAX_AG_LEN,
        "max_cdr_len": DATASET_MAX_CDR_LEN,
        "max_ag_atoms": DATASET_MAX_AG_ATOMS,
        "max_cdr_atoms": DATASET_MAX_CDR_ATOMS,
        "max_atoms_per_residue": DATASET_MAX_ATOMS_PER_RESIDUE,
        "pos_class_weight": train_set[4]
    }
    return train_set[:4], test_set[:4], param_dict  # Hide class weight


def open_dataset():
    if isfile(DATASET_PICKLE):
        print("Precomputed dataset found, loading...")
        with open(DATASET_PICKLE, "rb") as f:
            dataset = pickle.load(f)
    else:
        print("Computing and storing the dataset...")
        dataset = compute_entries()
        with open(DATASET_PICKLE, "wb") as f:
            pickle.dump(dataset, f, protocol=4)

    return dataset


def process_chains(ag_chain, ab_h_chain, ab_l_chain):
    # Extract CDRs
    cdrs = {}
    cdrs.update(extract_cdrs(ab_h_chain, ["H1", "H2", "H3"]))
    cdrs.update(extract_cdrs(ab_l_chain, ["L1", "L2", "L3"]))

    # Compute ground truth -- contact information
    num_residues = 0
    num_in_contact = 0
    contact = {}

    ag_atom_list = Selection.unfold_entities(ag_chain, 'A')
    ag_search = NeighborSearch(ag_atom_list)

    for cdr_name, cdr_chain in cdrs.items():
        contact[cdr_name] = \
            [residue_in_contact_with(res, ag_search) for res in cdr_chain]
        num_residues += len(contact[cdr_name])
        num_in_contact += sum(contact[cdr_name])

    # Convert to matrices
    cdr_atoms = []
    cont_truth = []
    cont_masks = []
    for cdr_name in ["H1", "H2", "H3", "L1", "L2", "L3"]:
        cdr_chain = cdrs[cdr_name]

        cdr_atom_feats = \
            residue_list_to_atom_features(cdr_chain,
                                          DATASET_MAX_ATOMS_PER_RESIDUE,
                                          DATASET_MAX_CDR_ATOMS)
        cdr_atoms.append(cdr_atom_feats)

        cont_mat = np.array(contact[cdr_name], dtype=float)
        cont_mat_pad = np.zeros((DATASET_MAX_CDR_LEN, 1))
        cont_mat_pad[:cont_mat.shape[0], 0] = cont_mat
        cont_truth.append(cont_mat_pad)

        cdr_mask = np.zeros((DATASET_MAX_CDR_LEN, 1), dtype=int)
        cdr_mask[:len(cdr_chain), 0] = 1.0
        cont_masks.append(cdr_mask)

    cdr_atoms = np.stack(cdr_atoms)
    lbls = np.stack(cont_truth)
    masks = np.stack(cont_masks)

    ag_chain = filter(is_aa_residue, ag_chain)
    ag_atom_feats = \
        residue_list_to_atom_features(ag_chain,
                                      DATASET_MAX_ATOMS_PER_RESIDUE,
                                      DATASET_MAX_AG_ATOMS)

    # Replicate AG chain 6 times
    ag_atoms = np.resize(ag_atom_feats,
                         (6, ag_atom_feats.shape[0], ag_atom_feats.shape[1]))

    return ag_atoms, cdr_atoms, lbls, masks, (num_in_contact, num_residues)
