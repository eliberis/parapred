import pickle
import pandas as pd
from os.path import isfile
from structure_processor import *

PDBS = "data/pdbs/{0}.pdb"
TRAIN_DATASET_DESC_FILE = "data/abip_train.csv"
TEST_DATASET_DESC_FILE = "data/abip_test.csv"
DATASET_MAX_CDR_LEN = 31  # For padding
DATASET_MAX_AG_LEN = 1269
DATASET_MAX_AG_ATOMS = 10213
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

    all_cdrs = []
    all_lbls = []
    all_ags = []
    all_ag_atoms = []
    all_cdr_masks = []

    for ag_chain, ab_h_chain, ab_l_chain, _ in load_chains(desc_file):
        # Sadly, Biopython structures can't be pickled, it seems
        ag_repl, ag_atoms, cdrs, lbls, cdr_mask, (nic, nr) =\
            process_chains(ag_chain, ab_h_chain, ab_l_chain,
                           max_ag_len=DATASET_MAX_AG_LEN,
                           max_cdr_len=DATASET_MAX_CDR_LEN)

        num_in_contact += nic
        num_residues += nr

        all_cdrs.append(cdrs)
        all_lbls.append(lbls)
        all_ags.append(ag_repl)
        all_cdr_masks.append(cdr_mask)
        all_ag_atoms.append(ag_atoms)

    cdrs = np.concatenate(all_cdrs, axis=0)
    lbls = np.concatenate(all_lbls, axis=0)
    ags = np.concatenate(all_ags, axis=0)
    cdr_masks = np.concatenate(all_cdr_masks, axis=0)
    ag_atoms = np.concatenate(all_ag_atoms, axis=0)

    return ags, ag_atoms, cdrs, lbls, cdr_masks, num_residues / num_in_contact


def compute_entries():
    train_set = process_dataset(TRAIN_DATASET_DESC_FILE)
    test_set = process_dataset(TEST_DATASET_DESC_FILE)
    param_dict = {
        "max_ag_len": DATASET_MAX_AG_LEN,
        "max_cdr_len": DATASET_MAX_CDR_LEN,
        "max_ag_atoms": DATASET_MAX_AG_ATOMS,
        "pos_class_weight": train_set[5]
    }
    return train_set[0:5], test_set[0:5], param_dict  # Hide class weight


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


def neighbour_list_to_matrix(neigh_list):
    # Convert a list of tuples into a 2-element list of lists
    l = [list(t) for t in zip(*neigh_list)]
    # weights = 1 / np.array(l[0])
    residues = residue_seq_to_one(l[1])
    # Multiply each column by a vector element-wise
    return seq_to_feat_matrix(residues) #  * weights[:, np.newaxis]


def atom_list_to_feat_seq(atoms, max_len):
    atom_coords = [a.coord for a in atoms]
    coord_mat = np.stack(atom_coords)

    coord_mat_pad = np.zeros((max_len, 3))
    coord_mat_pad[:coord_mat.shape[0], :] = coord_mat

    return coord_mat_pad


def process_chains(ag_chain, ab_h_chain, ab_l_chain,
                   max_cdr_len, max_ag_len):
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
            [residue_in_contact_with(res[0][1], ag_search) for res in cdr_chain]
        num_residues += len(contact[cdr_name])
        num_in_contact += sum(contact[cdr_name])

    # Convert to matrices
    cdr_mats = []
    cont_mats = []
    cdr_masks = []
    for cdr_name in ["H1", "H2", "H3", "L1", "L2", "L3"]:
        cdr_chain = cdrs[cdr_name]

        neigh_feats = [neighbour_list_to_matrix(n) for n in cdr_chain]
        cdr_mat = np.stack([m.flatten() for m in neigh_feats], axis=0)
        cdr_mat_pad = np.zeros((max_cdr_len, NEIGHBOURHOOD_FEATURES))
        cdr_mat_pad[:cdr_mat.shape[0], :] = cdr_mat
        cdr_mats.append(cdr_mat_pad)

        cont_mat = np.array(contact[cdr_name], dtype=float)
        cont_mat_pad = np.zeros((max_cdr_len, 1))
        cont_mat_pad[:cont_mat.shape[0], 0] = cont_mat
        cont_mats.append(cont_mat_pad)

        cdr_mask = np.zeros((max_cdr_len, 1), dtype=int)
        cdr_mask[:len(cdr_chain), 0] = 1.0
        cdr_masks.append(cdr_mask)

    cdrs = np.stack(cdr_mats)
    lbls = np.stack(cont_mats)
    masks = np.stack(cdr_masks)

    ag_neighs = [neighbour_list_to_matrix(
                    residue_neighbourhood(r, ag_chain, RESIDUE_NEIGHBOURS))
                 for r in ag_chain]
    ag_mat = np.stack([feat.flatten() for feat in ag_neighs])
    ag_mat_pad = np.zeros((max_ag_len, NUM_AG_FEATURES))
    ag_mat_pad[:ag_mat.shape[0], :] = ag_mat

    ag_atom_feats = atom_list_to_feat_seq(ag_atom_list, DATASET_MAX_AG_ATOMS)

    # Replicate AG chain 6 times
    ag_repl = np.resize(ag_mat_pad,
                        (6, ag_mat_pad.shape[0], ag_mat_pad.shape[1]))

    ag_atom = np.resize(ag_atom_feats,
                        (6, ag_atom_feats.shape[0], ag_atom_feats.shape[1]))

    return ag_repl, ag_atom, cdrs, lbls, masks, (num_in_contact, num_residues)
