import pickle
import pandas as pd
from os.path import isfile
from structure_processor import *

PDBS = "data/pdbs/{0}.pdb"
TRAIN_DATASET_DESC_FILE = "data/abip_train.csv"
TEST_DATASET_DESC_FILE = "data/abip_test.csv"
DATASET_MAX_CDR_LEN = 31  # For padding
DATASET_MAX_AG_LEN = 1267
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

    all_ags = []
    all_ag_edges = []
    all_cdrs = []
    all_cdr_edges = []
    all_lbls = []
    all_cdr_masks = []

    for ag_chain, ab_h_chain, ab_l_chain, _ in load_chains(desc_file):
        # Sadly, Biopython structures can't be pickled, it seems
        ag, ag_edges, cdrs, cdr_edges, lbls, cdr_mask, (nic, nr) =\
            process_chains(ag_chain, ab_h_chain, ab_l_chain,
                           max_ag_len=DATASET_MAX_AG_LEN,
                           max_cdr_len=DATASET_MAX_CDR_LEN)

        num_in_contact += nic
        num_residues += nr

        all_ags.append(ag)
        all_ag_edges.append(ag_edges)
        all_cdrs.append(cdrs)
        all_cdr_edges.append(cdr_edges)
        all_lbls.append(lbls)
        all_cdr_masks.append(cdr_mask)

    ags = np.concatenate(all_ags, axis=0)
    ag_edges = np.concatenate(all_ag_edges, axis=0)
    cdrs = np.concatenate(all_cdrs, axis=0)
    cdr_edges = np.concatenate(all_cdr_edges, axis=0)
    lbls = np.concatenate(all_lbls, axis=0)
    cdr_masks = np.concatenate(all_cdr_masks, axis=0)

    return ags, ag_edges, cdrs, cdr_edges, lbls, cdr_masks, \
           num_residues / num_in_contact


def compute_entries():
    train_set = process_dataset(TRAIN_DATASET_DESC_FILE)
    test_set = process_dataset(TEST_DATASET_DESC_FILE)
    param_dict = {
        "max_ag_len": DATASET_MAX_AG_LEN,
        "max_cdr_len": DATASET_MAX_CDR_LEN,
        "pos_class_weight": train_set[6]
    }
    return train_set[:6], test_set[:6], param_dict  # Hide class weight


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


def process_chains(ag_chain, ab_h_chain, ab_l_chain,
                   max_cdr_len, max_ag_len):

    # Clean up chains
    ab_h_chain = [r for r in ab_h_chain if is_aa_residue(r)]
    ab_l_chain = [r for r in ab_l_chain if is_aa_residue(r)]
    ag_chain = [r for r in ag_chain if is_aa_residue(r)]

    # Extract CDRs
    cdrs = {}
    cdrs.update(extract_cdrs(ab_h_chain, ["H1", "H2", "H3"]))
    cdrs.update(extract_cdrs(ab_l_chain, ["L1", "L2", "L3"]))

    # Compute ground truth -- contact information
    num_residues = 0
    num_in_contact = 0
    contact = {}

    ag_search = NeighborSearch(Selection.unfold_entities(ag_chain, 'A'))

    for cdr_name, cdr_chain in cdrs.items():
        contact[cdr_name] = \
            [residue_in_contact_with(res[0], ag_search) for res in cdr_chain]
        num_residues += len(contact[cdr_name])
        num_in_contact += sum(contact[cdr_name])

    # Convert to matrices
    cdr_mats = []
    cdr_edge_mats = []
    cont_mats = []
    cdr_masks = []
    for cdr_name in ["H1", "H2", "H3", "L1", "L2", "L3"]:
        cdr_chain = cdrs[cdr_name]

        neigh_feats = [neighbour_list_to_feat_matrices(n) for n in cdr_chain]
        res_fts, edge_fts = zip(*neigh_feats)

        cdr_neigh_fts = np.concatenate(res_fts, axis=0)
        cdr_mat_pad = np.zeros((max_cdr_len * NEIGHBOURHOOD_SIZE, NUM_AA_FEATURES))
        cdr_mat_pad[:cdr_neigh_fts.shape[0], :] = cdr_neigh_fts
        cdr_mats.append(cdr_mat_pad)

        cdr_edge_fts = np.concatenate(edge_fts, axis=0)
        cdr_edge_pad = np.zeros((max_cdr_len * (NEIGHBOURHOOD_SIZE ** 2), NUM_EDGE_FEATURES))
        cdr_edge_pad[:cdr_edge_fts.shape[0], :] = cdr_edge_fts
        cdr_edge_mats.append(cdr_edge_pad)

        cont_mat = np.array(contact[cdr_name], dtype=float)
        cont_mat_pad = np.zeros((max_cdr_len, 1))
        cont_mat_pad[:cont_mat.shape[0], 0] = cont_mat
        cont_mats.append(cont_mat_pad)

        cdr_mask = np.zeros((max_cdr_len, 1), dtype=int)
        cdr_mask[:len(cdr_chain), 0] = 1.0
        cdr_masks.append(cdr_mask)

    cdrs = np.stack(cdr_mats)
    cdr_edges = np.stack(cdr_edge_mats)
    lbls = np.stack(cont_mats)
    masks = np.stack(cdr_masks)

    ag_nhood = [residue_neighbourhood(r, ag_chain, NEIGHBOURHOOD_SIZE) for r in ag_chain]
    ag_neigh_feats = [neighbour_list_to_feat_matrices(n) for n in ag_nhood]
    ag_res_fts, ag_edge_fts = zip(*ag_neigh_feats)

    ag_res_fts = np.concatenate(ag_res_fts, axis=0)
    ag_mat_pad = np.zeros((max_ag_len * NEIGHBOURHOOD_SIZE, NUM_AA_FEATURES))
    ag_mat_pad[:ag_res_fts.shape[0], :] = ag_res_fts

    ag_edge_fts = np.concatenate(ag_edge_fts, axis=0)
    ag_edge_pad = np.zeros((max_ag_len * (NEIGHBOURHOOD_SIZE ** 2), NUM_EDGE_FEATURES))
    ag_edge_pad[:ag_edge_fts.shape[0], :] = ag_edge_fts

    # Replicate AG chain 6 times
    ag = np.resize(ag_mat_pad, (6, ag_mat_pad.shape[0], ag_mat_pad.shape[1]))
    ag_edges = np.resize(ag_edge_pad, (6, ag_edge_pad.shape[0], ag_edge_pad.shape[1]))

    return ag, ag_edges, cdrs, cdr_edges, lbls, masks, \
           (num_in_contact, num_residues)
