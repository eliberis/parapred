import pickle
import pandas as pd
from os.path import isfile
from structure_processor import *
from data.scrape import download_annotated_seq

PDBS = "data/pdbs/{0}.pdb"
MAX_CDR_LEN = 32  # 28 + 2 + 2


def load_chains(dataset_desc_filename, sequence_cache_file="downloaded_seqs.p"):
    if not isfile(sequence_cache_file):
        download_annotated_sequences(dataset_desc_filename, sequence_cache_file)

    with open(sequence_cache_file, "rb") as f:
        sequences = pickle.load(f)

    df = pd.read_csv(dataset_desc_filename)
    for _, entry in df.iterrows():
        pdb_name = entry['pdb']
        ab_h_chain = entry['Hchain']
        ab_l_chain = entry['Lchain']
        ag_chain = entry['antigen_chain']

        structure = get_structure_from_pdb(PDBS.format(pdb_name))
        model = structure[0]  # Structure only has one model

        if "|" in ag_chain:  # 2 chains
            c1, c2 = ag_chain.split(" | ")
            ag_atoms = \
                Selection.unfold_entities(model[c1], 'A') + \
                Selection.unfold_entities(model[c2], 'A')
        else:  # 1 chain
            ag_atoms = Selection.unfold_entities(model[ag_chain], 'A')

        ag_search = NeighborSearch(ag_atoms)

        ag_chain_struct = None if "|" in ag_chain else model[ag_chain]

        yield ag_search, model[ab_h_chain], model[ab_l_chain], \
              ag_chain_struct, sequences[pdb_name], pdb_name


def process_dataset(summary_file):
    num_in_contact = 0
    num_residues = 0

    all_cdrs = []
    all_lbls = []
    all_masks = []

    for ag_search, ab_h_chain, ab_l_chain, _, seqs, pdb in load_chains(summary_file):
        print("Processing PDB: ", pdb)

        cdrs, lbls, cdr_mask, (nic, nr) =\
            process_chains(ag_search, ab_h_chain, ab_l_chain, seqs, pdb,
                           max_cdr_len=MAX_CDR_LEN)

        num_in_contact += nic
        num_residues += nr

        all_cdrs.append(cdrs)
        all_lbls.append(lbls)
        all_masks.append(cdr_mask)

    cdrs = np.concatenate(all_cdrs, axis=0)
    lbls = np.concatenate(all_lbls, axis=0)
    masks = np.concatenate(all_masks, axis=0)

    return cdrs, lbls, masks, num_residues / num_in_contact


def compute_entries(summary_file):
    cdrs, lbls, masks, cl_w = process_dataset(summary_file)
    return {
        "cdrs": cdrs,
        "lbls": lbls,
        "masks": masks,
        "max_cdr_len": MAX_CDR_LEN,
        "pos_class_weight": cl_w
    }


def open_dataset(summary_file, dataset_cache="processed-dataset.p"):
    if isfile(dataset_cache):
        print("Precomputed dataset found, loading...")
        with open(dataset_cache, "rb") as f:
            dataset = pickle.load(f)
    else:
        print("Computing and storing the dataset...")
        dataset = compute_entries(summary_file)
        with open(dataset_cache, "wb") as f:
            pickle.dump(dataset, f)

    return dataset


def process_chains(ag_search, ab_h_chain, ab_l_chain, sequences, pdb, max_cdr_len):

    # Extract CDRs
    cdrs = {}
    cdrs.update(extract_cdrs(ab_h_chain, sequences["H"], "H"))
    cdrs.update(extract_cdrs(ab_l_chain, sequences["L"], "L"))

    # Compute ground truth -- contact information
    num_residues = 0
    num_in_contact = 0
    contact = {}

    for cdr_name, cdr_chain in cdrs.items():
        contact[cdr_name] = \
            [False if res[1] is None else residue_in_contact_with(res[1], ag_search) for res in cdr_chain]
        num_residues += len(contact[cdr_name])
        num_in_contact += sum(contact[cdr_name])

    if num_in_contact < 5:
        print("Antibody has very few contact residues: ", num_in_contact)

    # Convert to matrices
    # TODO: could simplify with keras.preprocessing.sequence.pad_sequences
    cdr_mats = []
    cont_mats = []
    cdr_masks = []
    for cdr_name in ["H1", "H2", "H3", "L1", "L2", "L3"]:
        # Convert Residue entities to amino acid sequences
        cdr_chain = [r[0] for r in cdrs[cdr_name]]

        cdr_mat = seq_to_one_hot(cdr_chain)
        cdr_mat_pad = np.zeros((max_cdr_len, NUM_FEATURES))
        cdr_mat_pad[:cdr_mat.shape[0], :] = cdr_mat
        cdr_mats.append(cdr_mat_pad)

        cont_mat = np.array(contact[cdr_name], dtype=float)
        cont_mat_pad = np.zeros((max_cdr_len, 1))
        cont_mat_pad[:cont_mat.shape[0], 0] = cont_mat
        cont_mats.append(cont_mat_pad)

        cdr_mask = np.zeros((max_cdr_len, 1), dtype=int)
        cdr_mask[:len(cdr_chain), 0] = 1
        cdr_masks.append(cdr_mask)

    cdrs = np.stack(cdr_mats)
    lbls = np.stack(cont_mats)
    masks = np.stack(cdr_masks)

    return cdrs, lbls, masks, (num_in_contact, num_residues)


def download_annotated_sequences(dataset_desc_filename, cache_file="downloaded_seqs.p"):
    df = pd.read_csv(dataset_desc_filename)
    output = {}

    for _, entry in df.iterrows():  # Do in parallel
        pdb_name = entry['pdb']
        ab_h_chain = entry['Hchain']
        ab_l_chain = entry['Lchain']

        print("Downloading " + pdb_name)
        seq = download_annotated_seq(pdb_name, ab_h_chain, ab_l_chain)
        output[pdb_name] = seq

    with open(cache_file, "wb") as f:
        pickle.dump(output, f)
