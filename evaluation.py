import numpy as np
import pickle
from sklearn.model_selection import KFold
from data_provider import load_chains, TEST_DATASET_DESC_FILE
from structure_processor import save_chain, save_structure, \
    produce_annotated_ab_structure, extended_epitope
from patchdock_tools import output_patchdock_ab_constraint, \
    output_patchdock_ag_constraint, process_transformations
from keras.callbacks import LearningRateScheduler

AB_STRUCT_SAVE_PATH = "data/{0}/{1}_AB.pdb"
AG_STRUCT_SAVE_PATH = "data/{0}/{1}_AG.pdb"
AB_PATCHDOCK_SAVE_PATH = "data/{0}/{1}_ab_patchdock.txt"
AG_PATCHDOCK_SAVE_PATH = "data/{0}/{1}_ag_patchdock.txt"
PATCHDOCK_RESULTS_PATH = "data/{0}/{1}.txt"


def annotate_and_save_test_structures(probs, folder="annotated"):
    chains = load_chains(TEST_DATASET_DESC_FILE)
    for i, (ag_chain, ab_h_chain, ab_l_chain, pdb_name) in enumerate(chains):
        p = probs[6*i:6*(i+1), :]
        ab_struct = produce_annotated_ab_structure(ab_h_chain, ab_l_chain, p)
        save_structure(ab_struct, AB_STRUCT_SAVE_PATH.format(folder, pdb_name))

        save_chain(ag_chain, AG_STRUCT_SAVE_PATH.format(folder, pdb_name))

        abpd_fname = AB_PATCHDOCK_SAVE_PATH.format(folder, pdb_name)
        output_patchdock_ab_constraint(ab_struct, filename=abpd_fname)

        agpd_fname = AG_PATCHDOCK_SAVE_PATH.format(folder, pdb_name)
        ext_epi = extended_epitope(ag_chain, ab_h_chain, ab_l_chain, cutoff=5.0)
        output_patchdock_ag_constraint(ext_epi, ag_chain.id, filename=agpd_fname)


def capri_evaluate_test_structures(folder="results"):
    num_decoys = {'high': 0, 'med': 0, 'low': 0}

    chains = load_chains(TEST_DATASET_DESC_FILE)
    for ag_chain, ab_h_chain, ab_l_chain, pdb_name in chains:
        trans_file = PATCHDOCK_RESULTS_PATH.format(folder, pdb_name)
        q = process_transformations(trans_file, ag_chain, ab_h_chain, ab_l_chain)
        if q is not None: num_decoys[q] += 1

    return num_decoys

def combine_datasets(train_set, test_set):
    ags_train, cdrs_train, lbls_train, mask_train = train_set
    ags_test, cdrs_test, lbls_test, mask_test = test_set
    ags = np.concatenate((ags_train, ags_test))
    cdrs = np.concatenate((cdrs_train, cdrs_test))
    lbls = np.concatenate((lbls_train, lbls_test))
    masks = np.concatenate((mask_train, mask_test))
    return ags, cdrs, lbls, masks


def kfold_cv_eval(model_func, dataset, output_file="crossval-data.p",
                  weights_template="weights-fold-{}.h5"):
    ags, cdrs, lbls, masks = dataset
    kf = KFold(n_splits=10, random_state=0, shuffle=True)

    all_lbls = []
    all_probs = []
    all_masks = []

    for i, (train_idx, test_idx) in enumerate(kf.split(cdrs)):
        print("Fold: ", i + 1)

        ags_train, cdrs_train, lbls_train, mask_train = \
            ags[train_idx], cdrs[train_idx], lbls[train_idx], masks[train_idx]
        ags_test, cdrs_test, lbls_test, mask_test = \
            ags[test_idx], cdrs[test_idx], lbls[test_idx], masks[test_idx]

        example_weight = np.squeeze((lbls_train * 1.5 + 1) * mask_train)
        model = model_func()

        rate_schedule = lambda e: 0.001 if e >= 5 else 0.01

        model.fit([ags_train, cdrs_train, np.squeeze(mask_train)],
                  lbls_train,
                  batch_size=32, epochs=17,
                  sample_weight=example_weight,
                  callbacks=[LearningRateScheduler(rate_schedule)])

        model.save_weights(weights_template.format(i))

        probs_test = model.predict([ags_test, cdrs_test, np.squeeze(mask_test)])
        all_lbls.append(lbls_test)
        all_probs.append(probs_test)
        all_masks.append(mask_test)

    lbl_mat = np.concatenate(all_lbls)
    prob_mat = np.concatenate(all_probs)
    mask_mat = np.concatenate(all_masks)

    with open(output_file, "wb") as f:
        pickle.dump((lbl_mat, prob_mat, mask_mat), f)


def flatten_with_lengths(matrix, lengths):
    seqs = []
    for i, example in enumerate(matrix):
        seq = example[:lengths[i]]
        seqs.append(seq)
    return np.concatenate(seqs)
