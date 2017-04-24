import numpy as np
import pickle
from plotting import plot_prec_rec_curve
from sklearn.model_selection import KFold
from data_provider import load_chains, TEST_DATASET_DESC_FILE
from structure_processor import save_chain, save_structure, \
    produce_annotated_ab_structure, extended_epitope
from patchdock_tools import output_patchdock_ab_constraint, \
    output_patchdock_ag_constraint, process_transformations

AB_STRUCT_SAVE_PATH = "data/annotated/{0}_AB.pdb"
AG_STRUCT_SAVE_PATH = "data/annotated/{0}_AG.pdb"
AB_PATCHDOCK_SAVE_PATH = "data/annotated/{0}_ab_patchdock.txt"
AG_PATCHDOCK_SAVE_PATH = "data/annotated/{0}_ag_patchdock.txt"
PATCHDOCK_RESULTS_PATH = "data/results/{0}.txt"


def annotate_and_save_test_structures(probs):
    chains = load_chains(TEST_DATASET_DESC_FILE)
    for i, (ag_chain, ab_h_chain, ab_l_chain, pdb_name) in enumerate(chains):
        p = probs[6*i:6*(i+1), :]
        ab_struct = produce_annotated_ab_structure(ab_h_chain, ab_l_chain, p)
        save_structure(ab_struct, AB_STRUCT_SAVE_PATH.format(pdb_name))

        save_chain(ag_chain, AG_STRUCT_SAVE_PATH.format(pdb_name))

        abpd_fname = AB_PATCHDOCK_SAVE_PATH.format(pdb_name)
        output_patchdock_ab_constraint(ab_struct, filename=abpd_fname)

        agpd_fname = AG_PATCHDOCK_SAVE_PATH.format(pdb_name)
        ext_epi = extended_epitope(ag_chain, ab_h_chain, ab_l_chain, cutoff=5.0)
        output_patchdock_ag_constraint(ext_epi, ag_chain.id, filename=agpd_fname)


def capri_evaluate_test_structures():
    chains = load_chains(TEST_DATASET_DESC_FILE)
    for ag_chain, ab_h_chain, ab_l_chain, pdb_name in chains:
        trans_file = PATCHDOCK_RESULTS_PATH.format(pdb_name)
        process_transformations(trans_file, ag_chain, ab_h_chain, ab_l_chain)


def combine_datasets(train_set, test_set):
    ags_train, cdrs_train, lbls_train = train_set
    ags_test, cdrs_test, lbls_test = test_set
    ags = np.concatenate((ags_train, ags_test))
    cdrs = np.concatenate((cdrs_train, cdrs_test))
    lbls = np.concatenate((lbls_train, lbls_test))
    return ags, cdrs, lbls


def kfold_cv_eval(model_func, dataset):
    ags, cdrs, lbls = dataset
    kf = KFold(n_splits=10, random_state=0, shuffle=True)

    all_lbls = []
    all_probs = []

    for i, (train_idx, test_idx) in enumerate(kf.split(cdrs)):
        print("Fold: ", i + 1)

        ags_train, cdrs_train, lbls_train = \
            ags[train_idx], cdrs[train_idx], lbls[train_idx]
        ags_test, cdrs_test, lbls_test = \
            ags[test_idx], cdrs[test_idx], lbls[test_idx]

        example_weight = np.squeeze(lbls_train * 5 + 1)  # 6-to-1 in favour of 1
        model = model_func()
        model.fit([ags_train, cdrs_train], lbls_train,
                             batch_size=32, epochs=30,
                             sample_weight=example_weight)

        model.save_weights("fold_weights/{}.h5".format(i))

        probs_test = model.predict([ags_test, cdrs_test])
        all_lbls.append(lbls_test)
        all_probs.append(probs_test)

    lbl_mat = np.concatenate(all_lbls)
    prob_mat = np.concatenate(all_probs)

    with open("fold_weights/dump.p", "wb") as f:
        pickle.dump((lbl_mat, prob_mat), f)

    # with open("fold_weights/dump.p", "rb") as f:
    #     lbl_mat, prob_mat = pickle.load(f)

    plot_prec_rec_curve(lbl_mat, prob_mat, "PR curve for a sequence-only model",
                        output_filename="fold_weights/full.pdf")