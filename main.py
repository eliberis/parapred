from data_provider import *
from structure_processor import *
from model import get_model
from plotting import *
import numpy as np

import patchdock_tools

def main():
    structure = get_structure_from_pdb("data/pdbs/1ahw.pdb")
    antigen_chain = structure[0]['F']
    patchdock_tools.process_transformations("parapred-transforms.txt", antigen_chain)

    # output_patchdock_file(structure, filename="patchdock_receptor_parapred.txt")
    #
    # structure = get_structure_from_pdb("1ahw_truth.pdb")
    # output_patchdock_file(structure, filename="patchdock_receptor_truth.txt")
    #
    # structure = get_structure_from_pdb("1ahw_truth.pdb")
    # output_patchdock_file(structure, filename="patchdock_receptor_cdrs.txt",
    #                       cutoff=0)
    return

    ags, cdrs, lbls, params = open_dataset()

    print("Ags:", ags.shape)
    print("CDRs:", cdrs.shape)
    print("Labels:", lbls.shape)

    max_ag_len = params["max_ag_len"]
    max_cdr_len = params["max_cdr_len"]
    pos_class_weight = params["pos_class_weight"]

    print("Max AG length:", max_ag_len)
    print("Max CDR length:", max_cdr_len)
    print("Pos class weight:", pos_class_weight)

    model = get_model(params["max_ag_len"], params["max_cdr_len"])
    print(model.summary())

    np.random.seed(seed=0)  # TODO replace with stratified split
    test_size = round(len(cdrs) * 0.20)
    indices = np.random.permutation(len(cdrs))

    ags_train = ags[indices[:-test_size]]
    cdrs_train = cdrs[indices[:-test_size]]
    lbls_train = lbls[indices[:-test_size]]
    ags_test = ags[indices[-test_size:]]
    cdrs_test = cdrs[indices[-test_size:]]
    lbls_test = lbls[indices[-test_size:]]
    example_weight = np.squeeze(lbls_train * 5 + 1)  # 6-to-1 in favour of 1

    # history = model.fit([ags_train, cdrs_train], lbls_train,
    #                     batch_size=32, epochs=30,
    #                     sample_weight=example_weight)

    model.load_weights("current.h5")

    # Only use when examples aren't permuted
    # for i, cdr_name in enumerate(["H1", "H2", "H3", "L1", "L2", "L3"]):
    #     curr_ags = ags_test[i::6]
    #     curr_cdrs = cdrs_test[i::6]
    #     curr_lbls = lbls_test[i::6]
    #     plot_prec_rec_curve(model, curr_ags, curr_cdrs, curr_lbls,
    #                         output_filename=(cdr_name + ".png"))

if __name__ == "__main__":
    main()