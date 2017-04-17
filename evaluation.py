from data_provider import load_chains, TEST_DATASET_DESC_FILE
from structure_processor import save_chain, save_structure, \
    produce_annotated_ab_structure
from patchdock_tools import output_patchdock_file

AB_STRUCT_SAVE_PATH = "data/annotated/{0}_AB.pdb"
AG_STRUCT_SAVE_PATH = "data/annotated/{0}_AG.pdb"
PATCHDOCK_SAVE_PATH = "data/annotated/{0}_patchdock.txt"


def annotate_and_save_test_structures(probs):
    chains = load_chains(TEST_DATASET_DESC_FILE)
    for i, (ag_chain, ab_h_chain, ab_l_chain, pdb_name) in enumerate(chains):
        p = probs[6*i:6*(i+1), :]
        ab_struct = produce_annotated_ab_structure(ab_h_chain, ab_l_chain, p)
        save_structure(ab_struct, AB_STRUCT_SAVE_PATH.format(pdb_name))

        save_chain(ag_chain, AG_STRUCT_SAVE_PATH.format(pdb_name))
        output_patchdock_file(ab_struct,
                              filename=PATCHDOCK_SAVE_PATH.format(pdb_name))
