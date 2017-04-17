from data_provider import load_chains, TEST_DATASET_DESC_FILE
from structure_processor import save_chain, save_structure, \
    produce_annotated_ab_structure, extended_epitope
from patchdock_tools import output_patchdock_ab_constraint, output_patchdock_ag_constraint

AB_STRUCT_SAVE_PATH = "data/annotated/{0}_AB.pdb"
AG_STRUCT_SAVE_PATH = "data/annotated/{0}_AG.pdb"
AB_PATCHDOCK_SAVE_PATH = "data/annotated/{0}_ab_patchdock.txt"
AG_PATCHDOCK_SAVE_PATH = "data/annotated/{0}_ag_patchdock.txt"


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
        ext_epi = extended_epitope(ag_chain, ab_h_chain, ab_l_chain)
        output_patchdock_ag_constraint(ext_epi, ag_chain.id, filename=agpd_fname)
