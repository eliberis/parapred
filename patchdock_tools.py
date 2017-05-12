import numpy as np
import copy
from Bio.PDB import *
from structure_processor import extended_epitope
from math import sin, cos, sqrt
from itertools import chain, repeat

def get_3d_rotation_matrix(x, y, z):
    cx, cy, cz = cos(x), cos(y), cos(z)
    sx, sy, sz = sin(x), sin(y), sin(z)

    return np.array([[cz*cy, -sy*sx*cz - sz*cx, -sy*cx*cz + sz*sx],
                     [sz*cy, -sy*sx*sz + cx*cz, -sy*cx*sz - sx*cz],
                     [   sy,             cy*sx,             cy*cx]])


def param_transform(ang_x, ang_y, ang_z, t_x, t_y, t_z):
    rot = get_3d_rotation_matrix(ang_x, ang_y, ang_z)
    transl = np.array([t_x, t_y, t_z])
    return lambda point: rot.dot(point) + transl


def _write_pd_constraint(f, chain_id, res):
    chain_id = chain_id if chain_id is not None else ''
    res_num = str(res.id[1]) + \
              (res.id[2] if res.id[2] != ' ' else '')
    f.write(res_num + ' ' + chain_id + '\n')


def output_patchdock_ab_constraint(structure, filename="ab-patchdock.txt",
                                   cutoff=50.00):
    model = structure[0]
    with open(filename, "w") as f:
        for chain in model.get_chains():
            for res in chain:
                if any(a.get_bfactor() > cutoff for a in res):
                    _write_pd_constraint(f, chain.id, res)


def output_patchdock_ag_constraint(ext_epitope, chain_id,
                                   filename="ag-patchdock.txt"):
    with open(filename, "w") as f:
        for res in ext_epitope:
            _write_pd_constraint(f, chain_id, res)


def transformed_chain(chain, transformer):
    chain = copy.deepcopy(chain)
    for residue in chain:
        for atom in residue:
            atom.coord = transformer(atom.coord)
    return chain


def backbone_rmsd(orig_ag_chain, trans_ag_chain):
    num_atoms = 0
    sq_diff = 0
    for or_res, tr_res in zip(orig_ag_chain, trans_ag_chain):
        for atom_type in ['C', 'N', 'O', 'CA']: # Heavy atoms?
            if not atom_type in or_res: continue
            diffs = or_res[atom_type].coord - tr_res[atom_type].coord
            sq_dist = np.sum(np.square(diffs))
            sq_diff += sq_dist
            num_atoms += 1
    return sqrt(sq_diff / num_atoms)


def interface_pairs(ag_chain, ab_h_chain, ab_l_chain, dist=5.0):
    ag_search = NeighborSearch(Selection.unfold_entities(ag_chain, 'A'))

    pairs = []
    # Augment residue iterator with chain id
    ab_h_chain = zip(repeat(ab_h_chain.id), ab_h_chain)
    ab_l_chain = zip(repeat(ab_l_chain.id), ab_l_chain)

    for ab_cid, ab_res in chain(ab_h_chain, ab_l_chain):
        ag_residues = []

        # For all atoms in the AB residue, find AG residues within reach
        for atom in ab_res:
            rs = ag_search.search(atom.coord, dist, level='R')
            ag_residues.extend(rs)

        if not ag_residues: continue
        ag_residues = Selection.uniqueify(ag_residues)
        for ag_res in ag_residues:
            ab_res_repr = (ab_cid, ) + ab_res.id
            ag_res_repr = (ag_chain.id, ) + ag_res.id
            pairs.append((ab_res_repr, ag_res_repr))

    return pairs


def calculate_f_nat(orig_iface, new_iface):
    pairs_recreated = 0
    for orig_pair in orig_iface:
        if orig_pair in new_iface:
            pairs_recreated += 1
    return pairs_recreated / len(orig_iface)


def decoy_class(f_nat, l_rmsd, i_rmsd):
    if f_nat >= 0.5 and l_rmsd <= 1.0 or i_rmsd <= 1.0: return 'high'
    if ((f_nat >= 0.3 and f_nat < 0.5) and (l_rmsd < 5.0 or i_rmsd < 2.0)) \
        or (f_nat >= 0.5 and l_rmsd > 1.0 and i_rmsd > 1.0): return 'med'
    if ((f_nat >= 0.1 and f_nat < 0.3) and (l_rmsd < 10.0 or i_rmsd < 4.0)) \
        or (f_nat >= 0.3 and l_rmsd > 5.0 and i_rmsd > 2.0): return 'low'
    return None


def process_transformations(trans_file, ag_chain, ab_h_chain, ab_l_chain,
                            limit=200):
    f = open(trans_file, "r")
    lines = f.readlines()
    f.close()

    num_decoys = {'high': 0, 'med': 0, 'low': 0}

    # A bit wrong?
    orig_iface = interface_pairs(ag_chain, ab_h_chain, ab_l_chain)
    orig_ext_epi = extended_epitope(ag_chain, ab_h_chain, ab_l_chain, cutoff=10.0)

    for i, line in enumerate(lines):
        if i >= limit: break
        transf_params = tuple([float(x) for x in line.split()])
        transformer = param_transform(*transf_params)

        new_ag_chain = transformed_chain(ag_chain, transformer)
        new_iface = interface_pairs(new_ag_chain, ab_h_chain, ab_l_chain)

        f_nat = calculate_f_nat(orig_iface, new_iface)
        if f_nat < 0.1:  # Will have no class
            continue

        l_rmsd = backbone_rmsd(ag_chain, new_ag_chain)
        new_ext_epi = transformed_chain(orig_ext_epi, transformer)
        i_rmsd = backbone_rmsd(orig_ext_epi, new_ext_epi)

        cls = decoy_class(f_nat, l_rmsd, i_rmsd)
        if cls is not None:
            # print("Decoy {} of PDB {}".format(i, trans_file))
            # print("Class:", cls)
            # print("f_nat:", f_nat)
            # print("L_RMSD:", l_rmsd)
            # print("I_RMSD:", i_rmsd)
            # print("----------------")
            num_decoys[cls] += 1

    # print(num_decoys)

    # Return highest quality decoy
    if num_decoys['high'] > 0: return 'high'
    elif num_decoys['med'] > 0: return 'med'
    elif num_decoys['low'] > 0: return 'low'
    else: return None
