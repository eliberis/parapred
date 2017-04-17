import numpy as np
from math import sin, cos, sqrt
from Bio.PDB import Chain, Residue
from typing import Callable

def get_3d_rotation_matrix(x, y, z):
    cx, cy, cz = cos(x), cos(y), cos(z)
    sx, sy, sz = sin(x), sin(y), sin(z)

    return np.matrix([[cz*cy, -sy*sx*cz - sz*cx, -sy*cx*cz + sz*sx],
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


def backbone_rmsd(lig_chain, transformer, interface_pred=None):
    interface_pred = interface_pred if interface_pred else lambda x: True
    num_atoms = 0
    sq_diff = 0
    for res in lig_chain:
        if not interface_pred(res): continue
        for atom_type in ['C', 'N', 'O', 'CA']:
            if not atom_type in res: continue

            atom_coord = np.array(res[atom_type].coord)
            atom_coord_new = transformer(atom_coord)

            sq_dist = np.sum(np.array(atom_coord-atom_coord_new)**2)
            sq_diff += sq_dist
            num_atoms += 1
    return sqrt(sq_diff / num_atoms)


def process_transformations(trans_file, antigen_chain):
    f = open(trans_file, "r")
    lines = f.readlines()
    f.close()

    for line in lines:
        transf_params = tuple([float(x) for x in line.split()])
        transformer = param_transform(*transf_params)
        print(backbone_rmsd(antigen_chain, transformer))

