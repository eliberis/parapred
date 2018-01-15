#!/usr/bin/env python
"""Parapred - neural network-based paratope predictor.

Parapred works on parts of antibody's amino acid sequence that correspond to the
CDR and two extra residues on the either side of it. The program will output
binding probability for every residue in the input. The program accepts two
kinds of input (see usage section below for examples):

(a) The full sequence of a VH or VL domain, or a larger stretch of the sequence
    of either the heavy or light chain comprising the CDR loops. (requires the
    additional module anarci for Chothia numbering of sequences, available at
    http://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/ANARCI.php).
    Command example: `parapred seq DIEMTQSPSSLSASVGDRVTITCR...`

(b) A fasta file with various antibody sequences, either light or heavy chains
    (requires anarci http://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/ANARCI.php)

(c) A Chothia-numbered PDB file together with antibody's heavy and light chain
    IDs (provided using `--abh` and `--abl` options). The program will overwrite
    the B-factor value in the PDB to a residue's binding probability.

    Multiple PDB files can be processed by specifying a description file (using
    the `--pdb-list` option), which is a CSV file containing columns `pdb`,
    `Hchain` and `Lchain` meaning PDB file name (.pdb extention will be appended
    if missing), heavy chain ID and light chain ID respectively. For example:

    pdb,Hchain,Lchain
    2uzi,H,L
    4leo,A,B

    Extra columns in the CSV file are allowed and will be ignored. A folder
    containing PDB files can be specified using the `--pdb-folder` option
    (defaults to the current directory).

(d) An amino acid sequence corresponding to a CDR with 2 extra residues on
    either side, e.g. `parapred cdr ARSGYYGDSDWYFDVGG`.

    Multiple CDR sequences can be processed at once by specifying a file,
    containing each sequence on a separate line (using the `--cdr-list` option).

Usage:
  parapred seq <sequence>
  parapred fasta <fasta_file>
  parapred pdb <pdb_file> [--abh <ab_h_chain_id>] [--abl <ab_l_chain_id>]
  parapred pdb --pdb-list <pdb_descr_file> [--pdb-folder=<path>]
  parapred cdr <cdr_seq>
  parapred cdr --cdr-list <cdr_file>
  parapred (-h | --help)

Options:
  -h --help                    Show this help.
  seq                          Takes the full amino acid sequence of a VH or VL
                               domain (requires anarci).
  fasta                        Takes a fasta-formatted file of amino acid
                               sequences corresponding or containing VH and VL
                               domains (requires anarci).
  pdb                          PDB-annotating mode. Replaces B-factor entries
                               with binding probabilities (in percentages). PDBs
                               must be Chothia-numbered.
  --abh <ab_h_chain_id>        Antibody's heavy chain ID [default: H].
  --abl <ab_l_chain_id>        Antibody's light chain ID [default: L].
  --pdb-list <pdb_descr_file>  List containing PDB file names and chain IDs in CSV format.
  --pdb-folder <path>          Path to a folder with PDB files [default: .].
  cdr <cdr_seq>                Given an individual CDR sequence with 2 extra residues
                               on either side, outputs binding probabilities for each residue.
  --cdr-list <cdr_file>        List containing CDR amino acid sequences, one per line.
"""
from __future__ import print_function

from docopt import docopt
from pandas import read_csv
import numpy as np
import pkg_resources

from .structure_processor import get_structure_from_pdb, extract_cdrs_from_structure, \
    residue_seq_to_one, produce_annotated_ab_structure, save_structure, aa_s, \
    seq_to_one_hot

from .full_seq_processor import get_CDR_simple, NUM_EXTRA_RESIDUES, read_fasta

MAX_CDR_LEN = 40
WEIGHTS = pkg_resources.resource_filename(__name__, "precomputed/weights.h5")

_model = None


def get_predictor():
    global _model
    from .model import ab_seq_model
    if _model is None:
        _model = ab_seq_model(MAX_CDR_LEN)
        _model.load_weights(WEIGHTS)
    return _model


def predict_sequence_probabilities(seqs):
    NUM_FEATURES = 28

    cdr_mats = []
    cdr_masks = []
    for seq in seqs:
        cdr_mat = seq_to_one_hot(seq)
        cdr_mat_pad = np.zeros((MAX_CDR_LEN, NUM_FEATURES))
        cdr_mat_pad[:cdr_mat.shape[0], :] = cdr_mat
        cdr_mats.append(cdr_mat_pad)

        cdr_mask = np.zeros((MAX_CDR_LEN, ), dtype=int)
        cdr_mask[:len(seq)] = 1
        cdr_masks.append(cdr_mask)

    cdrs = np.stack(cdr_mats)
    masks = np.stack(cdr_masks)

    model = get_predictor()
    probs = model.predict([cdrs, masks], batch_size=32)
    return np.squeeze(probs, axis=-1)


def process_single_pdb(pdb_file, ab_h_chain_id, ab_l_chain_id):
    structure = get_structure_from_pdb(pdb_file)
    model = structure[0]  # Structure only has one model
    for chain_id in [ab_h_chain_id, ab_l_chain_id]:
        if chain_id not in model:
            print("Chain {id} was not found in the file.".format(id=chain_id))
            exit(1)

    ab_h_chain = model[ab_h_chain_id]
    ab_l_chain = model[ab_l_chain_id]

    # Extract CDRs
    cdrs = {}
    cdrs.update(extract_cdrs_from_structure(ab_h_chain, "H"))
    cdrs.update(extract_cdrs_from_structure(ab_l_chain, "L"))

    seqs = []
    # Order is important to correctly map the results back into the PDB
    for cdr_name in ["H1", "H2", "H3", "L1", "L2", "L3"]:
        # Convert Residue entities to amino acid sequences
        seqs.append(residue_seq_to_one(cdrs[cdr_name]))

    probs = predict_sequence_probabilities(seqs)

    structure = produce_annotated_ab_structure(ab_h_chain, ab_l_chain,
                                               {"H": None, "L": None},
                                               np.expand_dims(probs, -1))
    save_structure(structure, pdb_file)


def process_multiple_pdbs(pdb_descr_file, pdb_folder):
    df = read_csv(pdb_descr_file)
    columns = list(df)
    if any(name not in columns for name in ["pdb", "Hchain", "Lchain"]):
        print("ERROR: The PDB description file must a CSV file with columns 'pdb', "
              "'Hchain', 'Lchain' describing the PDB file name, and H/L chain IDs.")
        return

    for _, entry in df.iterrows():
        pdb_name = entry['pdb']
        ab_h_chain_id = entry['Hchain']
        ab_l_chain_id = entry['Lchain']

        if not pdb_name.endswith(".pdb"):
            pdb_name = pdb_name + ".pdb"

        pdb_file = pdb_folder + "/" + pdb_name
        process_single_pdb(pdb_file, ab_h_chain_id, ab_l_chain_id)


def process_sequences(seqs):
    for s in seqs:
        for r in s:
            if r not in aa_s:
                raise ValueError("'{}' is not an amino acid residue. "
                                 "Only {} are allowed.".format(r, aa_s))

    prob = predict_sequence_probabilities(seqs)

    for i, s in enumerate(seqs):
        print("# ParaPred annotation of", s)
        for j, r in enumerate(s):
            print(r, prob[i, j])
        print("----------------------------------")


def process_single_cdr(cdr_seq):
    process_sequences([cdr_seq])


def process_cdr_sequences(cdr_file):
    with open(cdr_file, "r") as f:
        seqs = [s.rstrip() for s in f.readlines()]
        process_sequences(seqs)


# Full sequence processing by Pietro Sormanni
def process_full_VH_VL_sequence(full_sequence) :
    cdrs = get_CDR_simple( full_sequence.replace("\'",'').replace('\"','').replace(' ','').upper() )
    print(cdrs)
    for cdr_name in sorted(cdrs) :
        print("--- %-4s + %d residue per side ---"  % (cdr_name,NUM_EXTRA_RESIDUES))
        #print("-- Automatically extracted %-4s --" % (cdr_name))
        process_single_cdr(cdrs[cdr_name])


def process_fasta_file(fastafile) :
    sequences,_ = read_fasta(fastafile)
    for srec in sequences :
        print("\n> %s" % (srec.name))
        process_full_VH_VL_sequence( str(srec.seq) )


def main():
    arguments = docopt(__doc__, version='Parapred v1.0.1')
    if arguments["pdb"]:
        if arguments["<pdb_file>"]:
            process_single_pdb(arguments["<pdb_file>"],
                               arguments["--abh"], arguments["--abl"])
        else:
            process_multiple_pdbs(arguments["--pdb-list"],
                                  arguments["--pdb-folder"])
    elif arguments["cdr"]:
        if arguments["<cdr_seq>"]:
            process_single_cdr(arguments["<cdr_seq>"])
        else:
            process_cdr_sequences(arguments["--cdr-list"])
    elif arguments["seq"]:
        process_full_VH_VL_sequence(arguments["<sequence>"])
    elif arguments["fasta"]:
        process_fasta_file(arguments["<fasta_file>"])



if __name__ == '__main__':
    main()
