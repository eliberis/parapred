# Parapred --- antibody binding residue prediction

## Install

Requirements:
   * Python 3.6+ (Python 2.7 for just running the predictor)
   * Packages listed in `requirements.txt`.
     Use `sudo pip install -r requirements.txt` to install (or without `sudo` if
     you're using a virtualenv). If you are using a Python installation manager,
     such as Anaconda or Canopy, follow their package installation instructions instead.

To install:
   * Run `python setup.py install` in the root directory. `parapred` executable
     should now be available (run `parapred --help` to check.

## Usage
```
➜  ~ parapred --help
Parapred - neural network-based paratope predictor.

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
```

## Example output
```
➜  ~ parapred cdr ASGYTFTSYWI
... omitted TensorFlow messages ... 
# ParaPred annotation of ASGYTFTSYWI
A 0.00640214
S 0.0170926
G 0.158654
Y 0.478457
T 0.652317
F 0.0494037
T 0.757008
S 0.936228
Y 0.827189
W 0.967675
I 0.0131835
----------------------------------
```
