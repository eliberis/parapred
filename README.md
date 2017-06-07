# Antibody binding residue prediction
CST Part III Research Project.

Prerequisites: 
* Create a new Python virtual environment and install required 
packages using `pip install -r requirements.txt`
* Extract all PDB files from the [SabDab dataset](http://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/data/sabdab.zip) into the `data/pdbs` folder.
* Run the software with `python main.py` (you can select an appropriate entry 
point by point my modifying the last line of `main.py`)

File guide:
* `main.py`: Entry points for running software for different tasks.
* `model.py`: Keras deep learning models.
* `structure_processor.py`: Methods for processing amino acid sequences.
* `data_provider.py`: Methods for processing PDB files and encoding data as 
matrices.
* `evaluation.py` and `plotting.py`: Evaluation scripts and plotting routines.
* `patchdock_tools.py`: Patchdock integration (constraint output).
* `tests.py`: Unit tests.

Note that this a `git` repository. You can switch to a version of software 
for processing structural information by running `git checkout 
nhood-struct-info`. If you're feeling adventurous you can explore the 
following branches:
* `tf-fold`: sequential model implementation in TensorFlow Fold, which allows
 to run models without padding and creating more interesting graph/tree 
 processing algorithms. 
* `struct_info`: also a structural information-enabled model, but using a 
point-cloud approach (inspired by PointNet).
* `epitope-pred`: a model for epitope prediction!
* `epitope-pred-struct-info`: epitope prediction meets neighbourhood-based 
graph processing.
* `separate-models`: use separate models for each of the 6 CDR sequences.

