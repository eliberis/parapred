# Author: Pietro Sormanni
## FUNCTIONS FOR SEQUENCE PROCESSING OF INPUTS ####

import sys
from .structure_processor import NUM_EXTRA_RESIDUES

def get_CDR_simple(sequence ,allow=set(["H", "K", "L"]),scheme='chothia',seqname='' \
                   ,cdr1_scheme={'H':range(26-NUM_EXTRA_RESIDUES,33+NUM_EXTRA_RESIDUES),'L':range(24-NUM_EXTRA_RESIDUES,35+NUM_EXTRA_RESIDUES)} \
                   ,cdr2_scheme={'H':range(52-NUM_EXTRA_RESIDUES,57+NUM_EXTRA_RESIDUES),'L':range(50-NUM_EXTRA_RESIDUES,57+NUM_EXTRA_RESIDUES)} \
                   ,cdr3_scheme={'H':range(95-NUM_EXTRA_RESIDUES,103+NUM_EXTRA_RESIDUES),'L':range(89-NUM_EXTRA_RESIDUES,98+NUM_EXTRA_RESIDUES)}) :
    '''
    From a VH or VL amino acid sequences returns the three CDR sequences as determined from the input numbering (scheme) and the given ranges.
    default ranges are Chothia CDRs +/- NUM_EXTRA_RESIDUES residues per side.
      requires the python module anarci - Available from http://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/ANARCI.php

    For other numbering schemes see also http://www.bioinf.org.uk/abs/#cdrdef
    Loop    Kabat          AbM    Chothia1    Contact2
    L1    L24--L34    L24--L34    L24--L34    L30--L36
    L2    L50--L56    L50--L56    L50--L56    L46--L55
    L3    L89--L97    L89--L97    L89--L97    L89--L96
    H1    H31--H35B   H26--H35B   H26--H32..34  H30--H35B
    H1    H31--H35    H26--H35    H26--H32    H30--H35
    H2    H50--H65    H50--H58    H52--H56    H47--H58
    H3    H95--H102   H95--H102   H95--H102   H93--H101

    For generic Chothia identification can set auto_detect_chain_type=True and use:
    cdr1_scheme={'H':range(26,34),'L':range(24,34)}
    cdr2_scheme={'H':range(52,56),'L':range(50,56)}
    cdr3_scheme={'H':range(95,102),'L':range(89,97)}
    '''
    try :
        import anarci
    except ImportError :
        raise Exception("\n**ImportError** function get_CDR_simple() requires the python module anarci\n Available from http://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/ANARCI.php\n\n")

    res_num_all=anarci.number(sequence, scheme=scheme, allow=allow)
    if not hasattr(res_num_all[0], '__len__') :
        sys.stderr.write( "*ERROR* in get_CDR_simple() anarci failed on %s -returned %s chaintype=%s\n" % (seqname,str(res_num_all[0]),str(res_num_all[1])))
        return None
    cdr1,cdr2,cdr3='','',''
    chain_type=res_num_all[1]
    sys.stdout.write( '%s chain_type= %s\n'%(seqname,chain_type))
    if hasattr(cdr1_scheme, 'keys') : # supports dictionary or OrderedDict as input type - assume all cdr ranges are like this
        if chain_type=='K' and chain_type not in cdr1_scheme : chain_type='L' # Kappa light chain to Lambda light chain for this purpose
        if chain_type not in cdr1_scheme :
            raise Exception("\n chain_type %s not in input cdr1_scheme\n" % (chain_type))
        cdr1_scheme=cdr1_scheme[chain_type]
        cdr2_scheme=cdr2_scheme[chain_type]
        cdr3_scheme=cdr3_scheme[chain_type]
    # extract CDR sequences
    for num_tuple,res in res_num_all[0] :
        if num_tuple[0] in cdr1_scheme: cdr1+=res # num_tuple[1] may be an insertion code, (e.g. 111B)
        elif num_tuple[0] in cdr2_scheme: cdr2+=res
        elif num_tuple[0] in cdr3_scheme: cdr3+=res

    # put in parapred formta
    cdrs={'CDR1':cdr1.replace('-',''),'CDR2':cdr2.replace('-',''),'CDR3':cdr3.replace('-','')}
    return cdrs

class FakeSeq :
    # mimic of the Biopython SeqRecord object, but in this way there is no need to have biopython installed
    def __init__(self,seq='',seq_id='',seq_name='',description='') :
        self.seq=seq
        self.id=seq_id
        self.name=seq_name
        self.description=description
    def __len__(self) :
        return len(self.seq)
    def __str__(self):
        return self.seq
    def __repr__(self):
        restr='FakeSeq:%s:' % (self.name)
        if len(self.seq)>20 :
            restr+=self.seq[:15]+'...'+self.seq[-2:]
        else : restr+=self.seq
        return restr
    def __getslice__(self,i,j):
        return self.seq[i:j]
    def __getitem__(self,y):
        return self.seq[y]
    def __add__(self,y):
        return FakeSeq(seq=self.seq+str(y),seq_id=self.id,seq_name=self.name,description=self.description)

def uniq(input_el):
    #given list/tuple it returns the unique elements (in the order they first appear)
    output = []
    for x in input_el:
        if x not in output:
            output.append(x)
    return output

amino_list1=['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',  'X']
def read_fasta(filename, return_as_dictionary=False, description_parser_function=None,use_seq_class=True, check_sequence=amino_list1, name_first_spilt=True):
    '''
    reads an input file in fasta format - returns the sequences
    '''
    sequences=[]
    ids=[]
    names=[]
    descriptions=[]
    remove=[]
    first_residues=[]
    nseq=0
    for line in open(filename) :
        if line[0]=='>' :
            line=line[1:].strip()
            if description_parser_function!=None :
                seqid, name, description= description_parser_function(line)
            elif name_first_spilt:
                name=line.split()[0]
                seqid=name
                description=line[len(name)+1:] # could be ''
            else :
                description=''
                name=line
                seqid=line
            if use_seq_class :
                sequences+=[ FakeSeq(seq='',seq_id=seqid,seq_name=name,description=description)]
            else :
                sequences+=['']
            if '|' in name : # for uniprot downloaded regions (e.g. without signal peptides) the last part may be the amino acid range
                putative_range=name.split('|')[-1]
                if '-' in putative_range :
                    try :
                        start,end=map(int, putative_range.split('-'))
                        first_residues+=[start]
                    except Exception : first_residues+=[1]
                else : first_residues+=[1]
            else : first_residues+=[1]
            names+=[name]
            ids+=[seqid]
            descriptions+=[description]
            if check_sequence!=None and nseq>0 :
                for j,aa in enumerate(sequences[-2]) :
                    if aa not in check_sequence :
                        sys.stderr.write("\n**ERROR** residue %d %s in sequence %d %s NOT STANDARD --> can't process\n" % (j+1,aa,nseq,names[-2]) )
                        sys.stderr.flush()
                        remove+=[nseq-1]
                        break
            nseq+=1
        elif line!='' and line!='\n' :
            sequences[-1]+=line.strip()
    if check_sequence!=None and nseq>0 :
        for j,aa in enumerate(sequences[-1]) :
            if aa not in check_sequence :
                sys.stderr.write("\n**ERROR** residue %d %s in sequence %d %s NOT STANDARD --> can't process\n" % (j+1,aa,nseq,names[-2]) )
                sys.stderr.flush()
                remove+=[nseq-1]
                break
    if remove!=[] : # remove sequences not containing only standard amino acids
        remove=uniq(remove)
        for j in sorted(remove,reverse=True) :
            sys.stderr.write("**** SKIPPING sequence %d %s  --> contains NOT STANDARD residues that cannot be processed\n" % (j+1,names[j]) )
            sys.stderr.flush()
            del sequences[j],ids[j],names[j],descriptions[j]
    if len(sequences)==0 :
        sys.stderr.write("\n**** WARNING *** NO VALID sequence in  %s\n" % (filename))
        sys.stderr.flush()
    if use_seq_class :
        return sequences,first_residues
    else :
        return sequences,ids,names,descriptions,first_residues
