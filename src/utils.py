""" Helper script to generate a codon table """

import typing
from typing import Tuple
import itertools 
import numpy as np
from Bio.Seq import Seq
from collections import defaultdict

def generate_codon_tbl(): 
    """ Generate a codon table """
    BASES = "TCAG"
    codons = [a + b + c for a in BASES for b in BASES for c in BASES]

    # Use biopython to generate codon table
    aa = [str(Seq(j).translate()) for j in codons]

    # Mapping of codons to amino acids
    codon_to_aa = dict(zip(codons, aa))

    # NOTE: add 1 to the range items in both cases to leave room for padding
    # Token

    aa_unique = sorted(set(aa))
    # Map AA to number
    AA_NUMS = dict(zip(aa_unique, range(1, len(aa_unique) + 1)))
    
    # Map codons to numbers
    CODON_NUMS  = dict(zip(codon_to_aa, range(1, len(codons) + 1)))

    # Map codon number to AA numbers
    CODON_TO_AA_NUMS = {CODON_NUMS[k] : AA_NUMS[v] for k,v in codon_to_aa.items()}

    return AA_NUMS, CODON_NUMS, CODON_TO_AA_NUMS

def reverse_codon_to_aa(codon_to_aa_nums): 
    """ Reverse this mapping """
    aa_to_codons = defaultdict(lambda : [])
    for c, a in codon_to_aa_nums.items():
        aa_to_codons[a].append(c)
    return aa_to_codons

# CONSTANTS 
AA_NUMS, CODON_NUMS, CODON_TO_AA_NUMS  = generate_codon_tbl()
AA_TO_CODON_NUMS = reverse_codon_to_aa(CODON_TO_AA_NUMS)


# CONSTANTS 
AA_NUMS, CODON_NUMS, CODON_TO_AA_NUMS  = generate_codon_tbl()

# Names for dict
AA_SEQ = "aa_seq" 
CODON_SEQ = "codon_seq" 
SEQLEN= "seqlen" 

# Alphabet constants 
START_CODON = max(list(CODON_NUMS.values())) + 1
START_AA = max(list(AA_NUMS.values())) + 1

# Vocab Max
CODON_VOCAB_MAX = START_CODON
AA_VOCAB_MAX = START_AA

# Constants
PADDING = 0 

# Get test train splits
def get_train_val_test(obj : np.array, train_frac  : float, val_frac : float,
                       test_frac : float) -> Tuple[np.array, np.array, np.array]: 
    """get_train_val_test.

    Args:
        obj (np.array): obj
        train_frac (float): train_frac
        val_frac (float): val_frac
        test_frac (float): test_frac

    Returns:
        Tuple[np.array, np.array, np.array]:
    """
    num_examples = obj.shape[0]
    indices = np.arange(num_examples)

    # Shuffle indices
    np.random.shuffle(indices)

    # Get cutoffs
    train_cutoff = int(num_examples * train_frac)
    val_cutoff = int(train_cutoff + num_examples * val_frac)

    # Just go to end for test
    # test_cutoff= int(val_cutoff + num_examples * test_frac)

    train_indices  = indices[:train_cutoff]
    val_indices = indices[train_cutoff : val_cutoff]
    test_indices = indices[val_cutoff: ]

    train, val, test = obj[train_indices], obj[val_indices], obj[test_indices]
    return (train,val,test)

# Fasta helpers

def fasta_iter(file_ : str):
    """create iter object for fasta file"""
    f = open(file_)
    # We don't want the key, so we select x[1]
    faiter = (x[1] for x in itertools.groupby(f, lambda line: line[0] == ">"))

    sequences_dict = {} 
    for header in faiter:
        # Choose the first line iwthout the ">" char
        headerStr = header.__next__()[1:].strip()
        seq = "".join(s.strip() for s in faiter.__next__())

        yield headerStr, seq 

def get_full_fasta(file_ : str):
    """ convert iter object for fasta to dict """
    ret = {} 
    my_iter = fasta_iter(file_)
    for header, seq in my_iter: 
        header = header.split()[0].strip()
        seq = seq.strip()
        ret[header] = seq
    return ret

if __name__=="__main__": 
    generate_codon_tbl()


