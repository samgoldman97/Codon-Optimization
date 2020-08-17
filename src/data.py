from typing import List
import numpy as np
import torch
import re

from src import utils

class DNADataset(torch.utils.data.Dataset): 
    """ Transcript sequence """ 
    def __init__(self, seqlist: List[str]): 
        """ Init """
        super(DNADataset, self).__init__()
        self.seqlist = seqlist

        # Convert seq list into array of numpy items
        self.tokenize = lambda x : re.findall('.{%d}' % 3, x)

        # Sequences as triples
        self.tokenized_seqs = [self.tokenize(j) for j in self.seqlist]

        # Sequences as codon integers
        self.codon_seqs = [np.array([utils.CODON_NUMS[i] for i in j]) for j in self.tokenized_seqs]

        # Sequences as amino acid integers
        # NOTE: Hardcode first amino acid position as methionine always
        self.aa_seqs = [
            np.array([utils.CODON_TO_AA_NUMS[i]  
             if index > 0 else utils.AA_NUMS['M'] 
             for index, i in enumerate(j)]) for j in self.codon_seqs
        ]
        self.num_seqs = len(self.seqlist)

    def get_aa_list(self): 
        """Return the array that holds all the sequences tokenized to AA's """
        return self.aa_seqs

    def get_codon_list(self): 
        """ Return the array that holds all the sequences tokenized to codons"""
        return self.codon_seqs

    def export_codon_seq(self, file_): 
        """ Export codon seq lists to a file """ 
        with open(file_, "w") as fp: 
            fp.write("\n".join(self.seqlist))

    def __getitem__(self, idx : int): 
        """ Get Tuple[codon numpy seq, aa numpy seq, seqlen] 

        Seqlen does NOT include start token

        """
        return (self.codon_seqs[idx], self.aa_seqs[idx], len(self.aa_seqs[idx]))

    def __len__(self): 
        """ Length """
        return self.num_seqs

def dna_collate(x : List): 
    """ dna collate function 

    Return: 
        dict containing aa tensor, codon tensor, and lengths
    """
    lengths = [i[2] for i in x]
    max_length = max(lengths)

    new_codons = []
    new_aas = []

    # Pad then to tensor 
    # Add start token to all examples
    for codon_seq, aa_seq, seqlen in x: 
        new_codon_seq = np.zeros(max_length + 1)
        new_codon_seq[1:seqlen+1] = codon_seq  
        new_codon_seq[0] = utils.START_CODON

        new_aa_seq = np.zeros(max_length + 1)
        new_aa_seq[1:seqlen+1] = aa_seq
        new_aa_seq[0] = utils.START_AA

        new_aas.append(new_aa_seq)
        new_codons.append(new_codon_seq)

    codon_out = torch.from_numpy(np.vstack(new_codons)).long()
    aa_out = torch.from_numpy(np.vstack(new_aas)).long()

    return {utils.AA_SEQ: aa_out, utils.CODON_SEQ : codon_out, utils.SEQLEN : lengths}

