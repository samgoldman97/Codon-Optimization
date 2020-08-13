""" 
Use case: python src/main.py  --data-file data/ecoli.heg.fasta
"""
import os
import sys
import re
import logging
import argparse
from typing import List

import torch
import numpy as np

from src import utils
#from models import * 
#from helpers import *

def get_args(): 
    """ Get arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true", default=False, 
                        help="If set, use gpu")
    parser.add_argument("--debug", action="store_true", default=False, 
                        help="If true, use debug mode")
    parser.add_argument("--out-prefix", action="store", default="results/temp", 
                        help="Prefix for output")
    parser.add_argument("--data-file", action="store", 
                        help="Input fasta data file")
    return parser.parse_args()

class DNADataset(torch.utils.data.Dataset): 
    """ TODO: Shear sequences into 100 token sentences """ 

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

    def __getitem__(self, idx) -> Tuple[np.array, np.arrray]: 
        """ Get item """

        return (self.codon_seqs[idx], self.aa_seqs[idx])

    def __len__(self): 
        return self.num_seqs


def main(args: argparse.Namespace): 
    """ Main method"""

    device = torch.device("gpu") if args.gpu else torch.device("cpu")

    fasta_file = args.data_file
    sequences = np.array(list(utils.get_full_fasta(fasta_file).values()))

    # Make into test, val,  train??
    train,val,test = utils.get_train_val_test(sequences, 0.8,0.1,0.1)

    # Log num data
    logging.info(f"Num sequences in train: {len(train)}")
    logging.info(f"Num sequences in val: {len(val)}")
    logging.info(f"Num sequences in test: {len(test)}")

    # Just load it all into memory
    train_dataset = DNADataset(train)
    val_dataset = DNADataset(val)
    test_dataset = DNADataset(test)
    import pdb
    pdb.set_trace()

    # TODO: Handle start codons?! 



if __name__=="__main__": 
    args = get_args()

    # Make directory for outfile
    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)

    # Setup logger
    logging_level = logging.INFO
    logging.basicConfig(level=logging_level, 
                        format='%(asctime)s %(levelname)s: %(message)s', 
                        handlers=[ logging.StreamHandler(sys.stdout), 
                                  logging.FileHandler(args.out_prefix + '_run.log') ]
                        )
    logging.info(f"Args: {args}")

    main(args)



    train, test, TEXT = load_csv_data(csv_out_file, device=device)
    (AA_LABEL, index_table, codon_to_aa, 
     codon_to_aa_index, mask_tbl) = build_helper_tables(TEXT, 
                                                        start_codons, device=device)

    ###### FREQUENCY MODEL #######
    # Unigram model 
    # zero_dict = make_n_gram_dict(train, 0, codon_to_aa_index, TEXT, AA_LABEL)

    # aa_params = {
    # 	"CODON_TO_AA" : codon_to_aa_index,
    # 	"N_GRAM_DICTS" : [zero_dict],
    # 	"N_LIST" : [0],
    # 	"WEIGHT_LIST" : [1],
    # 	"OUT_VOCAB" : len(TEXT.vocab.stoi),
    # 	"DEVICE" : device, 
    # 	"TEXT" : TEXT
    # }

    # model = FreqModel()
    # aa_compress = AA_NGRAM(aa_params)
    # model.to(device), aa_compress.to(device)

    # test_ac, train_ac = (joint_ppl_acc(test, model, device, aa_compress, TEXT, mask_tbl), 
    # 						joint_ppl_acc(train, model, device, aa_compress, TEXT, mask_tbl))
    # print("Train: ", train_ac)
    # print("Test: ", test_ac)

    # output_iterator_to_file(test, TEXT, outfile="temp.txt")

    # res = get_prediction_iter(test, model, aa_compress, mask_tbl, device)
    # output_list_of_res(res, TEXT, outfile="../outputs/predictions/temp.txt")

    ##### LSTM + BiLSTM AA ##### 


    aa_compress_params = {
        "CODON_TO_AA" : index_table,
        "EMBED_DIM" : index_table.shape[1],
        "HIDDEN_LEN" : 50, 
        "NUM_LAYERS" : 1, 
        "LSTM_DROPOUT" : 0.1,
        "BIDIRECTIONAL" : True, 
        "START_INDEX" : TEXT.vocab.stoi["atg"], 
        "DEVICE" : device
    }

    model_params = {
        "VOCAB_SIZE" : len(TEXT.vocab),
        "EMBED_DIM" : 50,# None,
        "OUT_VOCAB": len(TEXT.vocab),
        "HIDDEN_LEN" : 50,
        "NUM_LAYERS" : 2,
        "LINEAR_DROPOUT" : 0.1,
        "LSTM_DROPOUT" : 0.1,    
        "AA_COMPRESS_SIZE" : (aa_compress_params["HIDDEN_LEN"] * 
                              (2 if aa_compress_params["BIDIRECTIONAL"] else 1)),
        "TEACHER_FORCE" : 1, 
        "DEVICE": device
    }

    model = NNLM(model_params)
    aa_compress = AA_BILSTM(aa_compress_params)
    model.to(device), aa_compress.to(device)


    train_params = {
        "num_epochs":1, 
        "lr":1e-3,  
        "weight_decay":0,
        "device":device, 
        "grad_clip": 100, 
        "plot_loss" : True,
        "TEACHER_FORCE" : 1
    }

    optimizer = torch.optim.Adam(model.parameters(), lr=train_params["lr"],
                                 weight_decay=train_params["weight_decay"])
    optimizer_aa = torch.optim.Adam(aa_compress.parameters(), 
                                    lr=train_params["lr"], 
                                    weight_decay=train_params["weight_decay"])
    # Number of training rounds
    for i in range(5): 
        train_model(train, model, aa_compress, TEXT, device, train_params,
                    optimizer, optimizer_aa)
        test_ac, train_ac = (joint_ppl_acc(test, model, device, aa_compress, TEXT, mask_tbl), 
                             joint_ppl_acc(train, model, device, aa_compress, TEXT, mask_tbl))
        print("Train: ", train_ac)
        print("Test: ", test_ac)

    res = get_prediction_iter(test, model, TEXT, aa_compress, mask_tbl, device)
    output_list_of_res(res, TEXT, outfile="../outputs/predictions/temp.txt")






