""" 
Use case: python src/main.py  --data-file data/ecoli.heg.fasta


Future goals: 
- Shear sequences into 100 tokens or partition them for more batching
- Try to incorporate more genomes to jointly train
- Use RNA seq data to get high expression transcripts rather than using
  reference


"""
import os
import sys
import re
import logging
import json
import argparse
import typing
import copy
from typing import List, Tuple
from collections import defaultdict
import random

import torch
from torch.utils.data import DataLoader
import numpy as np

# Import from this module
import utils, data, models

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
    parser.add_argument("--run-baselines", action="store_true", 
                        default=False, help="If true, run baselines")

    #### Train args ####

    parser.add_argument("--lr", action="store",
                        default=1e-3, type=float, help="Learning rate")
    parser.add_argument("--epochs", action="store",
                        default=10, type=int, help="Num epochs")
    parser.add_argument("--batch-size", action="store",
                        default=10, type=int, help="Batch size")

    ####  Model args ####

    # Joint model
    parser.add_argument("--joint-dropout", action="store",
                        default=0.1, type=float, help="Joint mlp dropout")
    parser.add_argument("--joint-hidden", action="store",
                        default=30, type=int, help="Joint distribution size for mlp")
    # Codon model
    parser.add_argument("--codon-model-name", action="store", 
                        type=str, help="Codon model name", 
                        choices = models.CODON_MODELS, default="lstm")
    parser.add_argument("--codon-lstm-layers", action="store",
                        default=2, type=int, help="Layers for codon lstm")
    parser.add_argument("--codon-hidden", action="store",
                        default=50, type=int, 
                        help="size of codon hidden dimmension")
    parser.add_argument("--codon-dropout", action="store",
                        default=0.2, type=float, help="Dropout for codon model")

    # AA model
    parser.add_argument("--aa-model-name", action="store", 
                        type=str, help="AA model name", 
                        choices = models.AA_MODELS, default="bilstm")
    parser.add_argument("--aa-bilstm-layers", action="store",
                        default=2, type=int, help="Layers for aa bilstm")
    parser.add_argument("--aa-hidden", action="store",
                        default=50, type=int, 
                        help="size of aa hidden dimmension")
    parser.add_argument("--aa-onedirect", action="store_true",
                        default=False, help="If true, use a one-d LSTM")
    parser.add_argument("--aa-dropout", action="store",
                        default=0.2, type=float, help="Dropout for aa")
    return parser.parse_args()


def make_n_gram_dict(dataset, n=3):  
    """ Helper function to create a frequency default dictionary

    Args: 
        dataset: Training bucket iterator
        n: Number of amino acids to each side of AA (e.g. 0 is unigram, 1 is trigram)
        amino_acid_conversion: index_table converting the codon index to AA index

    Returns: 
        default_dict: dictionary mapping a sequence of amino acids to probability over codons
    """

    # Map each context to a default dictionary of 0 that holds the probability
    # of every codon outcome
    n_gram_dict  = defaultdict(lambda : defaultdict(lambda : 0))

    # How far to look back 
    half_n = n // 2 
    
    # NOTE: This seq has --no-- start token or padding
    for codon_seq, aa_seq, seqlen in dataset: 
        for index, true_codon in enumerate(codon_seq): 
            context_start = max(0, index - half_n)
            # Add 1 because of python zero indexing 
            context_end = min(seqlen, index + half_n + 1)
            aa_context = aa_seq[context_start:context_end] 
            aa_context_str = str(list(aa_context))
            n_gram_dict[aa_context_str][true_codon] += 1

    # Now normalize
    for k, v in n_gram_dict.items(): 
        # total counts for this context 
        p_denom = sum(list(v.values()))

        # Renormalize 
        for codon, count in v.items(): 
            v[codon] = count / p_denom

    return n_gram_dict

def ngram_metrics(args, dataset, n_gram_dict_list, n_list, weights_list): 
    """ Compute accuracy and ppl for ngrams 
    Args: 
        args: Namespace args
        dataset: DNADataset 
        n_gram_dict_list: List of ngram dictionaries generated that correspond
            to n_list and should be weighted
        n_list: Corresponding context size (e.g. unigram n=1, trigrams n=3)
        weights_list: fraction of weighting

    Returns:
        dict of results for accuracy and ppl
    """

    total_predictions = 0   
    total_loss = 0 
    total_correct = 0 
    weights_list = [i / sum(weights_list) for i in weights_list]

    for codon_seq, aa_seq, seqlen in dataset: 
        for index, true_codon in enumerate(codon_seq): 

            # Hold weighted sum of ngram dict predictions
            prob_outputs = defaultdict(lambda : 0 )
            true_aa = aa_seq[index]
            unif = 1 / len(utils.AA_TO_CODON_NUMS[true_aa])
            for n_gram_dict, n, weight in zip(n_gram_dict_list, 
                                              n_list,
                                              weights_list): 
                # How far to look back 
                half_n = n // 2 
                context_start = max(0, index - half_n)
                # Add 1 because of python zero indexing 
                context_end = min(seqlen, index + half_n + 1)
                aa_context = aa_seq[context_start:context_end] 
                aa_context_str = str(list(aa_context))

                unobserved = False
                # Add_weight is unif!
                if sum(list(n_gram_dict[aa_context_str])) == 0: 
                    unobserved=True

                for k,v in n_gram_dict[aa_context_str].items(): 
                    # If we havenn't seen this context in this dict, add unif 
                    if unobserved: 
                        prob_outputs[k] += (unif * weight)
                    else: 
                        prob_outputs[k] += (v * weight)

            # Random choice on codon if unobserved context
            if sum(list(prob_outputs.values())) == 0.0:
                predicted_codon = random.choice(utils.AA_TO_CODON_NUMS[true_aa])
                # Probability of the outcome
                p_outcome = unif
            else:
                # extract maximum codon 
                predicted_codon =  max(dict(prob_outputs), 
                                       key=lambda x : prob_outputs[x])
                # Probability of the outcome
                p_outcome = prob_outputs[true_codon]

            total_predictions += 1
            total_loss += - np.log(p_outcome)

            if predicted_codon == true_codon: 
                total_correct += 1

    ppl = np.exp(total_loss / total_predictions)
    acc = total_correct / total_predictions 
    res = {"acc": acc, "ppl" : ppl}
    return res

def ngram_output(args, dataset, n_gram_dict_list, n_list, 
                 weights_list, sample=True, export_file=None): 
    """ Generate output sequences for a dictionary of ngrams.

    Optionally save to file

    Args: 
        args: Namespace args
        dataset: DNADataset 
        n_gram_dict_list: List of ngram dictionaries generated that correspond
            to n_list and should be weighted
        n_list: Corresponding context size (e.g. unigram n=1, trigrams n=3)
        weights_list: fraction of weighting
        sample (bool) : If true sample, else take argmax
        export_file (str) : Export file string

    Returns:
        AA sequence for output
    """

    weights_list = [i / sum(weights_list) for i in weights_list]
    codon_out_list = []
    for codon_seq, aa_seq, seqlen in dataset: 
        codon_out =  []
        for index, true_codon in enumerate(codon_seq): 
            # Hold weighted sum of ngram dict predictions
            prob_outputs = defaultdict(lambda : 0 )
            true_aa = aa_seq[index]
            unif = 1 / len(utils.AA_TO_CODON_NUMS[true_aa])
            for n_gram_dict, n, weight in zip(n_gram_dict_list, 
                                              n_list,
                                              weights_list): 
                # How far to look back 
                half_n = n // 2 
                context_start = max(0, index - half_n)
                # Add 1 because of python zero indexing 
                context_end = min(seqlen, index + half_n + 1)
                aa_context = aa_seq[context_start:context_end] 
                aa_context_str = str(list(aa_context))

                unobserved = False
                # Add_weight is unif!
                if sum(list(n_gram_dict[aa_context_str])) == 0: 
                    unobserved=True

                for k,v in n_gram_dict[aa_context_str].items(): 
                    # If we havenn't seen this context in this dict, add unif 
                    if unobserved: 
                        prob_outputs[k] += (unif * weight)
                    else: 
                        prob_outputs[k] += (v * weight)

            codon_probs = np.zeros(utils.CODON_VOCAB_MAX)
            for k,v  in prob_outputs.items(): 
                codon_probs[k] = v 

            # If we have zero in this, then set it to unif
            if np.sum(codon_probs) == 0.0: 
                for k in utils.AA_TO_CODON_NUMS[true_aa]: 
                    codon_probs[k] = unif

            # Sample or take a hard argmax
            if sample: 
                out_codon = np.random.choice(np.arange(utils.CODON_VOCAB_MAX),p = codon_probs)
            else: 
                out_codon =  np.argmax(codon_probs)


            # Convert this to a
            codon_out.append(out_codon)
        # Convert to string
        new_seq = "".join([utils.NUM_TO_CODON[s] for s in codon_out])
        codon_out_list.append(new_seq)

    if export_file is not None: 
        with open(export_file, "w") as fp: 
            fp.write("\n".join(codon_out_list))
    return codon_out_list

def baseline_model(args : argparse.Namespace): 
    """ Run ngram frequency based models"""

    # Train, val, test 
    train_dataset, val_dataset, test_dataset = get_train_val_test(args)
    unigram_dict = make_n_gram_dict(train_dataset, n=1)
    trigram_dict = make_n_gram_dict(train_dataset, n=3)
    fivegram_dict = make_n_gram_dict(train_dataset, n=5)

    for ngram_dict, n in zip([unigram_dict, trigram_dict, fivegram_dict],
                             [1,3,5]): 
        train_metrics = ngram_metrics(args, train_dataset,
                                    n_gram_dict_list=[ngram_dict], n_list=[n],
                                    weights_list=[1])
        val_metrics = ngram_metrics(args, val_dataset, 
                                    n_gram_dict_list=[ngram_dict], n_list=[n],
                                    weights_list=[1])
        test_metrics= ngram_metrics(args, test_dataset, 
                                    n_gram_dict_list=[ngram_dict], n_list=[n],
                                    weights_list=[1])
        for names, metrics in zip(["train", "val", "test"], 
                                  [train_metrics, val_metrics, test_metrics]): 
            logging.info(f"{n}-gram on dataset {names} results:{json.dumps(metrics, indent=2)}")
    
    # Run joint weighting of these
    joint_dicts = [unigram_dict, trigram_dict, fivegram_dict]
    n_list = [1,3,5]
    weight_list = [0.7, 0.3, 0]
    train_metrics = ngram_metrics(args, train_dataset,
                                n_gram_dict_list=joint_dicts, n_list=n_list,
                                weights_list=weight_list)
    val_metrics = ngram_metrics(args, val_dataset, 
                                n_gram_dict_list=joint_dicts, n_list=n_list,
                                weights_list=weight_list)
    test_metrics= ngram_metrics(args, test_dataset, 
                                n_gram_dict_list=joint_dicts, n_list=n_list,
                                weights_list=weight_list)

    # Export to file
    train_out = ngram_output(args, train_dataset,
                             n_gram_dict_list=[ngram_dict], n_list=n_list,
                             weights_list=weight_list, 
                             export_file=f"{args.out_prefix}_train_ngram.txt")
    val_out = ngram_output(args, val_dataset,
                           n_gram_dict_list=[ngram_dict], n_list=n_list,
                           weights_list=weight_list, 
                           export_file=f"{args.out_prefix}_val_ngram.txt")
    test_out= ngram_output(args, test_dataset,
                           n_gram_dict_list=[ngram_dict], n_list=n_list,
                           weights_list=weight_list, 
                           export_file=f"{args.out_prefix}_test_ngram.txt")

    for names, metrics in zip(["train", "val", "test"], 
                              [train_metrics, val_metrics, test_metrics]): 
        logging.info(f"1-3-5-gram on dataset {names} results:{json.dumps(metrics, indent=2)}")

def get_train_val_test(args): 
    """ Get train, val, test dna datasets """
    fasta_file = args.data_file
    sequences = np.array(list(utils.get_full_fasta(fasta_file).values()))

    # Make into test, val, train??
    train,val,test = utils.get_train_val_test(sequences, 0.8,0.1,0.1)


    # Log num data
    logging.info(f"Num sequences in train: {len(train)}")
    logging.info(f"Num sequences in val: {len(val)}")
    logging.info(f"Num sequences in test: {len(test)}")

    # Just load it all into memory
    train_dataset = data.DNADataset(train)
    val_dataset = data.DNADataset(val)
    test_dataset = data.DNADataset(test)

    # Write true seqs to file
    train_dataset.export_codon_seq(f"{args.out_prefix}_true_train_seqs.txt")
    val_dataset.export_codon_seq(f"{args.out_prefix}_true_val_seqs.txt")
    test_dataset.export_codon_seq(f"{args.out_prefix}_true_test_seqs.txt")

    return train_dataset, val_dataset, test_dataset


def neural_model(args: argparse.Namespace): 
    """ Main method"""

    # Train, val, test 
    train_dataset, val_dataset, test_dataset = get_train_val_test(args)

    # Build model 
    model = models.CodonModel(args) 
    loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                        shuffle=True, collate_fn=data.dna_collate)

    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_losses, val_losses = [], []
    best_loss = get_test_loss(model, args, val_dataset)
    best_model = copy.deepcopy(model)

    generate_out_seqs(model, args, train_dataset, 
                      save_file = f"{args.out_prefix}_train_gen.txt")

    for epoch in range(args.epochs): 
        epoch_losses = []
        model = model.train()

        if args.gpu: 
            model = model.cuda()

        for batch in loader: 

            # Convert batch to gpu 
            if args.gpu: 
                batch[utils.CODON_SEQ] = batch[utils.CODON_SEQ].cuda()
                batch[utils.AA_SEQ] = batch[utils.AA_SEQ].cuda()

            opt.zero_grad()

            # Model
            preds = model(batch)

            # Mask all codon seqs
            targs = batch[utils.CODON_SEQ][:, 1:]
            seqlens = batch[utils.SEQLEN]

            # Only select non padding
            loss_select= (targs != utils.PADDING)

            # Subset
            preds_subset = preds[loss_select]
            targs_subset = targs[loss_select]
            loss = loss_fn(preds_subset, targs_subset)
            loss.backward()

            # Step 
            opt.step()

            train_losses.append(loss.item())
            epoch_losses.append(loss.item())

        #  Compute loss just for this epoch 
        train_loss = np.mean(epoch_losses)
        val_loss = get_test_loss(model, args, val_dataset)
        val_losses.append(val_loss)

        if val_loss < best_loss:
            logging.info(f"Found new best model at epoch {epoch}")
            best_loss = val_loss
            best_model = copy.deepcopy(model)

        logging.info(f"Done with epoch #{epoch + 1}")
        logging.info(f"Train loss: {train_loss}")
        logging.info(f"Val loss: {val_loss}")

    # Get best model based on val
    model = best_model

    # Compute metrics
    train_metrics = compute_metrics(model, args, train_dataset)
    val_metrics = compute_metrics(model, args, val_dataset)
    test_metrics = compute_metrics(model, args, test_dataset)

    for names, metrics in zip(["train", "val", "test"], 
                              [train_metrics, val_metrics, test_metrics]): 
        logging.info(f"Dataset {names} results:{json.dumps(metrics, indent=2)}")

    logging.info(f"Generating generated results:") 

    # Generate output sequences
    generate_out_seqs(model, args, train_dataset, 
                      save_file = f"{args.out_prefix}_train_gen.txt")
    generate_out_seqs(model, args, val_dataset, 
                      save_file = f"{args.out_prefix}_val_gen.txt")
    generate_out_seqs(model, args, test_dataset, 
                      save_file = f"{args.out_prefix}_test_gen.txt")

def compute_metrics(model, args, dataset): 
    """ Compute ppl accuracy using the model given """
    losses = []
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    loader = DataLoader(dataset, batch_size=args.batch_size, 
                        shuffle=False, collate_fn=data.dna_collate)
    model = model.eval()
    total_acc =  0
    if args.gpu: 
        model = model.cuda()
    with torch.no_grad(): 
        for batch in loader: 

            # Convert batch to gpu 
            if args.gpu: 
                batch[utils.CODON_SEQ] = batch[utils.CODON_SEQ].cuda()
                batch[utils.AA_SEQ] = batch[utils.AA_SEQ].cuda()

            # Model
            preds = model(batch)
            # Mask all codon seqs
            targs = batch[utils.CODON_SEQ][:, 1:]
            seqlens = batch[utils.SEQLEN]

            # Only select non padding
            loss_select= (targs != utils.PADDING)

            # Subset
            preds_subset = preds[loss_select]
            targs_subset = targs[loss_select]
            loss = loss_fn(preds_subset, targs_subset).cpu().numpy()
            losses.extend(list(loss))

            # Accuracy
            total_acc += torch.sum(preds_subset.argmax(1) == targs_subset).item()

    total_loss = np.sum(losses)
    num_preds = len(losses)
    ppl = np.exp(total_loss / num_preds)
    total_acc = total_acc / num_preds
    res = {"ppl" : ppl, 
           "acc" : total_acc}
    return res

def generate_out_seqs(model, args, dataset, 
                      save_file = None): 
    """ Generate output sequences for a deep model """  

    loader = DataLoader(dataset, batch_size=args.batch_size, 
                        shuffle=False, collate_fn=data.dna_collate)
    total_seqs = []
    model = model.eval()
    if args.gpu: 
        model = model.cuda()
    with torch.no_grad(): 
        for batch in loader: 
            # Convert batch to gpu 
            if args.gpu: 
                batch[utils.CODON_SEQ] = batch[utils.CODON_SEQ].cuda()
                batch[utils.AA_SEQ] = batch[utils.AA_SEQ].cuda()

            # Model
            new_seqs = model.generate_codon_seqs(batch) 
            total_seqs.extend(new_seqs)

            if args.debug:
                # Break after 1
                break

    # Convert to string / codons
    new_seqs = ["".join([utils.NUM_TO_CODON[s] for s in seq])  
                for seq in total_seqs]

    if save_file is not None: 
        with open(save_file, "w") as fp: 
            fp.write("\n".join(new_seqs))

def get_test_loss(model, args, dataset): 
    """ Compute test loss for the model on a new dataset """
    losses = []
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    loader = DataLoader(dataset, batch_size=args.batch_size, 
                        shuffle=False, collate_fn=data.dna_collate)
    model = model.eval()
    if args.gpu: 
        model = model.cuda()
    with torch.no_grad(): 
        for batch in loader: 

            # Convert batch to gpu 
            if args.gpu: 
                batch[utils.CODON_SEQ] = batch[utils.CODON_SEQ].cuda()
                batch[utils.AA_SEQ] = batch[utils.AA_SEQ].cuda()

            # Model
            preds = model(batch)

            # Mask all codon seqs
            targs = batch[utils.CODON_SEQ][:, 1:]
            seqlens = batch[utils.SEQLEN]

            # Only select non padding
            loss_select= (targs != utils.PADDING)

            # Subset
            preds_subset = preds[loss_select]
            targs_subset = targs[loss_select]
            loss = loss_fn(preds_subset, targs_subset).cpu().numpy()
            losses.extend(list(loss))

    total_loss = np.mean(losses)
    return total_loss

if __name__=="__main__": 
    args = get_args()

    # Make directory for outfile
    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)

    # Setup logger
    logging_level = logging.INFO
    logging.basicConfig(level=logging_level, 
                        format='%(asctime)s %(levelname)s: %(message)s', 
                        handlers=[ logging.StreamHandler(sys.stdout), 
                                  logging.FileHandler(args.out_prefix + '_run.log')]
                        )
    logging.info(f"Args: {args}")

    # Run baselines or neural model 
    if args.run_baselines: 
        baseline_model(args)
    else: 
        neural_model(args)
