#!/usr/pin/python

''' 

helpers.py 
Helper python functions

'''

from Bio import Entrez, SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import IUPAC
import torch
import torchtext
from torchtext.data import Iterator, BucketIterator
from namedtensor import ntorch
from namedtensor.text import NamedField
from collections import defaultdict
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
import random
import re

##### Data functions ######

def set_difference(fasta1, fasta2, outfile):
    '''Find all sequences in fasta1 not in fasta2 and save them into outfile'''
    sequences1 = set(str(rec.seq) for rec in SeqIO.parse(fasta1, "fasta"))
    sequences2 = set(str(rec.seq) for rec in SeqIO.parse(fasta2, "fasta"))
    f1_only = sequences1.difference(sequences2)
    f2_only = sequences2.difference(sequences1)
    intersection = sequences1.intersection(sequences2)
    print("Len f1: ", len(sequences1))
    print("Len f2: ", len(sequences2))
    print("Len f1 only: ", len(f1_only))
    print("Len f2 only: ", len(f2_only))
    print("F1 - F1 only: ", len(sequences1) - len(f1_only))
    print("Len intersection", len(intersection))

    records = [SeqRecord(Seq(seq, IUPAC.protein)) for seq in f1_only]
    SeqIO.write(records, outfile, "fasta")
    

    
def download_human_transcripts(list_file, email_address, outfile="human_HE.fasta"): 
	'''Download human transcripts'''

	Entrez.email = email_address 
	with open(list_file, "r") as fp: 
		ids = [i.strip() for i in fp.readlines()]

	handle = Entrez.efetch(db="nucleotide", 
							 id=ids,
							 retmode ="xml", 
							 # rettype="fasta", 
							 strand=1)

	output = Entrez.parse(handle)
	seqs = []
	for entry in output:
		feat_tbl = entry["GBSeq_feature-table"]
		num_cds = 0 
		for j in feat_tbl: 
			if j['GBFeature_key'] == "CDS": 
				num_cds += 1
				cds = j["GBFeature_location"]

				# print(cds)
				cds_loc = j['GBFeature_intervals']
				# print("Num of features: ", len(cds_loc))
				start = int(cds_loc[0]['GBInterval_from'])
				end = int(cds_loc[0]['GBInterval_to'])
				# print(start, end)
				seq = entry["GBSeq_sequence"][start - 1 : end]
		if num_cds != 1: 
			print("Error: Too many CDS found", num_cds)
		seqs.append(SeqRecord(Seq(seq), 
								id=entry['GBSeq_locus'],
								description=entry["GBSeq_definition"]
								)
					)
	handle.close()
	with open(outfile, "w") as output_handle:
		SeqIO.write(seqs, output_handle, "fasta")

def build_helper_tables(TEXT, device): 
	'''  Load CSV file of nucleotide sequences
	Args: 
		TEXT: torchtext field for the vocab of nucleotides
		device : torch device

	Returns: 
		AA_LABEL: torch text field for amino acids to index
		index_table: look up table s.t. you can index with codon index and receive one hot for AA
		codon_to_aa: dictionary to move from codon to amino acid string
		codon_to_aa_index: look up table s.t. you can index with codon index and receive AA index
		mask_tbl: Index with codon and get a mask table to add to the output of the model and get synonymous options
	'''

	AA_LABEL = NamedField(names=("seqlen", ), 
						lower=True)
	bases = "tcag"
	codons = [a + b + c for a in bases for b in bases for c in bases]
	aa = [str(Seq(j).translate()) for j in codons]
	# Mapping of codons to amino acids
	codon_to_aa = dict(zip(codons, aa))
	# One hot encoding of all possible amino acids
	AA_LABEL.build_vocab(aa)

	# Make a look up table, such that you can index with the vocab item (e.g. a codon)
	# and get the one hot corresponding to its amino acid
	one_hot_vec = torch.eye(len(AA_LABEL.vocab))
	zero_vec =  torch.zeros(len(AA_LABEL.vocab), 1)
	# Useful..
	direct_look_up = [one_hot_vec[AA_LABEL.vocab.stoi[codon_to_aa[TEXT.vocab.itos[i]]]].unsqueeze(1) 
						if TEXT.vocab.itos[i] in codon_to_aa else zero_vec
						for i in range(len(TEXT.vocab.stoi))]

	# Shape codon x one hot 
	index_table = torch.cat(direct_look_up, dim=1).t()
	codon_to_aa_index = torch.argmax(index_table, 1)

	# Build masking table
	# Here, if it's a synonymous option, give it 0 value, if not, give -1e9
	# Add this with the output vector (i.e. output += mask_tbl[trg]) before softmax
	mask_tbl = torch.tensor(np.array([[0 if (codon in codon_to_aa and codon_2 in codon_to_aa and codon_to_aa[codon] == codon_to_aa[codon_2]) else -1e9 
				 for codon_2 in TEXT.vocab.itos] 
				for index, codon in enumerate(TEXT.vocab.itos)])).to(device)

	# For ease, make sure padding gets predicted as padding...
	mask_tbl[1,1] = 0

	return (AA_LABEL, index_table, codon_to_aa, codon_to_aa_index, mask_tbl)

# def get_first_orf(seq):
# 	'''  Helper function to truncate sequence to first ATG window and end at last multiple of 3

# 	NOTE: This is since deprecated because we handle this in preprocessing using non-naive annotations 

# 	Args: 
# 		seq: string to reformat
# 	Return: 
# 		A string beginning from ATG (else None) if "ATG" appears in seq and divisible by 3
	
# 	''' 
# 	for position, i in enumerate(seq[:-2]):
# 		if seq[position: position + 3] == "ATG":
# 			end_position = ((len(seq) - position) // 3) * 3
# 			return seq[position:end_position + position]
# 	return None

def convert_fasta_to_csv(file_name, out_file = "cds.csv", random_state = 1, print_stats = True):
	''' Convert input fasta file to csv file 
	Args: 
		file_name: Name of the fasta file to parse
		out_file: Name of output csv file of processed sequences
		random_state: Seed to use to shuffle the sequences
		print_stats: Print number of bases and codons after processing 
	Returns: 
		None
	'''

	sequences = [str(rec.seq) for rec in SeqIO.parse(file_name, "fasta")]
	df = pd.DataFrame(sequences, columns=["sequence"])
	df = df.sample(frac=1, random_state=random_state)
	# df = df[:500]
	df.to_csv(out_file, index=False, header=False)
	if print_stats: 	
		num_bases = 0
		print(len(df))
		for i,j in df.iterrows():
			num_bases += len(j.sequence)
		print("Bases, codons:" , num_bases, num_bases / 3)

def load_csv_data(csv_file, device, random_state = 1, train_split = 0.8, batch_size = 10 ): 
	'''  Load CSV file of nucleotide sequences
	Args: 
		csv_file: Name of the csv file of nucleotide sequences to model
		random_state: Integer for random seed of test train split
		train_split: Fraction of training test (float 0 to 1)
		device : torch device

	Returns: 
		train_bucket_iterator, test_bucket_iterator, TEXT
	'''
	# Prepend input with a start token
	tokenize = lambda x : ["<START>"] + re.findall('.{%d}' % 3, x)
	TEXT = NamedField(names=("seqlen", ), sequential=True, 
						lower=True, tokenize=tokenize)

	my_data = torchtext.data.TabularDataset(csv_file, format="CSV", 
											fields=[("sequence", TEXT)])
	# Randomly seed then separate train test
	random.seed(random_state)
	train, test = my_data.split(split_ratio=train_split, random_state=random.getstate())
	# Remove random seed
	random.seed(None)
	# Build vocab
	TEXT.build_vocab(train)

	# Create bucket iterators
	train_iter_bucket, test_iter_bucket = torchtext.data.BucketIterator.splits(
		(train, test), batch_sizes=(batch_size,batch_size), sort_within_batch=False, 
			sort_key=lambda x : len(x.sequence),
		device=torch.device(device, ))

	return train_iter_bucket, test_iter_bucket, TEXT

#### Frequency Model Helper Functions #####

def make_n_gram_dict(train_iter, n, amino_acid_conversion, TEXT, AA_LABEL):
	''' Helper function to create a frequency default dictionary
	
	Args: 
		train_iter: Training bucket iterator
		n: Number of amino acids to each side of AA (e.g. 0 is unigram, 1 is trigram)
		amino_acid_conversion: index_table converting the codon index to AA index
		TEXT: torchtext field for the vocab of nucleotides
		AA_LABEL: Torchtext for amino acids

	Returns: 
		default_dict: dictionary mapping a sequence of amino acids to probability over codons
	TODO: 
		Make this faster
	'''

	default_obj = lambda : torch.tensor(np.zeros(len(TEXT.vocab.stoi)))
	default_dict = defaultdict(default_obj)

	with torch.no_grad():
		ident_mat = np.eye(len(TEXT.vocab.stoi))
		ident_mat_aa = np.eye(len(AA_LABEL.vocab))
		for i, batch in enumerate(train_iter):
			# Select for all non zero tensors
			# Use this to find all indices that aren't padding
			seq_len = batch.sequence.shape["seqlen"]
			batch_size = batch.sequence.shape["batch"]

			# Pad amino acids and seq with <pad> token 
			pad_token = TEXT.vocab.stoi["<pad>"]
			additional_padding = ntorch.ones(batch_size, n, 
											names=("batch", "seqlen")).long()
			additional_padding *= pad_token

			seq = ntorch.cat([additional_padding, batch.sequence, additional_padding],
							dim="seqlen")

			# Now one hots.. 
			amino_acids = amino_acid_conversion[seq.values].detach().cpu().numpy()
			# Note: we should assert that start and pad are treated the same
			# This is because at test time, presumably we narrow the start for the AA.. 
			if i == 0:
				assert((amino_acids[0,n] == amino_acids[0,0]).all())

			seq = seq.detach().cpu().numpy()
			# Pad with padding token
			for batch_item in range(batch_size): 
				# start at n, end at seq_len - n
				for seq_item in range(n, seq_len - n):
					# Middle token is a discrete number representing the codon (0 to 66)
					middle_token = seq[batch_item, seq_item]
					# N gram is a 2d numpy array containing an amino acid embedding in each row
					n_gram = amino_acids[batch_item,seq_item - n : seq_item + n + 1]

					default_dict[str(n_gram)][middle_token] += 1

	for key in default_dict: 
		default_dict[key] /= (default_dict[key]).sum()
			
	return default_dict

##### Output model values to file #####

def get_prediction(batch, model, aa_compress, mask_tbl, device): 
	''' Predict outputs from sequence'''
	model.to(device)
	model.eval()
	with torch.no_grad():
		seq_len = batch.sequence.shape["seqlen"]
		text = batch.sequence.narrow("seqlen", 0, seq_len - 1)
		target = batch.sequence.narrow("seqlen", 1, seq_len - 1)
		# Forward
		predictions = model(text, aa_compress(target)) 
		mask_bad_codons = ntorch.tensor(mask_tbl[target.values], 
							 names=("seqlen", "batch", "vocablen")).float()
		predictions = (mask_bad_codons + predictions.float())
		predictions = predictions.argmax("vocablen")
	return predictions

def get_prediction_iter(iterator, model, aa_compress, mask_tbl, device): 
	''' Predict outputs from sequence'''

	model.to(device)
	model.eval()
	output = []
	with torch.no_grad():
		for batch in iterator:
			seq_len = batch.sequence.shape["seqlen"]
			text = batch.sequence.narrow("seqlen", 0, seq_len - 1)
			target = batch.sequence.narrow("seqlen", 1, seq_len - 1)
			# Forward
			predictions = model(text, aa_compress(target)) 
			mask_bad_codons = ntorch.tensor(mask_tbl[target.values], 
							 names=("seqlen", "batch", "vocablen")).float()
			predictions = (mask_bad_codons + predictions.float())
			predictions = predictions.argmax("vocablen")
			output.append(predictions)

	return output

def translate_to_seq(x, TEXT): 
	''' Takes in single tensor of name seqlen'''
	my_str = "".join([TEXT.vocab.itos[i] for i in x.values])
	my_str = my_str.split("<pad>")[0]
	my_str = my_str.split("<unk>")[0]

	if "<start>" in my_str: 
		my_str = my_str.split("<start>")[1]

	return my_str.upper()

def output_iterator_to_file(iter_, TEXT, outfile="iterator_output.txt"): 
	''' Output nucleotides to file; one gene per line'''
	with open(outfile, "w") as fp: 
		for batch in iter_:
			for index in range(batch.sequence.shape["batch"]): 
				new_seq = translate_to_seq(batch.sequence[{"batch": index}], TEXT)
				fp.write(new_seq + "\n")
		
def output_list_of_res(res, TEXT, outfile = "test.txt"):
	''' Output nucleotides to file; one gene per line'''
	with open(outfile, "w") as fp:
		for batch in res: 
			for index in range(batch.shape["batch"]): 
				new_seq = translate_to_seq(batch[{"batch": index}], TEXT)
				fp.write(new_seq + "\n")

##### Model train and eval ##### 

def train_model(train_iter, model, aa_compress, TEXT, device, train_params, 
								optimizer, optimizer_aa=None):

	''' Train a given model 
	
		Args: 
			train_iter: Bucket iter
			model : Model that works over data iter
			aa_compress: model compress
			TEXT: torchtext obj
			device: device
			train_params: 
				TEACHER_FROCE: teacher force param
				num_epochs: Num epochs to train
				plot_loss: Weather or not to plot the loss
			optimizer: optimizer obj
			optimizer_aa: optimizer obj for the amino acid rep 
		NOTE: 
			Optimizer_aa is optional if we want to learn on the compressions.

		return: model  
	'''
	
	model.train()
	if "TEACHER_FORCE" in train_params:
		model.teacher_force_prob = train_params["TEACHER_FORCE"]

	if optimizer_aa is not None:
		# Does this accidentally turn on gradients? 
		aa_compress.train()
		
	loss_function = ntorch.nn.CrossEntropyLoss().spec("vocablen")
	model.to(device)
	aa_compress.to(device)
	loss_values = []
	for epoch in range(train_params["num_epochs"]):
		epoch_loss = 0
		for i, batch in enumerate(train_iter):
			model.zero_grad()
			if optimizer_aa: 
				aa_compress.zero_grad()

			
			# Select for all non zero tensors
			# Use this to find all indices that aren't padding
			seq_len = batch.sequence.shape["seqlen"]
			batch_size = batch.sequence.shape["batch"]
			text = batch.sequence.narrow("seqlen", 0, seq_len - 1)
			target = batch.sequence.narrow("seqlen", 1, seq_len - 1)
			
			stacked_target = target.stack(dims=("batch", "seqlen"), 
														name="seqlen")

			mask = (stacked_target != TEXT.vocab.stoi["<pad>"])
			prop_indices = (ntorch.nonzero(mask)
											.get("inputdims", 0)
										 )
			# Forward
			predictions = model(text, aa_compress(target)) 
			
			# Stack the predictions into one long vector
			predictions = predictions.stack(dims=("batch", "seqlen"), name="seqlen")
						
			# Only find loss on sequences
			# TODO: Divide by batch size... 
			loss = loss_function(predictions.index_select("seqlen", prop_indices),
													 stacked_target.index_select("seqlen", prop_indices)) 
			loss /= batch_size
			
			epoch_loss += loss.item()
			
			loss.backward()
			# gradient clip
			torch.nn.utils.clip_grad_norm_(model.parameters(), train_params["grad_clip"])
			optimizer.step()
			
			if optimizer_aa: 
				torch.nn.utils.clip_grad_norm_(aa_compress.parameters(), train_params["grad_clip"])
				optimizer_aa.step()

			

		print("Epoch: {} -- Loss: {}".format(epoch, epoch_loss))        
		loss_values.append(epoch_loss)

	if train_params["plot_loss"]: 
		fig, ax = plt.subplots()
		ax.plot([t for t in range(len(loss_values))], loss_values)
		ax.set(xlabel='Epochs', ylabel='Loss', title='Loss during Optimization')
		plt.show()

	return model


def joint_ppl_acc(data_iter, model, device, aa_compress, TEXT, mask_tbl, teacher_force=1):
	''' Calculate perplexity and accuracy on data iter
	
		Args: 
			data_iter: Bucket iter
			model : Model that works over data iter
			device: device
			aa_compress: Helper model to consider dependnecies along model
			TEXT: text object over codons
			mask_tbl: Mask table 
			teacher_force: Whether to use teacher forcing or not. Default yes
	Returns: 
		{"acc": accuracy, "ppl": perplexity}
	'''

	model.to(device)
	model.eval()
	model.teacher_force_prob = teacher_force
	aa_compress.to(device)
	aa_compress.eval()
	ppl = 0
	num_total = 0 
	num_correct = 0 
	num_total = 0 
	loss_function = ntorch.nn.CrossEntropyLoss(reduction="none").spec("vocablen")
	with torch.no_grad():
		for i, batch in enumerate(data_iter):

			# Select for all non zero tensors
			# Use this to find all indices that aren't padding
			seq_len = batch.sequence.shape["seqlen"]
			text = batch.sequence.narrow("seqlen", 0, seq_len - 1)
			target = batch.sequence.narrow("seqlen", 1, seq_len - 1)

			stacked_target = target.stack(dims=("batch", "seqlen"), 
												name="seqlen")
			mask = (stacked_target != TEXT.vocab.stoi["<pad>"])
			prop_indices = (ntorch.nonzero(mask)
							.get("inputdims", 0)
						 ).rename("elements", "seqlen")
			# Forward
			predictions = model(text, aa_compress(target)) 


			# Mask all outputs that don't work
			mask_bad_codons = ntorch.tensor(mask_tbl[target.values], 
							 names=("seqlen", "batch", "vocablen")).float()
			predictions = (mask_bad_codons.double() + predictions.double())




			# Stack the predictions into one long vector and get correct indices
			predictions = (predictions.stack(dims=("batch", "seqlen"), name="seqlen")
						 .index_select("seqlen", prop_indices)
						)

			predictions_hard = predictions.argmax("vocablen")

			# Select correct indices from target
			stacked_target = (stacked_target.index_select("seqlen", prop_indices)
							 )
			num_correct += (predictions_hard == stacked_target).sum().item()   
			num_total += predictions_hard.shape["seqlen"]


			loss = loss_function(predictions, stacked_target)
			ppl += loss.sum().item()

			# For quick results, toggle this
			# if i == 20: 
			#	break  
	return {"acc" : num_correct / num_total, "ppl" : np.exp(ppl / num_total)}


def save_model(model, outfile_name): 
	torch.save(model.state_dict(), outfile_name)
