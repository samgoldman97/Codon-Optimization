'''
	models.py
'''

from namedtensor import ntorch
from namedtensor.text import NamedField
import torch


#### Frequency baesd models ###

class AA_NGRAM(ntorch.nn.Module):
	'''
	A model class that will take a string of codons and turn them into amino acids, then turn those amino acids into n gram based frequencies for what codon should be predicted in each position

	Args: 
		params: Dict containing the following: 
			"CODON_TO_AA" : Embedding to convert each codon to an amino acidindex
			"N_GRAM_DICTS" : List of dictionaries to use
			"N_LIST": How many indices each dict takes
			"WEIGHT_LIST" : How to weight each dictionary
			"OUT_VOCAB": How many items in the output
			"DEVICE" : device to use
			"TEXT" :torch text obj


	TODO: 
		- Ignore pading predicts in forward pass to save time
		- Also speed this up generally
	'''

	def __init__(self, params): 
		super(AA_NGRAM, self).__init__()
		self.codon_to_aa = params["CODON_TO_AA"]
		self.dict_list = params["N_GRAM_DICTS"]
		# How many indcies each dict takes
		self.n_list = params["N_LIST"]
		# probability to apply to each n gram used
		self.weight_list = params["WEIGHT_LIST"]
		self.longest_n = max(self.n_list)
		self.out_vocab = params["OUT_VOCAB"]
		self.device = params["DEVICE"]
		self.text = params["TEXT"]
	
	def forward(self, seq): 
		seq_len = seq.shape["seqlen"]
		batch_size = seq.shape["batch"]
		  
		pad_token = self.text.vocab.stoi["<pad>"]
		additional_padding = ntorch.ones(batch_size, self.longest_n, 
										names=("batch", "seqlen")).long().to(self.device)
		additional_padding *= pad_token
		
		seq = ntorch.cat([additional_padding, seq, additional_padding],
						dim="seqlen")
		
		
		amino_acids = self.codon_to_aa[seq.values]
		
		return_ar = ntorch.zeros(seq_len, batch_size, self.out_vocab,
								 names=("seqlen", "batch", "vocablen"))
		
		# convert to numpy to leave GPU 
		amino_acids = amino_acids.detach().cpu().numpy()
		for batch_item in range(batch_size): 
		  # start at n, end at seq_len - n
			for seq_item in range(self.longest_n, seq_len - self.longest_n):
				# Must iterate over all dictionaries
				for weight, n, ngram_dict in zip(self.weight_list,
												self.n_list, self.dict_list):
					# N gram is a 2d numpy array containing an amino acid embedding in each row
					n_gram = amino_acids[batch_item,seq_item - n : seq_item + n + 1]

					# note, we want to populate the return ar before padding!
					return_ar[{"seqlen" : seq_item - self.longest_n, 
							 "batch" : batch_item}] += weight * ngram_dict[str(n_gram)].float()

		return return_ar.to(self.device)

class FreqModel(ntorch.nn.Module):
	''' 
		Simple language model that uses the frequencies of the amino acids for modeling
		Work done in aa_info model for this 
	'''
	def __init__(self):
		super(FreqModel, self).__init__()

	def forward(self, text, aa_info):
		''' Pass in context for the next amino acid '''
		return aa_info.rename("hiddenlen", "vocablen")

#### Compress amino acids into simple embedding ####

class AA_COMPRESS(ntorch.nn.Module):
	'''
	A model to compress a codon sequence into its amino acid representation
	Can be easily used by passing in an embedding that turns each amino acid into a onehot 
	OR each amino acid into a frequency table representation for its codon (unigram model)


	Args: 
		params: Dict containing the following: 
			"CODON_TO_AA" : Embedding to convert each codon to an amino acidindex
	'''

	def __init__(self, params): 
		super(AA_COMPRESS, self).__init__()
		self.codon_to_aa = params["CODON_TO_AA"]
    	self.start_index = params["START_INDEX"]
		self.aa_embed = (ntorch.nn.Embedding.from_pretrained(self.codon_to_aa)
						 .spec("seqlen", "hiddenlen"))

		# don't learn these.. 
		self.aa_embed.weight.requires_grad_(False)  

	def forward(self, seq): 
		# replace first position with Methionine!
		seq_copy = seq.clone()
		seq_copy[{"seqlen" : 0}] = self.start_index
		seq = seq_copy

		return self.aa_embed(seq)




class AA_BILSTM(ntorch.nn.Module):
	'''
	A model to compress a codon sequence into its amino acid representation and
	 then run a bidirectional LSTM over this sequence	

	Args: 
		params: Dict containing the following: 
			"CODON_TO_AA" : Embedding to convert each codon to an amino acidindex
			"EMBED_DIM": How dimension of embedding 
			"HIDDEN_LEN" : Hidden dimension in biLSTM
			"LSTM_DROPOUT" : BiLSTM dropout after each layer
			"BIDIRECTIONAL" : Boolean 
			"DEVICE": device
	'''
  
	def __init__(self, params): 
		super(AA_BILSTM, self).__init__()
		self.codon_to_aa = params["CODON_TO_AA"]
		self.embedding_size = params["EMBED_DIM"]
		self.hiddenlen = params["HIDDEN_LEN"]
		self.num_layers = params["NUM_LAYERS"]
		self.lstm_dropout = params["LSTM_DROPOUT"]
		self.bidirectional = params["BIDIRECTIONAL"]
		self.device = params["DEVICE"]
		self.start_index = params["START_INDEX"]

		self.num_directions = 1
		if self.bidirectional:
		  self.num_directions = 2
		
		self.aa_embed = (ntorch.nn.Embedding.from_pretrained(self.codon_to_aa)
						 .spec("seqlen", "embedlen"))
		
		# don't learn these.. 
		self.aa_embed.weight.requires_grad_(False)  
		
		self.LSTM = (ntorch.nn.LSTM(self.embedding_size, self.hiddenlen,
									num_layers=self.num_layers, 
									bidirectional=self.bidirectional, dropout=self.lstm_dropout
								   )
					.spec("embedlen", "seqlen", name_out="hiddenlen")
					)
		
		
	def forward(self, seq): 
		'''
		Forward pass
		''' 
		# Replace start codon...
		seq_copy = seq.clone()
		seq_copy[{"seqlen" : 0}] = self.start_index
		seq = seq_copy

		aa_rep = self.aa_embed(seq)    
		h_0 = ntorch.zeros(self.num_layers * self.num_directions, aa_rep.shape["batch"], self.hiddenlen, 
							names=("layers", "batch", "hiddenlen")).to(self.device)
		c_0 = ntorch.zeros(self.num_layers * self.num_directions, aa_rep.shape["batch"], self.hiddenlen, 
							names=("layers", "batch", "hiddenlen")).to(self.device)
		
		h_0 = h_0.transpose("batch", "layers", "hiddenlen")
		c_0 = c_0.transpose("batch", "layers", "hiddenlen")
		hidden_states, (h_n, c_n) = self.LSTM(aa_rep, (h_0, c_0))
		return hidden_states
		

class NNLM(ntorch.nn.Module):
	''' 
	Simple LSTM class.
	A model that uses the previously predicted (or true) codons in addition to
		provided information about the amino acid for prediction

	Args: 
		params: Dict containing the following: 
			"VOCAB_SIZE" : Size of output vocab
			"EMBED_DIM": Dimension to embed; can be None
			"HIDDEN_LEN" : Hidden dimension in lstm 
			"LSTM_DROPOUT" : Dropout after each layer of LSTM 
			"LINEAR_DROPOUT" : Dropout to apply after last layer of LSTM 
			"OUT_VOCAB" : OUt vocab
			"TEACHER_FORCE" : value 0 - 1 to decide what fraction of the time to use teacher forcing, 
			"DEVICE" : device

			"bidirectional" : Boolean 
	'''
	def __init__(self, params):
		super(NNLM, self).__init__()
		self.vocab_size = params["VOCAB_SIZE"]
		self.embedding_size = params["EMBED_DIM"]
		self.hiddenlen = params["HIDDEN_LEN"]
		self.num_layers = params["NUM_LAYERS"]
		self.linear_dropout = ntorch.nn.Dropout(p=params["LINEAR_DROPOUT"])
		self.lstm_dropout = params["LSTM_DROPOUT"]
		self.out_vocab = params["OUT_VOCAB"]
		self.aa_info_size = params["AA_COMPRESS_SIZE"]
		self.teacher_force_prob = params["TEACHER_FORCE"]
		self.device = params["DEVICE"]
		if self.embedding_size is not None: 
		  self.embedding = ntorch.nn.Embedding(num_embeddings=params["VOCAB_SIZE"], 
											   embedding_dim = self.embedding_size).spec("seqlen", "embedlen")
		else: 
			self.embedding = (ntorch.nn.Embedding.
							from_pretrained(torch.eye(self.vocab_size)
										   )
						   ).spec("seqlen", "embedlen")

			self.embedding.weight.requires_grad_(False)
			self.embedding_size = self.vocab_size
		
		self.LSTM = (ntorch.nn.LSTM(self.embedding_size, self.hiddenlen, num_layers=self.num_layers)
					.spec("embedlen", "seqlen", name_out="hiddenlen")
					)
		self.linear = (ntorch.nn.Linear(self.hiddenlen + self.aa_info_size, 
										self.out_vocab)
					   .spec("hiddenlen", "vocablen")
					  )
		
	  
	def set_teacher_force(self, new_prob):
		self.teacher_force_prob = new_prob
		
	def forward(self, text, aa_info):
		''' 
		  Pass in context for the next amino acid
		'''
		
		# Reset for each new batch...
		h_0 = ntorch.zeros(text.shape["batch"], self.num_layers, self.hiddenlen, 
							names=("batch", "layers", "hiddenlen")).to(self.device)
		c_0 = ntorch.zeros(text.shape["batch"], self.num_layers, self.hiddenlen, 
							names=("batch", "layers", "hiddenlen")).to(self.device)
	 
		# If we should use all the sequence as input
		if self.teacher_force_prob == 1: 
		  text_embedding = self.embedding(text)
		  hidden_states, (h_n, c_n) = self.LSTM(text_embedding, (h_0, c_0))
		  output = self.linear_dropout(hidden_states)
		  output = ntorch.cat([output, aa_info], dim="hiddenlen")
		  output = self.linear(output)
		
		# If we should use some combination of teacher forcing
		else: 
			# Use for teacher forcing...
			outputs = []
			model_input = text[{"seqlen" : slice(0, 1)}]
			h_n, c_n = h_0, c_0
			for position in range(text.shape["seqlen"]): 
				text_embedding = self.embedding(model_input)
				hidden_states, (h_n, c_n) = self.LSTM(text_embedding, (h_n, c_n))

				output = self.linear_dropout(hidden_states)
				aa_info_subset = aa_info[{"seqlen" : slice(position, position+1)}]
				output = ntorch.cat([output, aa_info_subset], dim="hiddenlen")
				output = self.linear(output)
				outputs.append(output)

				# Define next input... 
				if random.random() < self.teacher_force_prob: 
					model_input = text[{"seqlen" : slice(position, position+1)}]
				else: 
					# Masking output... 
					mask_targets = text[{"seqlen" : slice(position, position+1)}].clone()
					if position == 0: 
						mask_targets[{"seqlen" : 0}] = TEXT.vocab.stoi["<start>"]
					mask_bad_codons = ntorch.tensor(mask_tbl[mask_targets.values], 
						names=("seqlen", "batch", "vocablen")).float()

					model_input = (output + mask_bad_codons).argmax("vocablen")
					# model_input = (output).argmax("vocablen")
			  
			output = ntorch.cat(outputs, dim="seqlen")
		return output
  






