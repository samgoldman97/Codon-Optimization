""" models.py """

import torch
import logging
from src import utils

def get_codon_model(codon_model_name, args): 
    """ Build codon model"""
    return {"null" : NullCodon, 
            "lstm" : LSTMCodon}[codon_model_name](args)

def get_aa_model(aa_model_name, args): 
    """ Build aa model"""
    return {"onehot" : AAOneHotEncoder, 
            "bilstm" : AABiLSTM}[aa_model_name](args)

CODON_MODELS = ["null", "lstm"]
AA_MODELS = ["onehot", "bilstm"]

class CodonModel(torch.nn.Module): 
    """ Codon model superclass """

    def __init__(self, args): 
        super(CodonModel, self).__init__()

        self.codon_model = get_codon_model(args.codon_model_name, args)
        self.aa_model = get_aa_model(args.aa_model_name, args)

        self.joint_dim = self.codon_model.get_out_dim() + self.aa_model.get_out_dim()

        # Can output all pred options but start codon since we'll never pred
        self.codon_pred_dim = utils.CODON_VOCAB_MAX 
        self.dropout = torch.nn.Dropout(p=args.joint_dropout )

        # Output logits
        self.lin_transform_mlp = torch.nn.Sequential(torch.nn.Linear(self.joint_dim, args.joint_hidden),
                                                     torch.nn.ReLU(), 
                                                     self.dropout,
                                                     torch.nn.Linear(args.joint_hidden, self.codon_pred_dim))

        # Use aa vocab max because we NEED start in aa vocab, but start is
        # ignored for codon
        self.mask_logits = torch.zeros((utils.AA_VOCAB_MAX + 1,
                                        utils.CODON_VOCAB_MAX)) 
        self.populate_mask_logits()
        self.mask_logits = torch.nn.Parameter(self.mask_logits)
        self.mask_logits.requires_grad = False
        self.trainable_params = sum(p.numel() for p in self.parameters() 
                                    if p.requires_grad)

        # Log number of params
        logging.info(f"Number of params in model: {self.trainable_params}")

 
    def populate_mask_logits(self): 
        """ mask bad tokens """

        # Table should be of size (num_amino_acids + start) by num codon possibilities 

        # fill with -1e-9
        self.mask_logits.fill_(-1e9)

        # For every place v (amino acid), codon (k), set mask to 0 
        for k,v in utils.CODON_TO_AA_NUMS.items(): 
            # Set codon to number
            self.mask_logits[v, k] = 0 

    def forward(self, batch): 
        """ Run forward pass over a batch"""
        aa_sequences = batch[utils.AA_SEQ]
        codon_sequences = batch[utils.CODON_SEQ]
        seqlens = batch[utils.SEQLEN]

        aa_context = self.aa_model(aa_sequences)
        codon_context = self.codon_model(codon_sequences)

        # Slice to create one off effect, 
        aa_context_slice = aa_context[:, 1:, :]
        codon_context_slice = codon_context[:, :-1, :]

        # Stagger them so they're OFF BY 1
        # Copy slice code from earlier
        concated = torch.cat([aa_context_slice, codon_context_slice], dim=2)

        # Activate
        concated = torch.nn.functional.relu(concated)

        # Dropout
        concated = self.dropout(concated)

        # Apply output layer
        res = self.lin_transform_mlp(concated)

        # mask
        # Select right mask for each aa
        # Only mask for the 2nd onward amino acid; skip <START> and M 
        mask = self.mask_logits[aa_sequences[:, 2:]]

        # Mask everything BUT the start token
        res[:, 1:, :] += mask

        # Return LOGITS
        return res

###### Codon models #####

class CodonEncoder(torch.nn.Module): 
    """ Super Class for Codone Encoder to use common features""" 
    def __init__(self, args): 
        super(CodonEncoder, self).__init__()

    def get_out_dim(self): 
        """ Return the output dimension for each position """
        return None

class NullCodon(CodonEncoder): 
    """ Output zeros for codon; don't take into account prediction """

    def __init__(self, args): 
        super(NullCodon, self).__init__(args)

    def get_out_dim(self): 
        """ Return the output dimension for each position """
        return 1

    def forward(self, x): 
        """ Return a zero vector; used for consistency in CodonModel"""
        device = x.device
        return torch.zeros(x.size()).unsqueeze(2).to(device)

class LSTMCodon(CodonEncoder): 
    """ LSTM Codon on the previously predicted or given codons """ 

    def __init__(self, args): 
        super(LSTMCodon, self).__init__(args)
        self.one_hot_embed = CodonOneHotEncoder(args)
        self.out_dim = args.codon_hidden
        self.lstm = torch.nn.LSTM(self.one_hot_embed.out_size, 
                                  hidden_size=args.codon_hidden,
                                  num_layers=args.codon_lstm_layers, 
                                  bidirectional=False, 
                                  dropout=args.codon_dropout, 
                                  batch_first=True)
    def forward(self,x): 
        """ Forward pass"""
        # Embed x 
        x = self.one_hot_embed(x)
        # Now run through lstm
        output, (hn, cn) = self.lstm(x)
        # Get output
        return output

    def get_out_dim(self): 
        """ Return the output dimension for each position """
        return self.out_dim


class CodonOneHotEncoder(CodonEncoder): 

    """ Calculate the one hot encoding of codons"""
    def __init__(self, args): 
        super(CodonOneHotEncoder, self).__init__(args)

        # AA vocab size 
        # 64 + start + pad
        self.vocab_size = utils.CODON_VOCAB_MAX + 1
        self.ident = torch.eye(self.vocab_size)

        self.out_size = self.vocab_size

        # Set padding token to 0!
        self.ident[utils.PADDING]  = 0  

        # Convert to parameter for gpu purposes
        self.one_hot_embedding = torch.nn.Parameter(self.ident)

        # Explicitly do not learn this!
        self.one_hot_embedding.requires_grad = False

    def forward(self,x): 
        return self.one_hot_embedding[x]

    def get_out_dim(self): 
        return self.out_size

###### AA models #####

class AAEncoder(torch.nn.Module): 
    """ Super Class for AA Encoder to use common features""" 
    def __init__(self, args): 
        super(AAEncoder, self).__init__()

    def get_out_dim(self): 
        """ Return the output dimension for each position """
        return None

class AABiLSTM(AAEncoder): 
    """ Run a BiLSTM over the AA sequence"""
    def __init__(self, args): 
        super(AABiLSTM, self).__init__(args)
        self.args = args

        # Compute out dimension
        self.out_dim = args.aa_hidden
        if not args.aa_onedirect: 
            self.out_dim = self.out_dim * 2

        self.embed_layer = AAOneHotEncoder(args)
        self.lstm = torch.nn.LSTM(self.embed_layer.out_size, 
                                  hidden_size=args.aa_hidden,
                                  num_layers=args.aa_bilstm_layers, 
                                  bidirectional=not args.aa_onedirect, 
                                  dropout=args.aa_dropout, 
                                  batch_first=True)
    def forward(self,x): 
        """ Forward pass"""
        # Embed x 
        x = self.embed_layer(x)
        # Now run through lstm
        output, (hn, cn) = self.lstm(x)
        # Get output
        return output

    def get_out_dim(self): 
        return self.out_dim
    
class AAOneHotEncoder(AAEncoder): 
    """ Run a BiLSTM over the AA sequence"""
    def __init__(self, args): 
        super(AAOneHotEncoder, self).__init__(args)

        # AA vocab size 
        # 20 + stop + start + pad
        self.aa_vocab_size = utils.AA_VOCAB_MAX + 1
        self.ident = torch.eye(self.aa_vocab_size)

        self.out_size = self.aa_vocab_size

        # Set padding token to 0!
        self.ident[utils.PADDING]  = 0  

        # Convert to parameter for gpu purposes
        self.one_hot_embedding = torch.nn.Parameter(self.ident)

        # Explicitly do not learn this!
        self.one_hot_embedding.requires_grad = False

    def forward(self,x): 
        return self.one_hot_embedding[x]

    def get_out_dim(self): 
        return self.out_size

#	def set_teacher_force(self, new_prob):
#		self.teacher_force_prob = new_prob
#		
#	def forward(self, text, aa_info):
#		''' 
#		  Pass in context for the next amino acid
#		'''
#		
#		# Reset for each new batch...
#		h_0 = ntorch.zeros(text.shape["batch"], self.num_layers, self.hiddenlen, 
#							names=("batch", "layers", "hiddenlen")).to(self.device)
#		c_0 = ntorch.zeros(text.shape["batch"], self.num_layers, self.hiddenlen, 
#							names=("batch", "layers", "hiddenlen")).to(self.device)
#	 
#		# If we should use all the sequence as input
#		if self.teacher_force_prob == 1: 
#		  text_embedding = self.embedding(text)
#		  hidden_states, (h_n, c_n) = self.LSTM(text_embedding, (h_0, c_0))
#		  output = self.linear_dropout(hidden_states)
#		  output = ntorch.cat([output, aa_info], dim="hiddenlen")
#		  output = self.linear(output)
#		
#		# If we should use some combination of teacher forcing
#		else: 
#			# Use for teacher forcing...
#			outputs = []
#			model_input = text[{"seqlen" : slice(0, 1)}]
#			h_n, c_n = h_0, c_0
#			for position in range(text.shape["seqlen"]): 
#				text_embedding = self.embedding(model_input)
#				hidden_states, (h_n, c_n) = self.LSTM(text_embedding, (h_n, c_n))
#
#				output = self.linear_dropout(hidden_states)
#				aa_info_subset = aa_info[{"seqlen" : slice(position, position+1)}]
#				output = ntorch.cat([output, aa_info_subset], dim="hiddenlen")
#				output = self.linear(output)
#				outputs.append(output)
#
#				# Define next input... 
#				if random.random() < self.teacher_force_prob: 
#					model_input = text[{"seqlen" : slice(position, position+1)}]
#				else: 
#					# Masking output... 
#					mask_targets = text[{"seqlen" : slice(position, position+1)}].clone()
#					if position == 0: 
#						mask_targets[{"seqlen" : 0}] = TEXT.vocab.stoi["<start>"]
#					mask_bad_codons = ntorch.tensor(mask_tbl[mask_targets.values], 
#						names=("seqlen", "batch", "vocablen")).float()
#
#					model_input = (output + mask_bad_codons).argmax("vocablen")
#					# model_input = (output).argmax("vocablen")
#			  
#			output = ntorch.cat(outputs, dim="seqlen")
#		return output
#  
#
#
#
#
