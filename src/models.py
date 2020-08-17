""" models.py """

import torch
import logging

import utils

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
        mask = self.mask_logits[aa_sequences[:, 1:]]

        # Mask everything BUT the preds for M token, which can be anythign
        res[:, 1:, :] += mask[:, 1:, :]

        # Return LOGITS
        return res

    def generate_codon_seqs(self, batch, teacher_force=False): 
        """ Generate codon sequences for the batch 
        TODO (refactoring) : 
        - Abstract the parts of this loop that mirror forward into that 
        Return: 
            List of lists of sequences
        """

        aa_sequences = batch[utils.AA_SEQ]
        codon_sequences = batch[utils.CODON_SEQ]
        seqlens = batch[utils.SEQLEN]

        with torch.no_grad(): 
            # Keep AA context constant
            aa_context = self.aa_model(aa_sequences)

            # mask; this shouldn't chagne as we output codons
            # Select right mask for each aa
            # Only mask for the 2nd onward amino acid; skip <START> and M 
            # Here, skip the <START> token
            mask = self.mask_logits[aa_sequences[:, 1:]]

            # Assume codon sequences have start token to start
            codon_sub_sequence = codon_sequences[:, [0]]
            generated_seqs = [codon_sub_sequence]

            # LSTM State holder
            prev_state = None

            # Now generate each position sequentially
            # Generate a codon at every position but last
            for codon_position in range(codon_sequences.shape[1] - 1):

                # Concat all generated
                codon_sub_sequence = generated_seqs[-1]

                # Avoid running whole LSTM up to this point
                if isinstance(self.codon_model, LSTMCodon): 
                    codon_context, prev_state = self.codon_model(codon_sub_sequence, return_full=True,
                                                                 state=prev_state,)
                else:
                    codon_context = self.codon_model(codon_sub_sequence)

                # Get only up to the current codon position 
                # Add 1 for the off by 1 shift of codon and add 1 for list
                # indexing
                aa_context_slice = aa_context[:, codon_position  + 1 :codon_position + 2 , :]
                # Only the last codon output  
                codon_context_slice = codon_context[:, -1:, :]

                # Stagger them so they're OFF BY 1
                # Copy slice code from earlier
                concated = torch.cat([aa_context_slice, codon_context_slice], dim=2)

                # Activate
                concated = torch.nn.functional.relu(concated)

                # Dropout
                concated = self.dropout(concated)

                # Apply output layer
                res = self.lin_transform_mlp(concated)

                # Don't mask the first round bc this is methionine generation
                if codon_position > 0: 
                    # Res should be zero because it's only 1 dimension we're
                    # interested in 
                    res[:, 0, :] += mask[: , codon_position, :]

                # TODO: Include sampling here, not just argmax
                new_inputs = res.argmax(2)
                generated_seqs.append(new_inputs)

        # Catenate everything BUT the start token
        gen_seqs = torch.cat(generated_seqs[1:], 1).cpu().numpy()
        out_seqs = [gen_seq[:seqlen] for gen_seq,seqlen in zip(gen_seqs, seqlens)]
        return out_seqs

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

    def forward(self,x, state=None, return_full = False): 
        """ Forward pass"""
        # Embed x 
        x = self.one_hot_embed(x)
        # Now run through lstm
        output, (hn, cn) = self.lstm(x, state)
        # Get output
        if not return_full: 
            return output
        else: 
            return output, (hn, cn)

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
