import torch
import torchtext
from namedtensor import ntorch
from namedtensor.text import NamedField

class FreqModel(ntorch.nn.Module):
  ''' 
  Simple language model that uses the frequencies of the amino acids for modeling
  Magic happens in aa_info model
  '''
  def __init__(self):
    super(FreqModel, self).__init__()

  def forward(self, text, aa_info):
    ''' 
      Pass in context for the next amino acid
    '''

    return aa_info.rename("hiddenlen", "vocablen")
    
class AA_COMPRESS(ntorch.nn.Module):
  '''
  Compress info for codon sequence at the amino acid level
  '''
  
  def __init__(self, params): 
    super(AA_COMPRESS, self).__init__()
    self.codon_to_aa = params["CODON_TO_AA"]
    self.aa_embed = (ntorch.nn.Embedding.from_pretrained(self.codon_to_aa)
                     .spec("seqlen", "hiddenlen"))
    
    # don't learn these.. 
    self.aa_embed.weight.requires_grad_(False)  
    
  def forward(self, seq): 
    return self.aa_embed(seq)    


class AA_NGRAM(ntorch.nn.Module):
  '''
  Compress info for codon sequence at the amino acid level
  TODO: Ignore pading predicts in forward pass to save time
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
    
  def forward(self, seq): 
    
    seq_len = seq.shape["seqlen"]
    batch_size = seq.shape["batch"]
      
    pad_token = TEXT.vocab.stoi["<pad>"]
    additional_padding = ntorch.ones(batch_size, self.longest_n, 
                                    names=("batch", "seqlen")).long()
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

    return return_ar
  

 

 class AA_BILSTM(ntorch.nn.Module):
  '''
  Compress info for codon sequence at the amino acid level
  '''
  
  def __init__(self, params): 
    super(AA_BILSTM, self).__init__()
    self.codon_to_aa = params["CODON_TO_AA"]
    self.embedding_size = params["EMBED_DIM"]
    self.hiddenlen = params["HIDDEN_LEN"]
    self.num_layers = params["NUM_LAYERS"]
    self.lstm_dropout = params["LSTM_DROPOUT"]
    self.bidirectional = params["BIDIRECTIONAL"]
    self.num_directions = 1
    if self.bidirectional:
      self.num_directions = 2
    
    self.aa_embed = (ntorch.nn.Embedding.from_pretrained(self.codon_to_aa)
                     .spec("seqlen", "embedlen"))
    
    # don't learn these.. 
    self.aa_embed.weight.requires_grad_(False)  
    
    self.LSTM = (ntorch.nn.LSTM(self.embedding_size, self.hiddenlen,
                                num_layers=self.num_layers, 
                                bidirectional=self.bidirectional
                               )
                .spec("embedlen", "seqlen", name_out="hiddenlen")
                )
    
    
  def forward(self, seq): 
    '''
    Forward pass
    ''' 
    aa_rep = self.aa_embed(seq)    
    h_0 = ntorch.zeros(self.num_layers * self.num_directions, aa_rep.shape["batch"], self.hiddenlen, 
                        names=("layers", "batch", "hiddenlen")).to(device)
    c_0 = ntorch.zeros(self.num_layers * self.num_directions, aa_rep.shape["batch"], self.hiddenlen, 
                        names=("layers", "batch", "hiddenlen")).to(device)
    
    h_0 = h_0.transpose("batch", "layers", "hiddenlen")
    c_0 = c_0.transpose("batch", "layers", "hiddenlen")
    hidden_states, (h_n, c_n) = self.LSTM(aa_rep, (h_0, c_0))
    
    return hidden_states

class NNLM(ntorch.nn.Module):
  ''' 
  Simple LSTM class.
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
    if self.embedding_size is not None: 
      self.embedding = ntorch.nn.Embedding(num_embeddings=params["VOCAB_SIZE"], 
                                           embedding_dim = self.embedding_size).spec("seqlen", "embedlen")
    else: 
      self.embedding = (ntorch.nn.Embedding.
                        from_pretrained(torch.eye(len(TEXT.vocab.itos))
                                       )
                       ).spec("seqlen", "embedlen")
      
      self.embedding.weight.requires_grad_(False)
      self.embedding_size = len(TEXT.vocab.itos)
    
    self.LSTM = (ntorch.nn.LSTM(self.embedding_size, self.hiddenlen, num_layers=self.num_layers)
                .spec("embedlen", "seqlen", name_out="hiddenlen")
                )
    self.linear = (ntorch.nn.Linear(self.hiddenlen + self.aa_info_size, 
                                    self.out_vocab)
                   .spec("hiddenlen", "vocablen")
                  )
    
  def set_to_eval(self):
    self.dropout.eval()
  
  def set_teacher_force(self, new_prob):
    self.teacher_force_prob = new_prob
    
  def forward(self, text, aa_info):
    ''' 
      Pass in context for the next amino acid
    '''
    
    # Reset for each new batch...
    h_0 = ntorch.zeros(text.shape["batch"], self.num_layers, self.hiddenlen, 
                        names=("batch", "layers", "hiddenlen")).to(device)
    c_0 = ntorch.zeros(text.shape["batch"], self.num_layers, self.hiddenlen, 
                        names=("batch", "layers", "hiddenlen")).to(device)
 
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
      for position in range(text.shape["seqlen"]): 
        text_embedding = self.embedding(model_input)
        hidden_states, (h_n, c_n) = self.LSTM(text_embedding, (h_0, c_0))

        output = self.linear_dropout(hidden_states)
        aa_info_subset = aa_info[{"seqlen" : slice(position, position+1)}]
        output = ntorch.cat([output, aa_info_subset], dim="hiddenlen")

        output = self.linear(output)
        outputs.append(output)

        # Define next input... 
        if random.random() < self.teacher_force_prob: 
          model_input = text[{"seqlen" : slice(position, position+1)}]
        else: 
          # TODO: Should we be masking this output?
          model_input = output.argmax("vocablen")
          
      output = ntorch.cat(outputs, dim="seqlen")
    return output
  
  
class AA_ONLY(ntorch.nn.Module):
  ''' 
  Simple model to predict the output codons using only the input amino acid
  '''
  def __init__(self, params):
    super(AA_ONLY, self).__init__()
    self.out_vocab = params["OUT_VOCAB"]
    self.aa_info_size = params["AA_COMPRESS_SIZE"]
    self.linear = (ntorch.nn.Linear(self.aa_info_size, 
                                    self.out_vocab)
                   .spec("hiddenlen", "vocablen")
                  )
    
  def forward(self, text, aa_info):
    ''' 
      Pass in context for the next amino acid
    '''

    output = aa_info
    output = self.linear(output)
    return output
  

