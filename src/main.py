import torch
from models import * 
from helpers import *


# Use CPU for baseline...
device = torch.device("cpu") #torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__=="__main__": 
	fasta_in_file = "../data/ecoli.heg.fasta"
	csv_out_file = "../data/ecoli.heg.csv"
	convert_fasta_to_csv(file_name = fasta_in_file, out_file=csv_out_file)
	train, test, TEXT = load_csv_data(csv_out_file, device=device)
	AA_LABEL, index_table, codon_to_aa, codon_to_aa_index, mask_tbl = build_helper_tables(TEXT, device=device)

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

	res = get_prediction_iter(test, model, aa_compress, mask_tbl, device)
	output_list_of_res(res, TEXT, outfile="../outputs/predictions/temp.txt")






