# Codon-Optimization

A deep learning based approach to the task of genetic codon prediction and optimization. We proposee an LSTM-Transducer model for this task, gaining modest improvements in accuracy and perplexity in predicting codon choice over frequency-based methods. 

The models were tested on highly expressed genes of [E. coli MG1655](http://genomes.urv.cat/HEG-DB/) and [Humans hg19](https://www.tau.ac.il/~elieis/HKG/) and implemented with PyTorch wrapper, namedtensor. 

Example of how to train our model on this data is shown in src/main.py

