# Codon-Optimization

A deep learning based approach to the task of genetic codon prediction and optimization. We propose an LSTM-Transducer model for this task, gaining modest improvements in accuracy and perplexity in predicting codon choice over frequency-based methods. 

This was originally implemented as an undergraduate project in Google Colab using a PyTorch wrapper, namedtensor for Harvard's CS287r Machine Learning for Natural Language Processing Course. After this work was presented as a [poster at MLCB 2019](https://mlcb.github.io/mlcb2019_proceedings/papers/paper_29.pdf), the code was revised with better coding practices for readability and reproducibility \*. Lastly, no model generated sequences have been experimentaly tested for expression in the lab against frequency baselines.

## Data 

The models were tested on highly expressed genes of [E. coli MG1655](http://genomes.urv.cat/HEG-DB/) and [Humans hg19](https://www.tau.ac.il/~elieis/HKG/). The highly expressed gene set in data (data/ecoli.heg.fasta and data/human_HE.fasta) can be used directly to train a new model or any set of transcripts in Fasta form can be used as input to the model. The script `src/download_human_genes.py` was used to resolve nucleotide sequences for the human housekeeping gene set. 

## Running the code

After downloading a set of transcripts in Fasta form for modeling from the above links and removing redundancy (e.g. using CD-Hit), a model can be trained and predictions generated on a random train/val/test split using the command: 


``` 
python src/main.py --data-file [data/datafile.fasta]
``` 

Different models can be selected over the codon layer using the `--codon-model-name` flag and over the amino acid layer using the `--aa-model-name` flag

To run baseline models: 

``` 
python src/main.py --data-file [data/datafile.fasta] --run-baselines
``` 

----
\* As this work is not being actively continued, while a `--gpu` flag is provided, this code has only been tested on CPU and has not yet been used to reproduce the results table or free energy analysis from the original colab experiments. 

