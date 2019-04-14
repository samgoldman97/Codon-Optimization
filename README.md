# Codon-Optimization
Spring 2019 CS287 Final Project. We take a neural based approach to the task of genetic codon optimization.

## References

### Codon optimization literature
[Tian, Jian, et al. "Predicting synonymous codon usage and optimizing the heterologous gene for expression in E. coli." Scientific reports 7.1 (2017): 9926.](https://www.nature.com/articles/s41598-017-10546-0)

[Tian, Jian, et al. "Presyncodon, a Web Server for Gene Design with the Evolutionary Information of the Expression Hosts." International journal of molecular sciences 19.12 (2018): 3872.](https://www.mdpi.com/1422-0067/19/12/3872/htm)

[Goodman, Daniel B., George M. Church, and Sriram Kosuri. "Causes and effects of N-terminal codon bias in bacterial genes." Science 342.6157 (2013): 475-479.](https://science-sciencemag-org.ezp-prod1.hul.harvard.edu/content/sci/342/6157/475.full.pdf) (requires HarvardKey login)

### Possible benchmark tasks

[Becq, Jennifer, Cécile Churlaud, and Patrick Deschavanne. "A benchmark of parametric methods for horizontal transfers detection." PLoS One 5.4 (2010): e9989.](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009989)

### NLP References

[Mueller, Jonas, David Gifford, and Tommi Jaakkola. "Sequence to better sequence: continuous revision of combinatorial structures." Proceedings of the 34th International Conference on Machine Learning-Volume 70. JMLR. org, 2017.](http://proceedings.mlr.press/v70/mueller17a/mueller17a.pdf)

#### Style Transfer  
[Zhao, Yanpeng, et al. "Language Style Transfer from Non-Parallel Text with Arbitrary Styles." (2018).](https://openreview.net/pdf?id=B1NKuC6SG)

[Shen, Tianxiao, et al. "Style transfer from non-parallel text by cross-alignment." Advances in neural information processing systems. 2017.](https://papers.nips.cc/paper/7259-style-transfer-from-non-parallel-text-by-cross-alignment.pdf)

[Prabhumoye, Shrimai, et al. "Style transfer through back-translation." arXiv preprint arXiv:1804.09000 (2018).](https://arxiv.org/abs/1804.09000)

[Zhao, Yanpeng et al. "Language Style Transfer from Sentences with Arbitrary Unknown Styles." arXiv preprint arXiv:1808.04071v1 (2018).](https://arxiv.org/pdf/1808.04071.pdf)

[Fu, Zhenxin et al. "Style Transfer in Text: Exploration and Evaluation." arXiv preprint arXiv:1711.06861v2 (2017).](https://arxiv.org/abs/1711.06861) 

### Other ideas  

#### Meta-learning  
Meta-learning by optimization has been hot lately with the development of methods based on gradient descent. Of primary note is Chelsea Finn et al.'s Model-Agnostic Meta-Learning (MAML), which aims to train models to do well for a variety of tasks. 

To look at our goal under a MAML framework, we could imagine coming up with an optimal codon for each host organism to be a task, and we want to be learn how to generalize across a whole set of organisms. This makes language modeling slightly more interesting.   
* [Blog Post 1](https://bair.berkeley.edu/blog/2017/07/18/learning-to-learn/)  
* [Blog Post 2](https://towardsdatascience.com/model-agnostic-meta-learning-maml-8a245d9bc4ac)  
* [Finn et al. "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks." ICML 2017.](https://arxiv.org/pdf/1703.03400.pdf)  

Towards NLP in particular:    
* [Gu, Jiatao et al. "Meta-Learning for Low-Resource Neural Machine Translation." EMNLP 2018.](https://arxiv.org/pdf/1808.08437.pdf)


## TODO

## Misc. Notes

## Data Sources

[Orthologous genes](http://www.pathogenomics.sfu.ca/ortholugedb)

[Ensembl download cDNA genomes](https://bacteria.ensembl.org/info/website/ftp/index.html)

[Ensembl genome download ftp link for e.coli](ftp://ftp.ensemblgenomes.org/pub/release-43/bacteria//fasta/bacteria_0_collection/escherichia_coli_str_k_12_substr_mg1655/cdna/)

[NCBI E.coli search](https://www.ncbi.nlm.nih.gov/search/all/?term=escherichia%20coli)
