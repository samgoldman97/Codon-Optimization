'''
make_codon_table.py
May 13, 2019

Make a codon table for a given fasta file and print it
'''

from codon_functions import *

file_name = "ecoli_cds.fna"
eColi_codon_counts = count_codons_fasta(file_name)
eColi_codon_table = calculate_codon_frequency(eColi_codon_counts)
for key, val in eColi_codon_table.items():
    print(key,val)
