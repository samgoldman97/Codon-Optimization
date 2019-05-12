from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqUtils import CodonUsage
from Bio.SeqUtils import IUPACData

import random
import csv



def count_codons(DNA_seq):
  ''' Calculate the counts of each codon given a CDS
  
  Args:
    dna sequence in indexable form
    
  Returns:
    dict{str, int}: dictionary with codons as key and corresponding number of occurences
  
  '''
  
  codons_dict = CodonUsage.CodonsDict.copy()
  for codon_start in range(0, len(DNA_seq), 3):
    codons_dict[str(DNA_seq[codon_start:codon_start+3])] += 1
   
  return codons_dict


  def count_codons_fasta(fasta_file):
  ''' Calculate the counts of each codon given a set of CDS
  
  Args:
    Fasta file 
    
  Returns:
    dict{str, int}: dictionary with codons as key and corresponding number of occurences
  
  '''
  
  codons_dict = CodonUsage.CodonsDict.copy()
  
  for record in SeqIO.parse(fasta_file, "fasta"):
    seq = record.seq
    if len(seq) % 3 != 0:
      continue
      
    #count the codons for this sequence
    for codon_start in range(0, len(seq), 3):
      codons_dict[str(seq[codon_start:codon_start+3])] += 1
   
  return codons_dict

def calculate_codon_frequency(codon_counts):
  ''' Calculate the counts of each codon given a set of CDS
  
  Args:
    codon usage table
    
  Returns:
    dict{str, float}: dictionary with codons as key and corresponding 
    frequency of codon for AA
  
  '''
  codon_freqs = CodonUsage.CodonsDict.copy()
  
  for _, synonymous_codons in CodonUsage.SynonymousCodons.items():
    total_AA_count = sum([codon_counts[codon] for codon in synonymous_codons])
    
    if total_AA_count == 0:
      continue
      
    for codon in synonymous_codons:
      codon_freqs[codon] = codon_counts[codon] / total_AA_count
  
  return codon_freqs

def cds_translate(dna_seq):
  '''Translates CDS to protein
  
  Args:
    single DNA seq
    
  Returns:
    corresponding amino acid sequence
  '''
  
  return dna_seq.translate()

def sample_from_weighted_dict(codon_freq):
  '''Given a frequency dictionary, samples a key based on the frequency 
  
  Args:
    dict{str, float} with codon and corresponding frequency (within AA)
    
  Returns:
    sampled codon 
  '''
  #a random num between 0 and 1 is generated, and the corresponding codon in that region is outputted
  random_val = random.random()
  total = 0.0
  for key, val in codon_freq.items():
    total += val
    if total > random_val:
      return key
    
  assert False, "No codon selected"
    
    

def reverse_translate(aa_seq, codon_table, rule="most_frequent"):
  '''Reverse translates given AA sequence to CDS form using most frequent 
  
  Args:
    amino acid sequence
    
  Returns:
    corresponding CDS
  '''
  
  output_sequence = []
  
  #remove stop codon tag
  stripped_aa_seq = aa_seq.rstrip("*")
  
  for aa in list(stripped_aa_seq):
    three_letter_aa = IUPACData.protein_letters_1to3[aa].upper()
    synonymous_codons = CodonUsage.SynonymousCodons[three_letter_aa]
    
    #sub-dictionary with frequency of one amino acid of interest
    aa_codon_dict = {codon:codon_table[codon] for codon in synonymous_codons}
    
    if rule == "most_frequent":
      codon = max(aa_codon_dict, key=aa_codon_dict.get)
      output_sequence.append(codon)
    
    elif rule == "sample_frequency":
      codon = sample_from_weighted_dict(aa_codon_dict)
      output_sequence.append(codon)
    
  return ("".join(output_sequence))
      

