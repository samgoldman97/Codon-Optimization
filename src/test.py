'''
    test.py
    April 11, 2019 
    Test helper functions
'''

from helpers import *
import os
import sys
sys.path.append("../sequences/")

if __name__=="__main__":
    sequence_file_name="../sequences/Escherichia_coli_str_k_12_substr_mg1655.ASM584v2.cdna.all.fa"
    names,data = read_in_fa(sequence_file_name)
    print(len(names), len(data))
    ## assessing the length of the data..

    




