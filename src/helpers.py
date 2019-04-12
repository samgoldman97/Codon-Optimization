'''
    helpers.py 
    April 10, 2019
    Helper functions for data import
'''


def hello_world():
    print("hello_world")

def read_in_fa(filename):
    ''' 
    Read in the fa file from ensembl that contains the cDNA
    Note: For ease, we filter out all gene sequences that are not multiples of 3 nucleotide sequences 
    
    '''
    data = []
    names = []
    with open(filename, "r") as fp:
        cur_string = None
        cur_name = ""
        temp_set = set()
        for line in fp:
            if line[0] == ">":
                # Beginning of line

                gene_name = line.strip().split("gene:")[1].split(" ")[0]
                cur_name=gene_name
                # Add to gene list if gene is a multiple of 3 nucleotides
                if cur_string and (len(cur_string) % 3 == 0):
                    data.append(cur_string)
                    names.append(gene_name)
                    
                cur_string = ""
            else:
                cur_string += (line.strip())
                


    return (names, data)
            
    

