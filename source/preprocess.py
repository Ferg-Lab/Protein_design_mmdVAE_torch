
"""

functions used to preprocess data ...


"""
from Bio import SeqIO

import pandas as pd
import numpy as np


# Read the fasta file
def get_seq(filename, get_header = False):
    assert filename.endswith('.fasta'), 'Not a fasta file.'
    
    records = list(SeqIO.parse(filename, "fasta"))
    records_seq = [i.seq for i in records]
    headers = [i.description for i in records]
    if get_header == True:
        return records_seq, headers
    else:
        return records_seq
    

def create_MSA(Seq_samples, samples, positions):
    single_seq, seq_list = [], []
    for jj in range(samples):
        single_seq = "".join((str(ii) for ii in Seq_samples[jj]))
        seq_list.append(single_seq)
    return seq_list


def pandas_list_to_array(df):
    return np.transpose(np.array(df.values.tolist()), (0, 2, 1))


def preprocess_inputs(df, token2int, cols=['Sequence']):
    return pandas_list_to_array(df[cols].applymap(lambda seq: [token2int[x] for x in seq]))

def OneHot_encode(data,sample):
    b = np.eye(21)[data[sample]].transpose(0,2,1)[:,:,0]
    return b








