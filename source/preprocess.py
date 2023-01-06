
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


def create_one_hot(df):
    """
    function description: create one hot encoded tensors
    
    """
    # We will use this dictionary to map each character to an integer so that it can be used as an input to our ML models:
    dict_int2aa = {0:"A",1:"C",2:"D",3:"E",4:"F",5:"G",6:"H",7:"I",8:"K",9:"L",10:"M",11:"N",12:"P",13:"Q",14:"R",15:"S",16:"T",17:"V",18:"W",19:"Y",20:"-"}

    token2int = {x:i for i, x in enumerate('ACDEFGHIKLMNPQRSTVWY-')}

    train_inputs = preprocess_inputs(df, token2int)
    
    sample = len(df)
    positions = len(df.iloc[0].Sequence)
    
    hot_inputs = np.zeros((train_inputs.shape[0], positions, 21))
    
    for sample in range(train_inputs.shape[0]):
        hot_encoded = OneHot_encode(train_inputs, sample)
        hot_inputs[sample, :,:] = hot_encoded
        
    return hot_inputs





