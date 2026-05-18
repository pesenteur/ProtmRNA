import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from model import ProtmRNA

import time
import numpy as np

import tensorflow as tf
import warnings

warnings.filterwarnings("ignore")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from utils import dict_res_to_idx
    
def read_fasta(file_path):
    """
    读取FASTA文件，返回序列ID和序列的字典
    
    Args:
        file_path: FASTA文件路径
    
    Returns:
        dict: 键为序列ID，值为序列字符串
    """
    sequences = {}
    current_id = None
    current_seq = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_id:
                    sequences[current_id] = ''.join(current_seq)
                current_id = line[1:]
                current_seq = []
            else:
                current_seq.append(line)
        
        if current_id:
            sequences[current_id] = ''.join(current_seq)
    return sequences
    
def main():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    my_model = ProtmRNA(n_layers=33, 
                        d_model=1280, 
                        n_heads=20, 
                        d_ffn=1280*4,
                        vocab_size=78)

    my_model(inputs=np.zeros((1, 1024)))
    my_model.load_model(name="./models/ProtmRNA_weights.h5")  
    print (len(my_model.trainable_variables))

    fasta_path = "/work/jpma/mRNAdataset/Data_fasta/CDS_fine_tune.fasta"
    mrna_sequences = read_fasta(fasta_path)
    print (len(mrna_sequences))
    
    for idx, name in enumerate(mrna_sequences):
        seq = mrna_sequences[name].split()
        print (idx, name, len(seq), seq)
        
        x = [dict_res_to_idx['<cls>']]
        for i in seq:
            if i in dict_res_to_idx:
                x.append(dict_res_to_idx[i])
            elif len(i) == 3:
                x.append(dict_res_to_idx['XXX'])
            else:
                assert len(i) == 1
                x.append(dict_res_to_idx['X'])
        x.append(dict_res_to_idx['<eos>'])
        
        x = np.array([x])
        logit, decode_out = my_model(x)  
        assert logit.shape == (1, len(seq)+2, 78)
    
if __name__ == '__main__':
    main()