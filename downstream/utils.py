import numpy as np
import tensorflow as tf
# import pickle
import pickle5 as pickle
import random
import gzip

def read_fasta(data_list, d_model):
    contents = []
    with gzip.open(data_list, "rb") as f:
        data = pickle.load(f)
    
    for i, item in enumerate(data):
        feature = item["feature"]
        label = item["label"]
        contents.append([feature, float(label)])
    return contents
        
class DataGenerator(tf.data.Dataset):
    def _generator(data_list:str, d_model:int, max_len:int=2048, shuffle:bool=False):
        
        data_list = data_list.decode('utf-8')

        print ("read file path...")
        contents = read_fasta(data_list, d_model)
        print ("file length: ", len(contents))

        random.Random(2026).shuffle(contents)
        
        if shuffle: np.random.shuffle(contents)
        
        for sample_idx in range(len(contents)):

            feat, label = contents[sample_idx]
            
            inputs = np.zeros((max_len, d_model))
            x_len = feat.shape[0]
            inputs[:x_len] = feat
            
            inputs_mask = np.zeros(max_len)
            inputs_mask[:x_len] = 1
            
            yield inputs, inputs_mask, x_len, label
    
    def __new__(cls, batch_size:int = 32, data_list:str = '',
                     d_model:int = 0, max_len:int=2048,
                     shuffle:bool = False):

        ds = tf.data.Dataset.range(1)
        ds = ds.interleave(lambda x: tf.data.Dataset.from_generator(
                                     cls._generator,
                                     output_types = (tf.float32, tf.float32, tf.int32, tf.float32),
                                     output_shapes = ((max_len, d_model), 
                                                      (max_len,),
                                                      (),
                                                      (),),
                                     args = (data_list, d_model, max_len, shuffle)))
        return ds.batch(batch_size).prefetch(batch_size*4)  

