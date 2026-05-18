import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from model import MLPHead
from utils import DataGenerator

import time
import numpy as np
from tensorflow import keras

import tensorflow as tf
import warnings

from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def model_train(x, x_mask, y, my_model_head, 
                tr_head, optimizer_head):
    
    with tf.GradientTape() as tape:
        loss, out = my_model_head(x, x_mask, y, training=True)
    
    trainable_variables = tr_head
    gradients = tape.gradient(loss, trainable_variables)
    gradients, global_norm = tf.clip_by_global_norm(gradients, clip_norm=1.0)
    optimizer_head.apply_gradients(zip(gradients, tr_head))      
    return loss, out

def model_infer(x, x_mask, y, my_model_head):
    loss, out = my_model_head(x, x_mask, y, training=False)
    return loss, out
    
def main():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    max_len = 1280
    d_model = 1280
    data_train = DataGenerator(batch_size=8,
                               data_list="../data/mRFP/train.pkl.gz",
                               max_len=max_len,
                               d_model=d_model,
                               shuffle=True)

    data_val = DataGenerator(batch_size=8,
                             data_list="../data/mRFP/dev.pkl.gz",
                             max_len=max_len,
                             d_model=d_model,
                             shuffle=False)

    data_test = DataGenerator(batch_size=1,
                              data_list="../data/mRFP/test.pkl.gz",
                              max_len=max_len,
                              d_model=d_model,
                              shuffle=False)
    
    my_model_head = MLPHead(d_out=1, 
                            d_model=d_model, 
                            d_ffn=d_model*2,
                            drop_rate=0.25)
    
    my_model_head(inputs=np.zeros((1, max_len, d_model)), 
                  inputs_mask=np.zeros((1, max_len)).astype(np.float32),
                  label=np.zeros([1]))
    print (len(my_model_head.trainable_variables))
    
    learning_rate_head = 1.e-3
    lr_head = tf.Variable(tf.constant(learning_rate_head), name='lr', trainable=False)
    
    optimizer_head = keras.optimizers.Adam(learning_rate=lr_head)    
    tr_head = my_model_head.trainable_variables
    
    epochs = 100
    best_epoch = 0
    best_acc = 0
    count = 0
    for epoch in range(epochs):
        print('LR:', lr_head.numpy(), "count:", count)
        start_time = time.time()
        losss = [] 
        for step, item in enumerate(data_train):

            x, x_mask, x_len, y = item
            x_len = np.max(x_len)
            
            x = x[:,:x_len]
            x_mask = x_mask[:,:x_len]
            assert x.shape[-1] == d_model
            
            loss, out = model_train(x, x_mask, y, my_model_head, 
                                    tr_head, optimizer_head)
            losss.append(loss)

            run_time = time.time() - start_time
            start_time = time.time()
                
        my_model_head.save_model(name="head_" + str(epoch)+"_weight.h5")   
            
        start_time = time.time()
        trues = []
        preds = []
        losss = []
        for step, item in enumerate(data_val):

            x, x_mask, x_len, y = item
            x_len = np.max(x_len)
            
            x = x[:,:x_len]
            x_mask = x_mask[:,:x_len]
            assert x.shape[-1] == d_model
            
            loss, out = model_infer(x, x_mask, y, my_model_head)
            trues.extend(y.numpy())
            preds.extend(out.numpy())
            losss.append(loss)
        
        trues = np.array(trues)
        preds = np.array(preds)
        pearson_corr, p_person = pearsonr(trues, preds)
        spearman_corr, p_spearman = spearmanr(trues, preds)
        print('Val, Epoch: %d, pearson_corr: %3.3e, spearman_corr: %3.3e, time: %3.3f'
              % (epoch, pearson_corr, spearman_corr, run_time)) 
        
        indicator = spearman_corr
        if indicator > best_acc:
            best_epoch = epoch
            best_acc = indicator
            my_model_head.save_model(name="head_best_weight.h5")   
            print ("best_epoch", best_epoch, "best_acc", best_acc)
        else:
            lr_head.assign(lr_head/2)
            count += 1
        
        if count == 5:
            print ("best_epoch", best_epoch)
            break
        
    my_model_head.load_model(name="head_best_weight.h5")  
    trues = []
    preds = []
    losss = []
    for step, item in enumerate(data_test):

        x, x_mask, x_len, y = item
        x_len = np.max(x_len)
        
        x = x[:,:x_len]
        x_mask = x_mask[:,:x_len]
        assert x.shape[-1] == d_model
        
        loss, out = model_infer(x, x_mask, y, my_model_head)
        trues.extend(y.numpy())
        preds.extend(out.numpy())
        losss.append(loss)

    trues = np.array(trues)
    preds = np.array(preds)
    pearson_corr, p_person = pearsonr(trues, preds)
    spearman_corr, p_spearman = spearmanr(trues, preds)
    print('Best, Epoch: %d, pearson_corr: %3.3e, spearman_corr: %3.3e, time: %3.3f'
          % (best_epoch, pearson_corr, spearman_corr, run_time)) 
    
    with open("results", 'w') as fw:
        assert len(trues) == len(preds)
        print (len(trues))
        for t,p in zip(trues, preds):
            fw.writelines(str(t) + "," + str(p) + "\n")
    print ("Done")         
        
if __name__ == '__main__':
    print('Running training through horovod.run')
    main()