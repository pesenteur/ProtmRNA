# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 15:56:34 2023

@author: whuxu
"""
import tensorflow as tf
import os

def gelu(x):
    """Implementation of the gelu activation function.
    """
    return x * 0.5 * (1.0 + tf.math.erf(x / tf.sqrt(2.0)))

class FeedForward(tf.keras.layers.Layer):
    def __init__(self, dim, d_ffn):
        super(FeedForward, self).__init__()
        self.fc1 = tf.keras.layers.Dense(d_ffn)
        self.fc2 = tf.keras.layers.Dense(dim)
        
    def call(self, x):
        x = gelu(self.fc1(x))
        x = self.fc2(x)
        return x
    
loss_mse = tf.keras.losses.MeanSquaredError()
class MLPHead(tf.keras.Model):
    def __init__(self, d_out, d_model, d_ffn, drop_rate):
        super(MLPHead, self).__init__()
        
        self.d_model = d_model
        self.d_out = d_out
        self.ffn_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.ffn = FeedForward(d_model, d_ffn)     
        self.dropout = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.final_layer = tf.keras.layers.Dense(d_out)

    def call(self, inputs, inputs_mask, label, training=False):
        b = inputs.shape[0]
        assert inputs.shape[-1] == self.d_model
        
        decode_out = self.dropout(inputs, training=training)
        decode_out = self.ffn(decode_out)
        decode_out = tf.nn.swish(decode_out)
        decode_out = self.ffn_norm(decode_out)
        decode_out = self.dropout2(decode_out, training=training)
        
        inputs_mask = tf.expand_dims(inputs_mask, axis=-1)
        decode_out = decode_out * inputs_mask

        decode_out = tf.reduce_sum(decode_out, axis=1)
        assert decode_out.shape == (b, self.d_model)
        sum_mask = tf.reduce_sum(inputs_mask, axis=1)
        decode_out = decode_out / (sum_mask + 1e-6)
        assert decode_out.shape == (b, self.d_model)
        
        logit = self.final_layer(decode_out)
        out = logit
        
        assert label.shape == (b,)
        loss = loss_mse(label, logit)
        return loss, tf.squeeze(out, -1)
    
    def load_model(self, name=None):
        print ("load_weights", name)
        self.load_weights(os.path.join('./models/', name))

    def save_model(self, name=None):
        print ("save_weights", name)
        self.save_weights(os.path.join('./models/', name),
                          save_format="h5")
