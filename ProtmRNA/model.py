# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 15:56:34 2023

@author: whuxu
"""
import numpy as np
import tensorflow as tf
from rotary_embedding import RotaryEmbedding
from utils import dict_res_to_idx

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, dict_res_to_idx['<pad>']), tf.float32)
    return seq[:, np.newaxis, np.newaxis, :]

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask

def create_mask(inp):
    look_ahead_mask = np.zeros((tf.shape(inp)[1], tf.shape(inp)[1]))
    dec_target_padding_mask = create_padding_mask(inp)
    combine_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return combine_mask

def scaled_dot_product_attention(q, k, v, mask):

    scaled_attention_logits = tf.matmul(q, k, transpose_b=True)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    scaled_attention_logits = tf.nn.softmax(scaled_attention_logits, axis=-1)
    
    output = tf.matmul(scaled_attention_logits, v)
    
    return output, scaled_attention_logits
    
class MutilHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MutilHeadAttention, self).__init__()
        
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % num_heads == 0
        self.depth = d_model // num_heads
        self.scaling = self.depth**-0.5

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

        self.rot_emb = RotaryEmbedding(dim=self.depth)
        
    def call(self, x, mask):
        
        q = x
        k = x
        v = x
        
        batch_size = tf.shape(q)[0]
        q_len = tf.shape(q)[1]
        kv_len = tf.shape(k)[1]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        q *= self.scaling

        q = tf.reshape(q, (batch_size, q_len, self.num_heads, self.depth))
        k = tf.reshape(k, (batch_size, kv_len, self.num_heads, self.depth))
        v = tf.reshape(v, (batch_size, kv_len, self.num_heads, self.depth))
        
        q, k = self.rot_emb(q, k)
        
        q = tf.transpose(q, perm=[0, 2, 1, 3])
        k = tf.transpose(k, perm=[0, 2, 1, 3])
        v = tf.transpose(v, perm=[0, 2, 1, 3])
        
        output, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        output = tf.transpose(output, [0, 2, 1, 3])
        
        output = tf.reshape(output, (batch_size, -1, self.d_model))
        output = self.dense(output)
        
        return output, attention_weights
    
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
    
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ffn):
        super(DecoderLayer, self).__init__()

        self.mha = MutilHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ffn)

        self.self_attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.final_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)

    def call(self, x, combine_mask, training=False):

        residual = x
        x = self.self_attn_layer_norm(x)
        x, _ = self.mha(x, combine_mask)
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = self.ffn(x)
        x = residual + x

        return x
            
class Decoder(tf.keras.layers.Layer):
    def __init__(self, n_layers, d_model, n_heads, d_ffn, vocab_size):
        super(Decoder, self).__init__()

        self.n_layers = n_layers
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        self.embedding = tf.keras.layers.Dense(d_model, use_bias=False)
        self.decoder_layers = [DecoderLayer(d_model, n_heads, d_ffn)
                               for _ in range(n_layers)]
        
    def call(self, inputs, combine_mask, training=False):

        assert len(tf.shape(inputs)) == 2
        padding_idx = dict_res_to_idx['<pad>']
        mask_idx = dict_res_to_idx['<mask>']
        
        padding_mask = tf.math.equal(inputs, padding_idx)  # B, T

        x = self.embedding(tf.one_hot(tf.cast(inputs, tf.int32), self.vocab_size))

        x = tf.where((inputs == mask_idx)[:, :, None], 0.0, x)

        mask_ratio_train = 0.15 * 0.8
        src_lengths = tf.reduce_sum(tf.cast(~padding_mask, x.dtype), axis=-1)
        masked_tokens = inputs == mask_idx
        mask_ratio_observed = tf.math.count_nonzero(masked_tokens, dtype=tf.float32, axis=-1)/ src_lengths
        x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]

        x = x * (1 - tf.cast(tf.expand_dims(padding_mask, -1), dtype=x.dtype))
            
        h = x
        for i in range(self.n_layers):
            h = self.decoder_layers[i](h, combine_mask, training)
        return h

class ProtmRNA(tf.keras.Model):
    def __init__(self, n_layers, d_model, n_heads, d_ffn, vocab_size):
        super(ProtmRNA, self).__init__()

        self.decoder = Decoder(n_layers, d_model, n_heads, d_ffn, vocab_size)
        
        self.ffn_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)

        self.lm_head = RobertaLMHead(
            embed_dim=d_model,
            output_dim=vocab_size,
        )
        
    def call(self, inputs, training=False):
        tar_input = inputs
        combine_mask = create_mask(tar_input)
        decode_out = self.decoder(tar_input, combine_mask, training)
        decode_out = self.ffn_norm(decode_out)
        logit = self.lm_head(decode_out, self.decoder.embedding.weights[0])
        out = tf.argmax(logit, -1)
        return out, decode_out
        
    def load_model(self, name=None):
        print ("load_lm_weights", name)
        self.load_weights(name)
        
class RobertaLMHead(tf.keras.layers.Layer):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim):
        super().__init__(name='lm_head')
        self.dense = tf.keras.layers.Dense(embed_dim)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.bias = self.add_weight("bias", shape=(output_dim,), initializer="zeros", trainable=True)
    
    def call(self, features, decoder):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)
        x = tf.matmul(x, decoder, transpose_b=True) + self.bias
        return x
