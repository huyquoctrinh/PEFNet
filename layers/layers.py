import tensorflow as tf 
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.layers import Lambda, Reshape, Add, Multiply, Conv2DTranspose, Activation
import tensorflow.keras.backend as K
from keras_cv_attention_models import attention_layers



def bn_act(inputs, activation='swish'):
    
    x = BatchNormalization()(inputs)
    if activation:
        x = Activation(activation)(x)
    
    return x



def conv_bn_act(inputs, filters, kernel_size, strides=(1, 1), activation='swish', padding='same'):
    
    x = Conv2D(filters, kernel_size=kernel_size, padding=padding)(inputs)
    x = bn_act(x, activation=activation)
    
    return x



def decode(inputs, filters, activation='swish', padding='same'):
    
    x = Conv2DTranspose(filters, kernel_size=(4, 4), strides=(2, 2), padding=padding)(inputs)
    x = bn_act(x, activation=activation)
    
    return x



def block(inputs, filters):
    
    x1 = conv_bn_act(inputs, filters, 1)
    x2 = conv_bn_act(inputs, filters, 3)
    x3 = conv_bn_act(inputs, filters, 5)
    
    return x1 + x2 + x3



def merge(l, filters=None):
    if filters is None:
        channel_axis = 1 if K.image_data_format() == "channels_first" else -1
        filters = l[0].shape[channel_axis]
    
    x = tf.keras.layers.Add()(l)
    x = block(x, filters)
    
    return x


def conv_pos_emb(inputs, filters):
    
    x = block(inputs, filters)
    pos_emb = attention_layers.PositionalEmbedding()
    emb = pos_emb(x)
    
    return emb

