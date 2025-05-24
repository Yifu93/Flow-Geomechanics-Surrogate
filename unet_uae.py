import numpy as np
from layers_3D_2D import *
from keras import backend as K
from keras.layers import Input, Flatten, Dense, Lambda, Reshape, concatenate, TimeDistributed, RepeatVector, ConvLSTM3D
from keras.models import Model
from keras.optimizers import Adam

def create_vae(input_shape):
    # Encoder
    input = Input(shape=input_shape, name='image')

    enc1 = conv_bn_relu(16, 3, 3, 3, stride=(2, 2, 2))(input)    
    enc2 = conv_bn_relu(32, 3, 3, 3, stride=(1, 1, 1))(enc1)    
    enc3 = conv_bn_relu(32, 3, 3, 3, stride=(2, 2, 2))(enc2)    
    enc4 = conv_bn_relu(64, 3, 3, 3, stride=(1, 1, 1))(enc3)

    x = res_conv(64, 3, 3, 3)(enc4)
    x = res_conv(64, 3, 3, 3)(x)
    
    encoder = Model(input, x, name='encoder')

    x    = res_conv(64, 3, 3, 3)(x)
    dec4 = res_conv(64, 3, 3, 3)(x)
    
    merge4 = concatenate([enc4, dec4], axis = -1)   
    dec3   = dconv_bn_nolinear(64, 3, 3, 3, stride=(1, 1, 1))(merge4)
    merge3 = concatenate([enc3, dec3], axis = -1)    
    dec2   = dconv_bn_nolinear(32, 3, 3, 3, stride=(2, 2, 2))(merge3)
    merge2 = concatenate([enc2, dec2], axis = -1)   
    dec1   = dconv_bn_nolinear(32, 3, 3, 3, stride=(1, 1, 1))(merge2)   
    merge1 = concatenate([enc1, dec1], axis = -1)  
    dec0   = dconv_bn_nolinear(16, 3, 3, 3, stride=(2, 2, 2))(merge1)
    
    output = Conv3D(1, (3, 3, 3), padding='same', activation=None)(dec0)
    
    base_model = Model(input, output)

    return base_model, encoder