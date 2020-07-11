# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 08:48:17 2020

@author: Sophon
"""

import numpy as np
from tensorflow.keras.layers import Input, Add, Dense, Activation,  BatchNormalization, Flatten
from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import plot_model
from tensorflow.keras.initializers import glorot_uniform

def convolutional_block(x, f_siz, ker_siz):
    x = Conv2D(filters = f_siz, kernel_size = (ker_siz, ker_siz), strides = (1, 1), padding = 'same', 
               name = 'conv_1', kernel_initializer = glorot_uniform(seed = 0))(x)
    x = BatchNormalization(axis = 3, name = 'bn1')(x)
    x = Activation('relu')(x)
    return x 

def identical_block(x, f_siz, layer_id):
    x_shortcut = x
    
    conv_name_base = 'res_conv_' + str(layer_id)
    bn_name_base = 'res_bn_' + str(layer_id)
    
    x = Conv2D(filters = f_siz, kernel_size = (3, 3), strides = (1, 1), padding = 'same', 
               name = conv_name_base + 'a', kernel_initializer = glorot_uniform(seed = 0))(x)
    x = BatchNormalization(axis = 3, name = bn_name_base + 'a')(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters = f_siz, kernel_size = (3, 3), strides = (1, 1), padding = 'same', 
               name = conv_name_base + 'b', kernel_initializer = glorot_uniform(seed = 0))(x)
    x = BatchNormalization(axis = 3, name = bn_name_base + 'b')(x)
    
    x = Add()([x, x_shortcut])
    x = Activation('relu')(x)
    
    return x

res_layer_num = 3

def resNet(input_shape):
    x_input = Input(input_shape)
    x = x_input
    
    x = convolutional_block(x, 16, 3)
    for res_idx in range(res_layer_num):
        x = identical_block(x, 16, res_idx)
    
    head_p = x
    # value head
    x = Conv2D(filters = 1, kernel_size = (1, 1), strides = 1, padding = 'same',
               name = 'val_conv', kernel_initializer = glorot_uniform(seed = 0))(x)
    x = BatchNormalization(axis = 3, name = 'val_bn')(x)
    x = Flatten()(x)
    x = Dense(16, activation = 'relu', name = 'fc_1')(x)
    x = Dense(1, activation = 'tanh', name = 'fc_2_out')(x)

    # policy head
    head_p = Conv2D(filters = 1, kernel_size = (1, 1), strides = 1, padding = 'same',
                    name = 'pol_conv', kernel_initializer = glorot_uniform(seed = 0))(head_p)
    head_p = BatchNormalization(axis = 3, name = 'pol_bn')(head_p)
    head_p = Flatten()(head_p)
    head_p = Dense(9, activation = 'softmax', name = 'pol_out')(head_p)
    
    # get model
    model = Model(x_input, [x, head_p])
    return model
    
model = resNet(input_shape = (3, 3, 4))
model.compile(optimizer = 'rmsprop', loss = ['mean_squared_error', 'categorical_crossentropy'])
model.summary()

model.save_weights('./model_weights_only.h5')
json_config = model.to_json()
with open('./model_json.json', 'w') as json_file:
    json_file.write(json_config)

plot_model(model, to_file='./net.png', show_shapes = True)