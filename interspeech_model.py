#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Parcollet Titouan

# Imports

import complexnn
from   complexnn                            import QuaternionDense, QuaternionConv2D, QuaternionConv1D

import numpy                                as     np
import os, pdb, socket, sys, time           as     T

import tensorflow                           as     tf

from   tensorflow.keras.callbacks           import Callback, ModelCheckpoint, LearningRateScheduler

from   tensorflow.keras.initializers        import Orthogonal
from   tensorflow.keras.layers              import Layer, Dropout, AveragePooling1D, AveragePooling2D, AveragePooling3D, MaxPooling2D, \
                                                    add, Add, concatenate, Concatenate, Input, Flatten, Dense, Convolution2D, \
                                                    BatchNormalization, Activation, Reshape, ConvLSTM2D, Conv2D, Lambda, Permute, \
                                                    TimeDistributed, SpatialDropout1D, PReLU
                                                    
from   tensorflow.keras.models              import Model, load_model, save_model
# from   tensorflow.keras.optimizers          import SGD, Adam, RMSprop

from   tensorflow.keras.regularizers        import l2
from   tensorflow.keras.utils               import to_categorical
import tensorflow.keras.backend             as     K
import tensorflow.keras.models              as     KM

from CTCModel import CTCModel

# from tensorflow.keras.utils.training_utils  import multi_gpu_model
# import logging                              as     L
# from keras.backend.tensorflow_backend       import set_session

""" CTC Loss - Implemented differently
#
# CTC Loss
#

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
"""

#
# Get Model
#

def interspeech_TIMIT_Model(d):
    n             = d.num_layers
    sf            = d.start_filter
    activation    = d.act
    advanced_act  = d.aact
    drop_prob     = d.dropout
    inputShape    = (778,3,40)  # (3,41,None)  
    filsize       = (3, 5)
    Axis   = 1
    
    if advanced_act != "none":
        activation = 'linear'

    convArgs      = {
            "activation":               activation,
            "data_format":              "channels_first",
            "padding":                  "same",
            "bias_initializer":         "zeros",
            "kernel_regularizer":       l2(d.l2),
            "kernel_initializer":       "random_uniform",
            }
    denseArgs     = {
            "activation":               d.act,        
            "kernel_regularizer":       l2(d.l2),
            "kernel_initializer":       "random_uniform",
            "bias_initializer":         "zeros",
            "use_bias":                 True
            }
    
    #### Check kernel_initializer for quaternion model ####
    if d.model == "quaternion":
        convArgs.update({"kernel_initializer": d.quat_init})  

    #
    # Input Layer & CTC Parameters for TIMIT
    #
    if d.model == "quaternion":
        I    = Input(shape=(778,4,40)) # Input(shape=(4,41,None))
    else:
        I = Input(shape=inputShape)

    labels = Input(name='the_labels', shape=[None], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    #
    # Input stage:
    #
    if d.model == "real":
        O = Conv2D(sf, filsize, name='conv', use_bias=True, **convArgs)(I)
        if advanced_act == "prelu":
            O = PReLU(shared_axes=[1,0])(O)
    else:
        O = QuaternionConv2D(sf, filsize, name='conv', use_bias=True, **convArgs)(I)
        if advanced_act == "prelu":
            O = PReLU(shared_axes=[1,0])(O)
    #
    # Pooling
    #
    O = MaxPooling2D(pool_size=(1, 3), padding='same')(O)


    #
    # Stage 1
    #
    for i in range(0,n // 2):
        if d.model=="real":
            O = Conv2D(sf, filsize, name='conv'+str(i), use_bias=True,**convArgs)(O)
            if advanced_act == "prelu":
                O = PReLU(shared_axes=[1,0])(O)
            O = Dropout(drop_prob)(O)
        else:
            O = QuaternionConv2D(sf, filsize, name='conv'+str(i), use_bias=True, **convArgs)(O)
            if advanced_act == "prelu":
                O = PReLU(shared_axes=[1,0])(O)
            O = Dropout(drop_prob)(O)
    
    #
    # Stage 2
    #
    for i in range(0,n // 2):
        if d.model=="real":
            O = Conv2D(sf*2, filsize, name='conv'+str(i+n/2), use_bias=True, **convArgs)(O)
            if advanced_act == "prelu":
                O = PReLU(shared_axes=[1,0])(O)
            O = Dropout(drop_prob)(O)
        else:
            O = QuaternionConv2D(sf*2, filsize, name='conv'+str(i+n/2), use_bias=True, **convArgs)(O)
            if advanced_act == "prelu":
                O = PReLU(shared_axes=[1,0])(O)
            O = Dropout(drop_prob)(O)
    
    #
    # Permutation for CTC 
    #
    print("Last Q-Conv2D Layer (output): ", K.int_shape(O))
    print("Shape tuple: ", K.int_shape(O)[0], K.int_shape(O)[1], K.int_shape(O)[2], K.int_shape(O)[3])
    #### O = Permute((3,1,2))(O)
    #### print("Last Q-Conv2D Layer (Permute): ", O.shape)
    # O = Lambda(lambda x: K.reshape(x, (K.shape(x)[0], K.shape(x)[1],
    #                                    K.shape(x)[2] * K.shape(x)[3])),
    #            output_shape=lambda x: (None, None, x[2] * x[3]))(O)
    
    # O = Lambda(lambda x: K.reshape(x, (K.int_shape(x)[0], K.int_shape(x)[1],
    #                                    K.int_shape(x)[2] * K.int_shape(x)[3])),
    #            output_shape=lambda x: (None, None, x[2] * x[3]))(O)
    
    O = tf.keras.layers.Reshape(target_shape=[-1, K.int_shape(O)[2] * K.int_shape(O)[3]])(O)
    
    #
    # Dense
    #
    print("Q-Dense input: ", K.int_shape(O))
    if d.model== "quaternion":
        print("first Q-dense layer: ", O.shape)
        O = TimeDistributed( QuaternionDense(256,  **denseArgs))(O)
        if advanced_act == "prelu":
            O = PReLU(shared_axes=[1,0])(O)
        O = Dropout(drop_prob)(O)
        O = TimeDistributed( QuaternionDense(256,  **denseArgs))(O)
        if advanced_act == "prelu":
            O = PReLU(shared_axes=[1,0])(O)
        O = Dropout(drop_prob)(O)
        O = TimeDistributed( QuaternionDense(256,  **denseArgs))(O)
        if advanced_act == "prelu":
            O = PReLU(shared_axes=[1,0])(O)
    else:
        O = TimeDistributed( Dense(1024, **denseArgs))(O)
        if advanced_act == "prelu":
            O = PReLU(shared_axes=[1,0])(O)
        O = Dropout(drop_prob)(O)
        O = TimeDistributed( Dense(1024, **denseArgs))(O)
        if advanced_act == "prelu":
            O = PReLU(shared_axes=[1,0])(O)
        O = Dropout(drop_prob)(O)
        O = TimeDistributed( Dense(1024, **denseArgs))(O)
        if advanced_act == "prelu":
            O = PReLU(shared_axes=[1,0])(O)

    # pred = TimeDistributed( Dense(61,  activation='softmax', kernel_regularizer=l2(d.l2), use_bias=True, bias_initializer="zeros", kernel_initializer='random_uniform' ))(O)
    pred = TimeDistributed( Dense(61, kernel_regularizer=l2(d.l2), use_bias=True, bias_initializer="zeros", kernel_initializer='random_uniform' ))(O)
    
    output = Activation('softmax', name='softmax')(pred)

    network = CTCModel([I], [output])
    # network.compile(Adam(lr=0.0001))
    """ CTC Loss - Implemented differently
    if d.ctc:
        # CTC For sequence labelling
        O = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([pred, labels,input_length,label_length])
    
        # Creating a function for testing and validation purpose
        val_function = K.function([I],[pred])
        
        # Return the model
        return Model(inputs=[I, input_length, labels, label_length], outputs=O), val_function
    """
    
    # return Model(inputs=I, outputs=pred)
    return network
    
