# -*- coding: utf-8 -*-
""" @author: vince
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Parcollet Titouan
# Modified by: Francesco Vincelli

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
from   tensorflow.keras.optimizers          import SGD, Adam, RMSprop

from   tensorflow.keras.regularizers        import l2
from   tensorflow.keras.utils               import to_categorical
import tensorflow.keras.backend             as     K
import tensorflow.keras.models              as     KM


# ctc_loss
def ctc_loss(y_true, y_pred, input_length, label_length, real_y_true_ts):
    return K.ctc_batch_cost(real_y_true_ts, y_pred, input_length, label_length)

#
# Get Model
#

def ctc_interspeech_TIMIT_Model(d):
    n             = d.num_layers
    sf            = d.start_filter
    activation    = d.act
    advanced_act  = d.aact
    drop_prob     = d.dropout
    inputShape    = (778,3,40)  # (3,41,None)  
    filsize       = (3, 5)
    Axis   = 1
    
    max_num_class = 61
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
        
    # Others inputs for the CTC approach
    labels = Input(name='labels', shape=[None])
    input_length = Input(name='input_length', shape=[1])
    label_length = Input(name='label_length', shape=[1])
    
    label_length_input = Input((1,),name="label_length_input")
    pred_length_input = Input((1,),name="pred_length_input")
    y_true_input = Input((max_num_class,), name="y_true_input")


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
    
    conv_shape = O.get_shape()
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
        O = TimeDistributed( Dense(1024,  **denseArgs))(O)
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
    y_pred = TimeDistributed( Dense(61, activation="softmax", kernel_regularizer=l2(d.l2), use_bias=True, bias_initializer="zeros", kernel_initializer='random_uniform' ))(O)
    
    # output = Activation('softmax', name='softmax')(pred)
    # ctc_model = Model(inputs=[I, pred_length_input, label_length_input, y_true_input],outputs=output)
    
    #%%
    # the actual loss calc occurs here despite it not being an internal Keras loss function

    def ctc_lambda_func(args):
        y_pred, labels, input_length, label_length = args
        # the 2 is critical here since the first couple outputs of the RNN tend to be garbage:
        #y_pred = y_pred[:, 2:, :]
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
    
    #%%
    labels = Input(shape=[max_num_class], dtype='float32') # max_num_class=61
    input_length = Input(shape=[1], dtype='int64')
    label_length = Input(shape=[1], dtype='int64')
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    
    # clipnorm seems to speeds up convergence
    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    
    ctc_model = Model(inputs=[I, labels, input_length, label_length], outputs=loss_out)
    
    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    ctc_model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
    
    # return Model(inputs=I, outputs=pred)
    return ctc_model, conv_shape # , label_length_input, pred_length_input, y_true_input


""" Create a Quaternion CNN model
prams
    n_classes: number of classes for classification task
return
    model: Quaternion-CNN model
"""
def QCNN_model():
    kern = 8
    dropout_prob = 0.3
    advanced_act = 'prelu'
    max_str_len = 61
    
    denseArgs     = {   
            "kernel_regularizer":       l2(1e-5),
            "kernel_initializer":       "random_uniform",
            "bias_initializer":         "zeros",
            "use_bias":                 True
            }
    
    input_data = Input(shape=(778,4,40))
    
    # First QConv layer
    x = QuaternionConv2D(kern, 2, strides=(1,1), padding="same", use_bias=True)(input_data)
    x = PReLU(shared_axes=[1,0])(x)
    x = Dropout(dropout_prob)(x)
    
    # Second QConv layer    
    x = QuaternionConv2D(kern*2, 2, strides=(1,1), padding="same", use_bias=True)(x)
    x = PReLU(shared_axes=[1,0])(x)
    x = Dropout(dropout_prob)(x)
    
    # Third QConv layer
    x = QuaternionConv2D(kern*4, 2, strides=(1,1), padding="same", use_bias=True)(x)
    x = PReLU(shared_axes=[1,0])(x)
    x = Dropout(dropout_prob)(x)
    
    # Fourth QConv layer
    x = QuaternionConv2D(kern*8, 2, strides=(1,1), padding="same", use_bias=True)(x)
    x = PReLU(shared_axes=[1,0])(x)
    x = Dropout(dropout_prob)(x)
    
    conv_shape = x.get_shape()
    
    """ Modified layers
    # Conv 1D layers (1-3)
    for i in range(n_layers//3):
        x = QuaternionConv1D(kern*2, 2, strides=1, activation="relu", padding="valid", use_bias=True)(x)
        x = PReLU()(x)
        
    # Conv 1D layers (4-6)
    for i in range(n_layers//3):
        x = QuaternionConv1D(kern*4, 2, strides=1, activation="relu", padding="valid", use_bias=True)(x)
        x = PReLU()(x)
       
    # Conv 1D layers (7-9)
    for i in range(n_layers//3):
        x = QuaternionConv1D(kern*8, 2, strides=1, activation="relu", padding="valid", use_bias=True)(x)
        x = PReLU()(x)
    """
        
    # FLatten layer
    # flat   = Flatten()(x)
    
    # Reshape Layer
    x = Reshape(target_shape=[K.int_shape(x)[1], K.int_shape(x)[2] * K.int_shape(x)[3]])(x)
    # Dense 1
    d1 = TimeDistributed( QuaternionDense(256, **denseArgs))(x) #, activation='relu'))(flat)
    if advanced_act == "prelu":
        d1 = PReLU(shared_axes=[1,0])(d1)
    d1 = Dropout(dropout_prob)(d1)
    
    """
    # Dense 2
    d2 = TimeDistributed( QuaternionDense(256, **denseArgs))(d1) #, activation='relu'))(x)
    if advanced_act == "prelu":
        d2 = PReLU(shared_axes=[1,0])(d2)
    d2 = Dropout(dropout_prob)(d2)
    
    # Dense 3
    d3 = TimeDistributed( QuaternionDense(256, **denseArgs))(d2) #, activation='relu'))(x)
    if advanced_act == "prelu":
        d3 = PReLU(shared_axes=[1,0])(d3)
    """
    y_pred = TimeDistributed( Dense(61, activation="softmax"))(d1)
    
    # model = Model(inputs = input_data, outputs = y_pred)
    # model.summary()
    
    
    # the actual loss calc occurs here despite it not being an internal Keras loss function
    def ctc_lambda_func(args):
        y_pred, labels, input_length, label_length = args
        # the 2 is critical here since the first couple outputs of the RNN tend to be garbage:
        #y_pred = y_pred[:, 2:, :]
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
    
    
    labels = Input(shape=[max_str_len], dtype='float32') # max_str_len=61
    input_length = Input(shape=[1], dtype='int64')
    label_length = Input(shape=[1], dtype='int64')
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    
    # clipnorm seems to speeds up convergence
    opt = SGD(learning_rate=0.01, momentum=0.5) # Adam(learning_rate=0.01, beta_1=0.5)
    
    ctc_model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
    
    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    ctc_model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=opt)
    
    return ctc_model, conv_shape # , label_length_input, pred_length_input, y_true_input



""" Create a non-quaternion DNN model (Conv 1D and 2D)
prams
    n_classes: number of classes for classification task
return
    model:CRNN model
    
===============================================================================

def DS2_model(n_classes, conv1d=True):
    if conv1d:
        inputs = Input(shape=(98,40))
    else:
        inputs = Input(shape=(98,40,1))
        
    batch_norm = BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)(inputs)
    
    # First Conv1D layer
    if conv1d:
        conv1 = Conv1D(32,2, padding='valid', activation='relu', strides=1)(batch_norm)
        pool1 = MaxPooling1D(pool_size=3)(conv1)
    else:
        conv1 = Conv2D(32,2, padding='valid', activation='relu')(batch_norm)
        pool1 = MaxPooling2D(pool_size=(3,3))(conv1)
    d1 = Dropout(0.3)(pool1)
    
    # Second Conv1D layer
    if conv1d:
        conv2 = Conv1D(64,2, padding='valid', activation='relu', strides=1)(d1)
        pool2 = MaxPooling1D(pool_size=3)(conv2)
    else:
        conv2 = Conv2D(64,2, padding='valid', activation='relu')(d1)
        pool2 = MaxPooling2D(pool_size=(3,3))(conv2)
    d2 = Dropout(0.3)(pool2)
    
    
    # Third Conv1D layer
    if conv1d:
        conv3 = Conv1D(128,2, padding='valid', activation='relu', strides=1)(d2)
        pool3 = MaxPooling1D(pool_size=3)(conv3)
    else:
        conv3 = Conv2D(128,2, padding='valid', activation='relu')(d2)
        pool3 = MaxPooling2D(pool_size=(3,3))(conv3)
    d3 = Dropout(0.3)(pool3)
    
    # RNN
    if conv1d:
        batch_norm2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)(d3)
        print("\n\n**************\nGRU input shape:\n",batch_norm2.shape)
        print("***************\n\n")
        bgru1 = Bidirectional(GRU(units=128, return_sequences=True), merge_mode='sum')(batch_norm2)
        bgru2 = Bidirectional(GRU(units=128, return_sequences=True), merge_mode='sum')(bgru1)
        bgru3 = Bidirectional(GRU(units=128, return_sequences=False), merge_mode='sum')(bgru2)
        batch_norm3 = BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)(bgru3)
    else:
        batch_norm2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)(d3)
        bgru1 = TimeDistributed(Bidirectional(LSTM(units=128, return_sequences=True), merge_mode='sum'))(batch_norm2)
        bgru2 = TimeDistributed(Bidirectional(GRU(units=128, return_sequences=True), merge_mode='sum'))(bgru1)
        bgru3 = TimeDistributed(Bidirectional(GRU(units=128, return_sequences=False), merge_mode='sum'))(bgru2)
        batch_norm3 = BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)(bgru3)
        
    # Flatten layer
    flat = Flatten()(batch_norm3)
    
    # Dense Layer 1
    dense = Dense(256, activation='relu')(flat)
    # Outputs (Dense layer 2)
    outputs = Dense(n_classes, activation="softmax")(dense)
    
    model = Model(inputs = inputs, outputs = outputs)
    model.summary()
    
    return model
"""