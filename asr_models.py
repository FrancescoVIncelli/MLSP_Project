# -*- coding: utf-8 -*-
""" @author: vince
"""

# ===============
# Import packages

import complexnn
from   complexnn.conv import QuaternionConv2D, QuaternionConv1D # , QuaternionBatchNormalization #, QuaternionLSTM
from   complexnn.dense import QuaternionDense
from   complexnn.bn import QuaternionBatchNormalization

import numpy                                as     np
import os, pdb, socket, sys, time           as     T

import tensorflow                           as     tf

from   tensorflow.keras.callbacks           import Callback, ModelCheckpoint, LearningRateScheduler

from   tensorflow.keras.initializers        import Orthogonal
from   tensorflow.keras.layers              import Dropout, AveragePooling1D, AveragePooling2D, MaxPooling1D, MaxPooling2D, ZeroPadding1D, ZeroPadding2D, \
                                                    Input, Flatten, Dense, Conv1D, Conv2D, LSTM, GRU, ConvLSTM2D, Lambda, Reshape, Permute, \
                                                    BatchNormalization, Bidirectional, TimeDistributed, SpatialDropout1D, PReLU
                                                    
from   tensorflow.keras.models              import Model, load_model, save_model, Sequential
from   tensorflow.keras.optimizers          import SGD, Adam, RMSprop
from   tensorflow.keras.initializers        import GlorotNormal, GlorotUniform, Orthogonal, RandomNormal
from   tensorflow.keras.regularizers        import l2
from   tensorflow.keras.utils               import to_categorical
import tensorflow.keras.backend             as     K
import tensorflow.keras.models              as     KM

from tensorflow.keras.layers import add, concatenate, Activation

# from   speechvalley.utils                   import calcPER  as PER
# from   keras_ctcmodel.CTCModel              import CTCModel as CTCModel



# the actual loss calc occurs here despite it not being
# an internal Keras loss function

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    # y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


# For a real OCR application, this should be beam search with a dictionary
# and language model.  For this example, best path is sufficient.

def decode_batch(test_func, word_batch):
    out = test_func([word_batch])[0]
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = labels_to_text(out_best)
        ret.append(outstr)
    return ret



# =====================
# Neural Network Models


# =================== #
# Simple CNN-1D Model # (OK)

def simple_cnn_1D(args):
    #
    # Model parameters
    #
    model_type   = args.model_type
    input_shape  = args.input_shape
    
    n_convs      = args.n_layers_convs
    n_dense      = args.n_layers_dense
    
    max_str_len  = args.max_str_len
    num_classes  = args.num_classes
    
    activation   = args.act
    advanced_act = args.aact
    
    padding      = args.padding
    start_filter = args.sf_dim
    kernel_size  = args.kernel_size
    
    kernel_init  = args.kernel_init
    l2_reg       = args.l2_reg
    shared_axes  = args.shared_axes
    opt          = args.opt
    
    #
    # Layer parameters
    #
    
    # Conv layers parameters
    convArgs      = {
            "activation":               activation,
            "data_format":              "channels_last",
            "padding":                  padding,
            "bias_initializer":         "zeros",
            "kernel_regularizer":       l2(l2_reg),
            "kernel_initializer":       kernel_init,
            "use_bias":                 True
            }
    # Dense layers parameters
    denseArgs     = {
            "activation":               activation,        
            "kernel_regularizer":       l2(l2_reg),
            "kernel_initializer":       kernel_init, # "glorot_normal"
            "bias_initializer":         "zeros",
            "use_bias":                 True
            }
    
    #
    # Model building
    #
    
    # Input Stage
    if model_type == "quaternion":
        quat_init = 'quaternion'
        convArgs.update({"kernel_initializer": quat_init})
        
    input_data = Input(name='the_input', shape=input_shape, dtype='float64')
    
    # Convolutional Stage
    if model_type=='real':
        # batch-normalization
        O = BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)(input_data)
    
        for idx in range(n_convs):
            conv_filters=start_filter*(2**idx)
            O = Conv1D(conv_filters, kernel_size, name='conv_{}'.format(idx), **convArgs)(O)
            if advanced_act=='prelu':
                O = PReLU()(O)
            
        # batch-normalization
        O = BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)(O)
    else:
        # batch-normalization
        O = QuaternionBatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)(input_data)
        
        for idx in range(n_convs):
            conv_filters=start_filter*(2**idx)
            O = QuaternionConv1D(conv_filters, kernel_size, name='conv_{}'.format(idx), **convArgs)(O)
            if advanced_act=='prelu':
                O = PReLU()(O)
        
        # batch-normalization
        O = QuaternionBatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)(O)
    # conv_shape = O.shape
    
    # reshape
    # O = Reshape(target_shape=[K.int_shape(O)[1] * K.int_shape(O)[2]])(O)
    
    # Fully-Connected Stage
    if model_type=='real':
        for idx in range(n_dense):
            O = TimeDistributed(Dense(1024, **denseArgs, name='dense_{}'.format(idx)))(O)
            if advanced_act=='prelu':
                O = PReLU()(O)
        
        # batch-normalization
        O = BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)(O)
    else:
        for idx in range(n_dense):
            O = TimeDistributed(QuaternionDense(256, **denseArgs, name='dense_{}'.format(idx)))(O)
            if advanced_act=='prelu':
                O = PReLU()(O)
        
        # batch-normalization
        O = QuaternionBatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)(O)
        
    #
    # Prediction (Output) Stage
    #
    inner = Dense(num_classes, kernel_regularizer=l2(l2_reg), use_bias=True, bias_initializer="zeros", kernel_initializer=kernel_init)(O)
    y_pred = Activation('softmax', name='softmax')(inner)
    
    # Print summary of neural model
    Model(inputs=input_data, outputs=y_pred).summary()
    
    labels = Input(name='the_labels', shape=[max_str_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    
    # clipnorm seems to speeds up convergence
    # sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    opt_dict = {
        'adam': Adam(lr=0.02, clipnorm=5),
        'sgd': SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    }
    optimizer = opt_dict[opt]
    
    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
    
    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)
    
    return model




# =================== #
# Simple CNN 2D Model # (OK)

def simple_cnn_2D(args):
    #
    # Model parameters
    #
    model_type   = args.model_type
    input_shape  = args.input_shape
    
    n_convs      = args.n_layers_convs
    n_dense      = args.n_layers_dense
    
    max_str_len  = args.max_str_len
    num_classes  = args.num_classes
    
    activation   = args.act
    advanced_act = args.aact
    
    padding      = args.padding
    start_filter = args.sf_dim
    kernel_size  = args.kernel_size
    
    kernel_init  = args.kernel_init
    l2_reg       = args.l2_reg
    shared_axes  = args.shared_axes
    opt          = args.opt
    
    #
    # Layer parameters
    #
    
    # Conv layers parameters
    convArgs      = {
            "activation":               activation,
            "data_format":              "channels_last",
            "padding":                  padding,
            "bias_initializer":         "zeros",
            "kernel_regularizer":       l2(l2_reg),
            "kernel_initializer":       kernel_init,
            "use_bias":                 True
            }
    # Dense layers parameters
    denseArgs     = {
            "activation":               activation,        
            "kernel_regularizer":       l2(l2_reg),
            "kernel_initializer":       kernel_init, # "glorot_normal"
            "bias_initializer":         "zeros",
            "use_bias":                 True
            }
    
    #
    # Model building
    #
    
    # Input Stage
    if model_type == "quaternion":
        quat_init = 'quaternion'
        convArgs.update({"kernel_initializer": quat_init})
        
    input_data = Input(name='the_input', shape=input_shape, dtype='float64')
    
    # Convolutional Stage
    if model_type=='real':
        # batch-normalization
        O = BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)(input_data)
    
        for idx in range(n_convs):
            conv_filters=start_filter*(2**idx)
            O = Conv2D(conv_filters, kernel_size, name='conv_{}'.format(idx), **convArgs)(O)
            if advanced_act=='relu':
                O = PReLU()(O)
            
        # batch-normalization
        O = BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)(O)
    else:
        # batch-normalization
        O = QuaternionBatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)(input_data)
        
        for idx in range(n_convs):
            conv_filters=start_filter*(2**idx)
            O = QuaternionConv2D(conv_filters, kernel_size, name='conv_{}'.format(idx), **convArgs)(O)
            if advanced_act=='relu':
                O = PReLU()(O)
        
        # batch-normalization
        O = QuaternionBatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)(O)
    # conv_shape = O.shape
    
    #
    # Reshaping Stage
    #
    O = Reshape(target_shape=[K.int_shape(O)[1], K.int_shape(O)[2] * K.int_shape(O)[3]])(O)
    
    # Fully-Connected Stage
    if model_type=='real':
        for idx in range(n_dense):
            O = TimeDistributed(Dense(1024, **denseArgs, name='dense_{}'.format(idx)))(O)
            if advanced_act=='prelu':
                O = PReLU()(O)
        
        # batch-normalization
        O = BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)(O)
    else:
        for idx in range(n_dense):
            O = TimeDistributed(QuaternionDense(256, **denseArgs, name='dense_{}'.format(idx)))(O)
            if advanced_act=='prelu':
                O = PReLU()(O)
        
        # batch-normalization
        O = QuaternionBatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)(O)
    
    #
    # Prediction (Output) Stage
    #
    inner = Dense(num_classes, kernel_regularizer=l2(l2_reg), use_bias=True, bias_initializer="zeros", kernel_initializer=kernel_init)(O)
    y_pred = Activation('softmax', name='softmax')(inner)
    
    # Print summary of neural model
    Model(inputs=input_data, outputs=y_pred).summary()
    
    labels = Input(name='the_labels', shape=[max_str_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    
    # clipnorm seems to speeds up convergence
    opt_dict = {
        'adam': Adam(lr=0.02, clipnorm=5),
        'sgd': SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    }
    # sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    # adam = Adam(lr=0.02, clipnorm=5)
    
    optimizer = opt_dict[opt]
    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
    
    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)
    
    return model




# ============================== #
# Simple Sequential CNN 2D Model # (OK)

def sequential_cnn_1D(args):
    #
    # Model parameters
    #
    model_type   = args.model_type
    # input_shape  = args.input_shape
    
    n_convs      = args.n_layers_convs
    n_dense      = args.n_layers_dense
    
    # max_str_len  = args.max_str_len
    num_classes  = args.num_classes
    
    activation   = args.act
    advanced_act = args.aact
    
    padding      = args.padding
    start_filter = args.sf_dim
    kernel_size  = args.kernel_size
    
    kernel_init  = args.kernel_init
    l2_reg       = args.l2_reg
    shared_axes  = args.shared_axes
    opt          = args.opt
    
    #
    # Layer parameters
    #
    
    # Conv layers parameters
    convArgs      = {
            "activation":               activation,
            "data_format":              "channels_last",
            "padding":                  padding,
            "bias_initializer":         "zeros",
            "kernel_regularizer":       l2(l2_reg),
            "kernel_initializer":       kernel_init,
            "use_bias":                 True
            }
    # Dense layers parameters
    denseArgs     = {
            "activation":               activation,        
            "kernel_regularizer":       l2(l2_reg),
            "kernel_initializer":       kernel_init, # "glorot_normal"
            "bias_initializer":         "zeros",
            "use_bias":                 True
            }
    
    #
    # Model building
    #
    
    if model_type == "quaternion":
        quat_init = 'quaternion'
        convArgs.update({"kernel_initializer": quat_init})
        
    # input_data = Input(name='the_input', shape=input_shape, dtype='float64')
    
    model = Sequential()
    
    # Convolutional Stage
    if model_type=='real':
        # batch-normalization
        # O = BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)
        
        for idx in range(n_convs):
            conv_filters=start_filter*(2**idx)
            O = Conv1D(conv_filters, kernel_size, name='conv_{}'.format(idx), **convArgs)
            model.add(O)
            if advanced_act=='prelu':
                O = PReLU()
                model.add(O)
            
        # batch-normalization
        O = BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)
        model.add(O)
    else:
        # batch-normalization
        # O = QuaternionBatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)
        
        for idx in range(n_convs):
            conv_filters=start_filter*(2**idx)
            O = QuaternionConv1D(conv_filters, kernel_size, name='conv_{}'.format(idx), **convArgs)(O)
            if advanced_act=='prelu':
                O = PReLU()
                model.add(O)
        
        # batch-normalization
        O = QuaternionBatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)(O)
    # conv_shape = O.shape
    
    # Fully-Connected Stage
    if model_type=='real':
        for idx in range(n_dense):
            O = TimeDistributed(Dense(1024, **denseArgs))
            model.add(O)
            if advanced_act=='prelu':
                O = PReLU()
                model.add(O)
        
        # batch-normalization
        O = BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)
        model.add(O)
    else:
        for idx in range(n_dense):
            O = TimeDistributed(QuaternionDense(1024, **denseArgs))
            model.add(O)
            if advanced_act=='prelu':
                O = PReLU()
                model.add(O)
        
        # batch-normalization
        O = QuaternionBatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)(O)
        
    #
    # Prediction (Output) Stage
    #
    y_pred = Dense(num_classes, activation='softmax', kernel_regularizer=l2(l2_reg), use_bias=True, bias_initializer="zeros", kernel_initializer=kernel_init)
    model.add(y_pred)
    
    # clipnorm seems to speeds up convergence
    opt_dict = {
        'sgd': SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5),
        'adam': Adam(lr=0.02, clipnorm=5)
    }
    optimizer = opt_dict[opt]
    
    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    # model.summary()
    
    return model


def sequential_cnn_2D(args):
    #
    # Model parameters
    #
    model_type   = args.model_type
    # input_shape  = args.input_shape
    
    n_convs      = args.n_layers_convs
    n_dense      = args.n_layers_dense
    
    # max_str_len  = args.max_str_len
    num_classes  = args.num_classes
    
    activation   = args.act
    advanced_act = args.aact
    
    padding      = args.padding
    start_filter = args.sf_dim
    kernel_size  = args.kernel_size
    
    kernel_init  = args.kernel_init
    l2_reg       = args.l2_reg
    shared_axes  = args.shared_axes
    opt          = args.opt
    
    #
    # Layer parameters
    #
    
    # Conv layers parameters
    convArgs      = {
            "activation":               activation,
            "data_format":              "channels_last",
            "padding":                  padding,
            "bias_initializer":         "zeros",
            "kernel_regularizer":       l2(l2_reg),
            "kernel_initializer":       kernel_init,
            "use_bias":                 True
            }
    # Dense layers parameters
    denseArgs     = {
            "activation":               activation,        
            "kernel_regularizer":       l2(l2_reg),
            "kernel_initializer":       kernel_init, # "glorot_normal"
            "bias_initializer":         "zeros",
            "use_bias":                 True
            }
    
    #
    # Model building
    #
    
    if model_type == "quaternion":
        quat_init = 'quaternion'
        convArgs.update({"kernel_initializer": quat_init})
        
    # input_data = Input(name='the_input', shape=input_shape, dtype='float64')
    
    model = Sequential()
    
    # Convolutional Stage
    if model_type=='real':
        # batch-normalization
        # O = BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)
        
        for idx in range(n_convs):
            conv_filters=start_filter*(2**idx)
            O = Conv2D(conv_filters, kernel_size, name='conv_{}'.format(idx), **convArgs)
            model.add(O)
            if advanced_act=='prelu':
                O = PReLU()
                model.add(O)
            
        # batch-normalization
        O = BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)
        model.add(O)
    else:
        # batch-normalization
        # O = QuaternionBatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)
        
        for idx in range(n_convs):
            conv_filters=start_filter*(2**idx)
            O = QuaternionConv2D(conv_filters, kernel_size, name='conv_{}'.format(idx), **convArgs)(O)
            if advanced_act=='prelu':
                O = PReLU()
                model.add(O)
        
        # batch-normalization
        O = QuaternionBatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)(O)
    # conv_shape = O.shape
    
    #
    # Reshaping Stage
    #
    O = Reshape(target_shape=[K.int_shape(O)[1], K.int_shape(O)[2] * K.int_shape(O)[3]])(O)
    
    # Fully-Connected Stage
    if model_type=='real':
        for idx in range(n_dense):
            O = TimeDistributed(Dense(1024, **denseArgs))
            model.add(O)
            if advanced_act=='prelu':
                O = PReLU()
                model.add(O)
        
        # batch-normalization
        O = BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)
        model.add(O)
    else:
        for idx in range(n_dense):
            O = TimeDistributed(QuaternionDense(1024, **denseArgs))
            model.add(O)
            if advanced_act=='prelu':
                O = PReLU()
                model.add(O)
        
        # batch-normalization
        O = QuaternionBatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)(O)
        
    #
    # Prediction (Output) Stage
    #
    y_pred = Dense(num_classes, activation='softmax', kernel_regularizer=l2(l2_reg), use_bias=True, bias_initializer="zeros", kernel_initializer=kernel_init)
    model.add(y_pred)
    
    # clipnorm seems to speeds up convergence
    opt_dict = {
        'sgd': SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5),
        'adam': Adam(lr=0.02, clipnorm=5)
    }
    optimizer = opt_dict[opt]
    
    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    
    return model
