# -*- coding: utf-8 -*-

import tensorflow as tf
import complexnn
from   complexnn                         import QuaternionConv1D, QuaternionConv2D, QuaternionDense

from   tensorflow.keras.layers           import Layer, Dropout, Input, Flatten, Dense, Convolution2D, \
    BatchNormalization, Conv2D, Lambda, Permute, TimeDistributed, PReLU
                
from   tensorflow.keras.models           import Model, load_model, save_model
from   tensorflow.keras.optimizers       import SGD, Adam, RMSprop

from   tensorflow.keras.regularizers     import l2
from   tensorflow.keras.utils            import to_categorical
import tensorflow.keras.backend          as     K

#from   keras.callbacks                       import Callback, ModelCheckpoint, LearningRateScheduler
#from   keras.initializers                    import Orthogonal

import numpy as np

    
class CNN(object):
    def __init__(self, quaternion, batch_size, lr, epochs):
        self.quaternion = quaternion
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        
    def _build_model(self):
        n_conv_layers = 6
        fmap_size   = 8
        drop_prob     = 0.3
        inputShape    = (3,41,None)  
        filsize       = (3, 5)
        Axis   = 1
        
        # Convolutional layers hyperparameters
        convArgs      = {
                "activation":               "relu",
                "data_format":              "channels_first",
                "padding":                  "same",
                "bias_initializer":         "zeros",
                "kernel_regularizer":       l2(1e-5)
                #"kernel_initializer":       "random_uniform",
                }
        # Dense layers hyperparameters
        denseArgs     = {
                "activation":               "relu",        
                "kernel_regularizer":       l2(1e-5),
                "kernel_initializer":       "random_uniform",
                "bias_initializer":         "zeros",
                "use_bias":                 True
                }
         
    
        #
        # Input Layer & CTC Parameters for TIMIT
        #
        if self.quaternion:
            inputs = Input(shape=(None,41,4))
        else:
            inputs = Input(shape=(None,41,1))
    
        batch_size=32
        y_true = Input(name='y_true', shape=(None,61), dtype='float32')
        y_pred = Input(name='y_pred', shape=(None,61), dtype='float32')
        input_length = Input(name='input_length', shape=(batch_size,1), dtype='int64')
        label_length = Input(name='label_length', shape=(batch_size,1), dtype='int64')
        ctc_args = [y_true, y_pred, input_length, label_length]
        
        # ==================== #
        # Convolutional Layers #
        
        # ===== Input step ===== #
        # Input (Q)Conv-2D Layer #
        
        if self.quaternion:
            conv = QuaternionConv2D(fmap_size, filsize, name='conv', use_bias=True, **convArgs)(inputs)
            conv = PReLU(shared_axes=[1,0])(conv)
        else:
            conv = Conv2D(fmap_size, filsize, name='conv', use_bias=True, **convArgs)(inputs)
            conv = PReLU(shared_axes=[1,0])(conv)
        # Pooling
        conv = MaxPooling2D(pool_size=(1, 3), padding='same')(conv)
        
        quaternion=True
        
        # ========= Convolution stage ============= #
        # 1st-2nd-3rd-4th-5th-6th (Q)Conv-2D Layers #
        for i in range(0,n_conv_layers//3):
            if quaternion: #self.quaternion:
                conv = QuaternionConv2D(fmap_size*2, filsize, name='conv'+str(i), use_bias=True, **convArgs)(conv)
                conv = PReLU(shared_axes=[1,0])(conv)
                conv = Dropout(drop_prob)(conv)
            else:
                conv = Conv2D(fmap_size*2, filsize, name='conv'+str(i), use_bias=True,**convArgs)(conv)
                conv = PReLU(shared_axes=[1,0])(conv)
                conv = Dropout(drop_prob)(conv)
        
        for i in range(0,n_conv_layers//3):
            if quaternion:
                conv = QuaternionConv2D(fmap_size*4, filsize, name='conv'+str(i), use_bias=True, **convArgs)(conv)
                conv = PReLU(shared_axes=[1,0])(conv)
                conv = Dropout(drop_prob)(conv)
            else:
                conv = Conv2D(fmap_size*4, filsize, name='conv'+str(i), use_bias=True,**convArgs)(conv)
                conv = PReLU(shared_axes=[1,0])(conv)
                conv = Dropout(drop_prob)(conv)
                
        for i in range(0,n_conv_layers//3):
            if quaternion:
                conv = QuaternionConv2D(fmap_size*8, filsize, name='conv'+str(i), use_bias=True, **convArgs)(conv)
                conv = PReLU(shared_axes=[1,0])(conv)
                conv = Dropout(drop_prob)(conv)
            else:
                conv = Conv2D(fmap_size*8, filsize, name='conv'+str(i), use_bias=True,**convArgs)(conv)
                conv = PReLU(shared_axes=[1,0])(conv)
                conv = Dropout(drop_prob)(conv)
        
        
        # =================== #
        # Permutation for CTC #
            
        # [TO MODIFY]
        # permute = Permute((3,1,2))(conv)
        # lam = Lambda(lambda x: K.reshape(x, (K.shape(x)[0], K.shape(x)[1], K.shape(x)[2] * K.shape(x)[3])),
        #              output_shape=lambda x: (None, None, K.shape(x)[1] * K.shape(x)[3]))(permute)
        
        lam = Lambda(lambda x: tf.reshape(x,  (tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2] * tf.shape(x)[3])))(conv)

        # ============= #
        # Dense layers  #
        
        if quaternion: #self.quaternion:
            dense = TimeDistributed( QuaternionDense(256,  **denseArgs))(conv) #(lam)
            dense = PReLU(shared_axes=[1,0])(dense)
            dense = Dropout(drop_prob)(dense)
            
            dense2 = TimeDistributed( QuaternionDense(256,  **denseArgs))(dense)
            dense2= PReLU(shared_axes=[1,0])(dense2)
            dense2 = Dropout(drop_prob)(dense2)
            
            dense3 = TimeDistributed( QuaternionDense(256,  **denseArgs))(dense2)
            dense3= PReLU(shared_axes=[1,0])(dense3)
        else:
            dense = TimeDistributed( Dense(1024,  **denseArgs))(conv) #(lam)
            dense = PReLU(shared_axes=[1,0])(dense)
            dense = Dropout(drop_prob)(dense)
            
            dense2 = TimeDistributed( Dense(1024, **denseArgs))(dense)
            dense2 = PReLU(shared_axes=[1,0])(dense2)
            dense2 = Dropout(drop_prob)(dense2)
            
            dense3 = TimeDistributed( Dense(1024, **denseArgs))(dense2)
            dense3 = PReLU(shared_axes=[1,0])(dense3)
    
        # ===================== #
        # Fully connected layer #
        outputs = TimeDistributed( Dense(62,  activation='softmax', kernel_regularizer=l2(1e-5), use_bias=True,
                                         bias_initializer="zeros", kernel_initializer='random_uniform' ))(dense3)
          
        # [TODO] CTC For sequence labelling
        # ctc = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([ctc_args])
        
        # Return the model
        # val_function = K.function([inputs],[pred])
        return Model(inputs=[I, input_length, labels, label_length], outputs=O), val_function
    
        
    # [TO MODIFY]
    def ctc_lambda_func(args):
        y_true, y_pred, input_length, label_length = args
        return K.ctc_batch_cost(y_true, y_pred, input_length, label_length)