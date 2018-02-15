from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.optimizers import RMSprop

from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate

import matplotlib.pyplot as plt 
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


### IOU Loss Function ###
def IOU_loss(y_true, y_pred):
    '''IOU loss function - max to 1'''
    smooth = 1. 
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f)
    
    return 2*(intersection + smooth) / (union + smooth)

def IOU_inverse(y_true, y_pred):
    '''inverse IOU for minimizing '''
    return 1-IOU_loss(y_true, y_pred)


### Unet Building Blocks ### 
def downsample_block(IN, filters, activation, batchnorm, droprate, regularizer): 
    '''downsampling block - downslope + pit of the Unet '''
    CONV = Conv2D(filters, (3, 3), activation = activation, padding = 'same',  kernel_regularizer=regularizer)(IN)
    CONV = BatchNormalization()(CONV) if batchnorm else CONV
    CONV = Dropout(droprate)(CONV) if droprate is not None else CONV
    CONV = Conv2D(filters, (3, 3), activation = activation, padding = 'same', kernel_regularizer=regularizer)(CONV)
    CONV = BatchNormalization()(CONV) if batchnorm else CONV
    OUT = MaxPooling2D((2, 2))(CONV)
    
    return CONV, OUT 


def upsample_block(IN, DOWN, filters, activation, batchnorm, droprate, regularizer): 
    '''upsampling block - upslope of the Unet
    arg DOWN must mirror size of downslope'''
    UP = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same',  kernel_regularizer=regularizer) (IN)
    UP = concatenate([UP, DOWN])
    OUT = Conv2D(filters, (3, 3), activation=activation, padding='same',  kernel_regularizer=regularizer) (UP)
    OUT = BatchNormalization()(OUT) if batchnorm else OUT
    OUT = Dropout(droprate)(OUT) if droprate is not None else OUT
    OUT = Conv2D(filters, (3, 3), activation=activation, padding='same',  kernel_regularizer=regularizer) (OUT)
    OUT = BatchNormalization()(OUT) if batchnorm else OUT
    
    return OUT



def build_model(scale_factor = 3, activation='elu', learn_rate = 0.001, batchnorm=False, droprate = None, regularizer = None, verbose=0):
    '''Build a Unet! 
    https://arxiv.org/abs/1505.04597
    https://github.com/pietz/unet-keras/blob/master/unet.py
    Utilizes custom IOU loss function and defaults: 
        optimizer = 'rmsprop', 
        loss = 'binary_crossentropy'
    
    args : 
        scale_factor (int) : 1/scalefactor is size to reduce input image by 
        activation (str) : activation function for all layers except last (sigmoid)
        learn_rate (float) : learning rate
        batchnorm (bool) : whether to add batchnorm after every convolutional layer 
        droprate (float) : Keras droprate - default None 
        verbose (bool) : prints model summary if True
    
    returns : Keras Functional Model 
    '''
#     
    scale = tuple([int(x/scale_factor) for x in (1200, 1920)])
    print ('input images resized to {}'.format(scale))
    
    K.clear_session()
    inputs = Input((scale[0], scale[1], 3))
#     resize = Lambda(lambda image: K.tf.image.resize_images(image, scale))(inputs)
    norm = Lambda(lambda x: (x/255-0.5))(inputs)

    '''DOWNSAMPLE '''
    c1, p1 = downsample_block(norm, 64, activation, batchnorm, droprate, regularizer)
    c2, p2 = downsample_block(p1, 128, activation, batchnorm, droprate, regularizer)
    c3, p3 = downsample_block(p2, 256, activation, batchnorm, droprate, regularizer)
    c4, p4 = downsample_block(p3, 512, activation, batchnorm, droprate, regularizer)
#     c5, p5 = downsample_block(p4, 1024, activation, batchnorm, droprate)
    
    ''' UPSAMPLE '''
#     c6 = upsample_block(c5, c4, 128, activation, batchnorm, droprate)
    c5 = upsample_block(c4, c3, 256, activation, batchnorm, droprate, regularizer)
    c6 = upsample_block(c3, c2, 128, activation, batchnorm, droprate, regularizer)
    c7 = upsample_block(c2, c1, 64, activation, batchnorm, droprate, regularizer)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c7)
#     outputs = Lambda(lambda image: K.tf.image.resize_images(image, (1200, 1920)))(outputs)

    model = Model (inputs=[inputs], outputs = [outputs])
    model.compile(optimizer = RMSprop(lr=learn_rate), 
                  loss = IOU_inverse,
                 metrics = [IOU_loss])
    
    if verbose: 
        model.summary()

    print ('model compiled')
    return model 


def plot_history(model_hist, metric): 

    history_dict = model_hist.history
    loss_values = history_dict['{}'.format(metric)]
    val_loss_values = history_dict['val_{}'.format(metric)]
    epochs = range(1, len(model_hist.epoch) + 1)

    plt.figure(figsize=(8, 3))
    plt.plot(epochs, loss_values, label='Training {}'.format(metric))
    plt.plot(epochs, val_loss_values, label='Validation {}'.format(metric))
    plt.title('{} Training and Validation {}'.format(model_hist.model.name, smetric))
    plt.xlabel('epochs')
    plt.ylabel('{}'.format(metric))
    plt.legend()
    
    save_path = os.path.join('img', '{}_{}.png'.format(model_hist.model.name, metric))
    plt.savefig(save_path)
    # plt.show()

