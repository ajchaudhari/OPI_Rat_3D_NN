import tensorflow as tf

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv3D, Conv3DTranspose
from keras.layers.pooling import MaxPooling3D, GlobalMaxPool3D
from keras.layers.merge import concatenate, add
from keras.layers import Input, MaxPooling3D, UpSampling3D, Conv3D
from keras.layers.normalization import BatchNormalization

# Build U-Net model
dropout = 0.2
filterSize = 3
actF = tf.keras.layers.LeakyReLU(alpha=0.1) # 'relu' 
hn = tf.keras.initializers.RandomUniform() #'random_uniform' # 'glorot_uniform' #

def unet(input_size, numClasses):
    
    inputs = Input(input_size)


    conv1 = Conv3D(64, filterSize, activation = actF, padding = 'same', kernel_initializer = hn)(inputs)
    conv1 = Conv3D(64, filterSize, activation = actF, padding = 'same', kernel_initializer = hn)(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(128, filterSize, activation = actF, padding = 'same', kernel_initializer = hn)(pool1)
    conv2 = Conv3D(128, filterSize, activation = actF, padding = 'same', kernel_initializer = hn)(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(256, filterSize, activation = actF, padding = 'same', kernel_initializer = hn)(pool2)
    conv3 = Conv3D(256, filterSize, activation = actF, padding = 'same', kernel_initializer = hn)(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = Conv3D(512, filterSize, activation = actF, padding = 'same', kernel_initializer = hn)(pool3)
    conv4 = Conv3D(512, filterSize, activation = actF, padding = 'same', kernel_initializer = hn)(conv4)
    drop4 = Dropout(dropout)(conv4)

    up5 = Conv3D(512, 2, activation = actF, padding = 'same', kernel_initializer = hn)(UpSampling3D(size = (2,2,2))(drop4))
    merge5 = concatenate([conv3,up5], axis = 4)
    conv5 = Conv3D(256, filterSize, activation = actF, padding = 'same', kernel_initializer = hn)(merge5)
    conv5 = Conv3D(256, filterSize, activation = actF, padding = 'same', kernel_initializer = hn)(conv5)

    up6 = Conv3D(256, 2, activation = actF, padding = 'same', kernel_initializer = hn)(UpSampling3D(size = (2,2,2))(conv5))
    merge6 = concatenate([conv2,up6], axis = 4)
    conv6 = Conv3D(128, filterSize, activation = actF, padding = 'same', kernel_initializer = hn)(merge6)
    conv6 = Conv3D(128, filterSize, activation = actF, padding = 'same', kernel_initializer = hn)(conv6)

    up7 = Conv3D(128, 2, activation = actF, padding = 'same', kernel_initializer = hn)(UpSampling3D(size = (2,2,2))(conv6))
    merge7 = concatenate([conv1,up7], axis = 4)
    conv7 = Conv3D(64, filterSize, activation = actF, padding = 'same', kernel_initializer = hn)(merge7)
    conv7 = Conv3D(64, filterSize, activation = actF, padding = 'same', kernel_initializer = hn)(conv7)

    
    # Softmax
    conv8 = Conv3D(numClasses, (1,1,1), activation = 'softmax')(conv7)

    model = Model(inputs = inputs, outputs = conv8)

    return model 
