from tensorflow.keras.layers import Dropout
import tensorflow as tf
import tensorflow
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2DTranspose,Input, Reshape, Conv2D, Flatten
from tensorflow.keras.layers import Dense,concatenate


def lrelu(x): #from pix2pix code
    a=0.2
    # adding these together creates the leak part and linear part
    # then cancels them out by subtracting/adding an absolute value term
    # leak: a*x/2 - a*abs(x)/2
    # linear: x/2 + abs(x)/2

    # this block looks like it has 2 inputs on the graph unless we do this
    x = tf.identity(x)
    return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

def lrelu_output_shape(input_shape):
    shape = list(input_shape)
    return tuple(shape)
from tensorflow.keras.layers import Lambda
layer_lrelu=Lambda(lrelu, output_shape=lrelu_output_shape)



######################################################################################
################## NEtwork Architecture ##################################################
##########################################################################################

def create_model(shape_input):
    In_shape=shape_input
    In = Input(shape=(In_shape,))
    fc1 = Dense(2*In_shape, kernel_initializer='normal', activation='linear')(In)
    fc2 = Dense(In_shape, kernel_initializer='normal', activation='linear')(fc1)

    # interpretation layer
    fc3 = Dense(100, activation='linear')(fc2)
    hidden1 = Dropout(0.3)(fc3)
    hidden1_reshape = Reshape((10, 10, 1))(hidden1)
    e_2=tensorflow.keras.layers.BatchNormalization()(hidden1_reshape)
    e_2=layer_lrelu(e_2)

    layer2D_1 = Conv2DTranspose(filters=10, kernel_size=(3,3), strides=(1, 1),padding="same", activation='linear')(e_2)
    layer2D_1=tensorflow.keras.layers.BatchNormalization()(layer2D_1)
    layer2D_1=layer_lrelu(layer2D_1)

    layer2D_2 = Conv2DTranspose(filters=10, kernel_size=(3,3), strides=(1, 1), dilation_rate=(2,2),padding="same", activation='linear')(e_2)
    layer2D_2=tensorflow.keras.layers.BatchNormalization()(layer2D_2)
    layer2D_2=layer_lrelu(layer2D_2)


    layer2D_3 = Conv2DTranspose(filters=10, kernel_size=(3,3), strides=(1, 1), dilation_rate=(3,3), padding="same", activation='linear')(e_2)
    layer2D_3=tensorflow.keras.layers.BatchNormalization()(layer2D_3)
    layer2D_3=layer_lrelu(layer2D_3)

    ############################################################################
    layer2D_4 = concatenate([layer2D_1,layer2D_2,layer2D_3])
    layer2D_4=tensorflow.keras.layers.BatchNormalization()(layer2D_4)
    ############################################################################

    layer2D_5 = Conv2DTranspose(filters=100, kernel_size=(3,3), strides=(2, 2), kernel_regularizer=tensorflow.keras.regularizers.l2(0.01), activation='linear')(layer2D_4)
    layer2D_5=tensorflow.keras.layers.BatchNormalization()(layer2D_5)
    layer2D_5=layer_lrelu(layer2D_5)

    layer2D_6 = Conv2DTranspose(filters=100, kernel_size=(3,3), strides=(2, 2), kernel_regularizer=tensorflow.keras.regularizers.l2(0.01), activation='linear')(layer2D_5)
    layer2D_6=tensorflow.keras.layers.BatchNormalization()(layer2D_6)
    layer2D_6=layer_lrelu(layer2D_6)

    layer2D_7 = Conv2DTranspose(filters=3, kernel_size=(3,3), strides=(1, 1), kernel_regularizer=tensorflow.keras.regularizers.l2(0.08), activation='linear' )(layer2D_6)
    layer2D_7=tensorflow.keras.layers.BatchNormalization()(layer2D_7)

    output_1 = tensorflow.keras.layers.Activation('relu')(layer2D_7)#
    return Model(inputs= In, outputs=output_1)  

    # model_tensorization.summary()
