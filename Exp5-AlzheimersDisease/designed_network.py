import tensorflow.keras
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2DTranspose,Input, Reshape, Conv2D, Flatten
from tensorflow.keras.layers import Dense
import tensorflow as tf
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Dropout


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
layer_lrelu=Lambda(lrelu, output_shape=lrelu_output_shape, name='Leaky_ReLU')


def create_model(shape_input_1, shape_input_2, shape_input_3,
                  shape_input_4, shape_input_5):
    
    ######################################################################################
    ################## NEtwork Architecture ##################################################
    ##########################################################################################

    ####################################### MRI FCN ###############################################
    # mri FCN
    MRI_inp_dim = shape_input_1
    MRI_visible = Input(shape=(MRI_inp_dim,), name='MRI')
    hiddenMRI1 = Dense(2*MRI_inp_dim, kernel_initializer='normal', activation='linear')(MRI_visible)
    hiddenMRI2 = hiddenMRI1
    MRI_output = Dense(MRI_inp_dim, kernel_initializer='normal', activation='linear')(hiddenMRI2)

    ####################################### PET FCN ###############################################
    PET_inp_dim = shape_input_2
    PET_visible = Input(shape=(PET_inp_dim,), name='PET')
    hiddenPET1 = Dense(2*PET_inp_dim, kernel_initializer='normal', activation='linear')(PET_visible)
    hiddenPET2=hiddenPET1
    PET_output = Dense(PET_inp_dim, kernel_initializer='normal', activation='linear')(hiddenPET2)

    ####################################### COG FCN ###############################################
    # mri FCN
    COG_inp_dim = shape_input_3
    COG_visible = Input(shape=(COG_inp_dim,), name='COG')
    hiddenCOG1 = Dense(2*COG_inp_dim, kernel_initializer='normal', activation='linear')(COG_visible)
    hiddenCOG2=hiddenCOG1
    COG_output = Dense(COG_inp_dim, kernel_initializer='normal', activation='linear')(hiddenCOG2)

    ####################################### CSF FCN ###############################################
    CSF_inp_dim = shape_input_4
    CSF_visible = Input(shape=(CSF_inp_dim,), name='CSF')
    hiddenCSF1 = Dense(2*CSF_inp_dim, kernel_initializer='normal', activation='linear')(CSF_visible)
    hiddenCSF2=hiddenCSF1
    CSF_output = Dense(CSF_inp_dim, kernel_initializer='normal', activation='linear')(hiddenCSF2)

    ####################################### RF FCN ###############################################
    RF_inp_dim = shape_input_5
    RF_visible = Input(shape=(RF_inp_dim,), name='RF')
    hiddenRF1 = Dense(2*RF_inp_dim, kernel_initializer='normal', activation='linear')(RF_visible)
    hiddenRF2 = hiddenRF1
    RF_output = Dense(RF_inp_dim, kernel_initializer='normal', activation='linear')(hiddenRF2)

    #################################### Concat FCN ###############################################

    merge = concatenate([MRI_output, PET_output, COG_output, CSF_output,RF_output])
    # interpretation layer
    hidden1 = Dense(100, activation='linear')(merge)
    hidden1 = Dropout(0.4)(hidden1)
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


    layer2D_5 = Conv2DTranspose(filters=100, kernel_size=(3,3), strides=(2, 2), kernel_regularizer=tensorflow.keras.regularizers.l2(0.01), activation='linear')(layer2D_4)
    layer2D_5=tensorflow.keras.layers.BatchNormalization()(layer2D_5)
    layer2D_5=layer_lrelu(layer2D_5)


    #layer2D_5_2 = Conv2DTranspose(filters=100, kernel_size=(3,3), strides=(2, 2), kernel_regularizer=tensorflow.keras.regularizers.l2(0.01), activation='linear')(layer2D_5)
    #layer2D_5_2=tensorflow.keras.layers.BatchNormalization()(layer2D_5_2)
    #layer2D_5_2=layer_lrelu(layer2D_5_2)



    layer2D_6 = Conv2DTranspose(filters=3, kernel_size=(3,3), strides=(1, 1), kernel_regularizer=tensorflow.keras.regularizers.l2(0.08), activation='linear' )(layer2D_5)
    layer2D_6=tensorflow.keras.layers.BatchNormalization()(layer2D_6)

    output_1 = tensorflow.keras.layers.Activation('relu')(layer2D_6)#

    # model_tensorization.summary()
    #dot_img_file = 'Network_23x23.png'
    #tf.keras.utils.plot_model(model_tensorization, to_file=dot_img_file, show_shapes=True)



    ##############################################################################
    ################## End NEtwork Architecture ##########################################
    ######################################################################################

    return Model(inputs= [MRI_visible, PET_visible, COG_visible, CSF_visible, RF_visible], outputs=output_1)