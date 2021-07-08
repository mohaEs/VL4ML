# -*- coding: utf-8 -*-
"""
Created on Wed May  1 10:28:19 2019

@author: User
"""

# Multilayer Perceptron
import pandas
import numpy
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


import tensorflow.keras
import math
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2DTranspose,Input, Reshape, Conv2D, Flatten
from tensorflow.keras.layers import Dense,concatenate
from sklearn.metrics import mean_squared_error
import argparse
#from tensorflow.keras.utils.np_utils import to_categorical
import matplotlib
matplotlib.use("Agg")
import tensorflow as tf
import matplotlib.pyplot as plt


from tensorflow.keras.layers import Dropout
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr, spearmanr

####################
####### Functions #############
####################


def custom_loss_2 (y_true, y_pred):
    A = tensorflow.keras.losses.mean_absolute_error(y_true, y_pred)
    return A





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


def FnCreateTargetImages(Labels):
    # Neon color
    OutputImages=np.zeros(shape=(len(Labels),23,23,3))    
    
    for i in range(len(Labels)):
          
        if Labels[i]==0:
            OutputImages[i,6:17,6:17,0]=0
            OutputImages[i,6:17,6:17,1]=1
            OutputImages[i,6:17,6:17,2]=0
            
            OutputImages[i,3:20,3,:]=1
            OutputImages[i,3:20,20,:]=1
            OutputImages[i,20,3:20,:]=1
            OutputImages[i,3,3:20,:]=1
            
            
        elif Labels[i]==1:
            OutputImages[i,6:17,6:17,0]=0.5
            OutputImages[i,6:17,6:17,1]=0.8
            OutputImages[i,6:17,6:17,2]=0
            
            OutputImages[i,3:20,3,:]=0.8
            OutputImages[i,3:20,20,:]=0.8
            OutputImages[i,20,3:20,:]=0.8
            OutputImages[i,3,3:20,:]=0.8
            
        elif Labels[i]==2:
            OutputImages[i,6:17,6:17,0]=1
            OutputImages[i,6:17,6:17,1]=1
            OutputImages[i,6:17,6:17,2]=0
            
            OutputImages[i,3:20,3,:]=0.6
            OutputImages[i,3:20,20,:]=0.6
            OutputImages[i,20,3:20,:]=0.6
            OutputImages[i,3,3:20,:]=0.6
            
        elif Labels[i]==3:
            OutputImages[i,6:17,6:17,0]=1
            OutputImages[i,6:17,6:17,1]=0.5
            OutputImages[i,6:17,6:17,2]=0
            
            OutputImages[i,3:20,3,:]=0.4
            OutputImages[i,3:20,20,:]=0.4
            OutputImages[i,20,3:20,:]=0.4
            OutputImages[i,3,3:20,:]=0.4
            
        elif Labels[i]==4:
            OutputImages[i,6:17,6:17,0]=1
            OutputImages[i,6:17,6:17,1]=0
            OutputImages[i,6:17,6:17,2]=0
            
            OutputImages[i,3:20,3,:]=0.2
            OutputImages[i,3:20,20,:]=0.2
            OutputImages[i,20,3:20,:]=0.2
            OutputImages[i,3,3:20,:]=0.2
                    
            # plt.figure()
            # plt.imshow(OutputImages[19,:,:,:])
    return OutputImages



def FnCreateValidLabes(Labels):
    return range(len(Labels))

     

####################
###### End of functions ##############    
####################    


####################
###### Reading input arguments ##############        
#########################################################
######### Hyper paramters configurations ##################
########### ##########################################################


n_splits=10

parser = argparse.ArgumentParser()
parser.add_argument("--output", )
parser.add_argument("--max_epochs",  )
parser.add_argument("--BatchSize", )
parser.add_argument("--k", )
parser.add_argument("--m", )
a = parser.parse_args()


a.max_epochs=300
a.BatchSize=100
a.output='./1/'

import os
try:
    os.stat(a.output)
except:
    os.mkdir(a.output) 



####################
###### Reading Data ##############    
####################    

AllDataset = pandas.read_csv('./data/train.csv', low_memory=False)



# creating independent features X and dependant feature Y
y = AllDataset['diagnosis']


from skimage import io


X_all=np.zeros((len(AllDataset),256,256,3),dtype=np.uint8)
y_all=np.zeros((len(AllDataset),1), dtype=int)

for i in range(len(AllDataset)):
    FileName=AllDataset.values[i,0]
    FileName=FileName+'.jpg'
    #print(FileName)    
    IMG = io.imread('./data/resizedversion_256x256/'+ FileName)
    X_all[i,:,:,:]=IMG
    y_all[i,0]=AllDataset.values[i,1]
    # plt.imshow(X_all[i,:,:,:])
    



########################################
############## genreating input outputs from data #######
########################################

TargetTensors=FnCreateTargetImages(y_all)
Y_all_tensors=TargetTensors
#plt.imshow(Y_all_tensors[1,:,:,:])
#print(y_all[1])
#plt.imshow(Y_all_tensors[2,:,:,:])
#print(y_all[2])
#plt.imshow(Y_all_tensors[3,:,:,:])
#print(y_all[3])
#plt.imshow(Y_all_tensors[7,:,:,:])
#print(y_all[7])
#plt.imshow(Y_all_tensors[13,:,:,:])
#print(y_all[13])

######################################################################################
################## NEtwork Architecture ##################################################
##########################################################################################


In_shape=(X_all.shape[1],X_all.shape[2],X_all.shape[3])

In = Input(shape=In_shape)

layer2D_encoder_1=Conv2D(filters=100, kernel_size=(3,3), strides=(2, 2), padding='valid', activation='linear', use_bias=True)(In)
layer2D_encoder_2=Conv2D(filters=50, kernel_size=(3,3), strides=(2, 2), padding='valid', activation='linear', use_bias=True)(layer2D_encoder_1)
layer2D_encoder_3=Conv2D(filters=50, kernel_size=(3,3), strides=(2, 2), padding='valid', activation='linear', use_bias=True)(layer2D_encoder_2)

layer_1D_layer=Flatten()(layer2D_encoder_3)

fc1 = Dense(50, kernel_initializer='normal', activation='linear')(layer_1D_layer)
fc2 = Dense(50, kernel_initializer='normal', activation='linear')(fc1)

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


layer2D_6 = Conv2DTranspose(filters=3, kernel_size=(3,3), strides=(1, 1), kernel_regularizer=tensorflow.keras.regularizers.l2(0.08), activation='linear' )(layer2D_5)
layer2D_6=tensorflow.keras.layers.BatchNormalization()(layer2D_6)

output_1 = tensorflow.keras.layers.Activation('relu')(layer2D_6)#
model_tensorization=Model(inputs= In, outputs=output_1)  

model_tensorization.summary()



##############################################################################
################## End NEtwork Architecture ##########################################
######################################################################################


##########################################################################################
################## Training on folding ###################################################
##########################################################################################


OPTIMIZER_2=tensorflow.keras.optimizers.Adam(lr=0.001, beta_1=0.99, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model_tensorization.compile(loss=custom_loss_2, optimizer=OPTIMIZER_2)
model_tensorization.save_weights('SavedInitialWeights_tensors.h5')
callback_1 = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=30, restore_best_weights=True)
callback_2 = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)


for repeator in range(0,1):

    kfold = StratifiedKFold(n_splits, shuffle=True, random_state=repeator)
    FoldCounter=0
    for train, test in kfold.split(X_all[:,0], y_all[:]):
        FoldCounter=FoldCounter+1   
        
        
        model_tensorization.load_weights('SavedInitialWeights_tensors.h5')        
        Y_train_here_4Net=Y_all_tensors[train,:,:,:]
        X_train_here_4Net=X_all[train,:]
        #print(X_train_here_4Net.shape)
        
        
        print('---Repeat No:  ', repeator+1, '  ---Fold No:  ', FoldCounter)        
        
        
        
        History = model_tensorization.fit(X_train_here_4Net, Y_train_here_4Net, validation_split=0.1,
          epochs=a.max_epochs, batch_size=a.BatchSize, verbose=1
          , callbacks=[callback_1, callback_2])#250-250

        # summarize history for loss
        plt.plot(History.history['loss'])
        plt.plot(History.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.grid()
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(a.output+'Fold_'+str(FoldCounter)+'_History.png')
        plt.close()


        X_test_here_4Net=X_all[test,:]
        Y_test_here_4Net=Y_all_tensors[test,:,:,:]
        for i in range(len(test)):            
            plt.figure(1)
            plt.subplot(121)
            plt.imshow(((model_tensorization.predict(X_test_here_4Net)[i,:,:,:])))
            plt.subplot(122)
            plt.imshow(Y_test_here_4Net[i,:,:,:])
            plt.savefig(a.output+'Fold_'+str(FoldCounter)+ '_Test_'+str(i),dpi=200)
            plt.close('all')
            

       
