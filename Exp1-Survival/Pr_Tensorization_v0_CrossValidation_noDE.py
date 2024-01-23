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

# from tensorflow import set_random_seed
# set_random_seed(2)

import tensorflow.keras
import math
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2DTranspose,Input, Reshape, Conv2D, Flatten
from tensorflow.keras.layers import Dense,concatenate
from sklearn.metrics import mean_squared_error
import argparse
#from tensorflow.keras.utils.np_utils import to_categorical
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
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
    #B = keras.losses.categorical_crossentropy(y_true[:,-4:], y_pred[:,-4:])
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
    OutputImages=np.zeros(shape=(len(Labels),23,23,3))    
    
    for i in range(len(Labels)):
        
        
        if Labels[i]==0:
            OutputImages[i,3:20,3:20,0]=1
            OutputImages[i,3:20,3:20,1]=1
            OutputImages[i,3:20,3:20,2]=1
            
        elif Labels[i]==1:
            OutputImages[i,3:20,3:20,0]=1
            OutputImages[i,3:20,3:20,1]=0.4
            OutputImages[i,3:20,3:20,2]=0
                    
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


a.max_epochs=500
a.BatchSize=1000
a.output='./1/'

import os
try:
    os.stat(a.output)
except:
    os.mkdir(a.output) 



####################
###### Reading Data ##############    
####################    

AllDataset = pandas.read_csv('training_v2.csv', low_memory=False)

# creating independent features X and dependant feature Y
y = AllDataset['hospital_death']
X = AllDataset
X = AllDataset.drop(['hospital_death'],axis = 1)
X = X.drop(['encounter_id'],axis = 1)
X = X.drop(['patient_id'],axis = 1)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
LabelEncoder_X = LabelEncoder()
le = LabelEncoder()
X['gender'] = le.fit_transform(X['gender'].astype(str))
X['ethnicity'] = le.fit_transform(X['ethnicity'].astype(str))
X['hospital_admit_source'] = le.fit_transform(X['hospital_admit_source'].astype(str))
X['icu_admit_source'] = le.fit_transform(X['icu_admit_source'].astype(str))
X['icu_stay_type'] = le.fit_transform(X['icu_stay_type'].astype(str))
X['icu_type'] = le.fit_transform(X['icu_type'].astype(str))
X['apache_2_bodysystem'] = le.fit_transform(X['apache_2_bodysystem'].astype(str))
X['apache_3j_bodysystem'] = le.fit_transform(X['apache_3j_bodysystem'].astype(str))


# from sklearn.preprocessing import Imputer
# imputer = Imputer(missing_values = 'NaN', strategy= 'mean',axis=0)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy= 'mean')
X_all_imputed= imputer.fit_transform(X.values)


############# scaling features

from sklearn import preprocessing
mm_scaler = preprocessing.MinMaxScaler()
X_scaled = mm_scaler.fit_transform(X_all_imputed)
# mm_scaler.transform(X_test)




############## downsampling. because data is hugely biased to 0 label


NumberOf_Label_1=np.sum(y)
NumberOf_Label_0=len(y)-np.sum(y)

Ind_label_tuple=np.nonzero(y.values) 
Ind_label_1=Ind_label_tuple[0]


x_1=X_scaled[Ind_label_1]
x_0=np.delete(X_scaled,Ind_label_1,axis=0)
y_0=y.drop(Ind_label_1)

ind_random_0=np.random.permutation(NumberOf_Label_1)
x_selected_0=x_0[ind_random_0]

X_all=np.concatenate((x_selected_0, x_1), axis=0)


y_1=y[Ind_label_1]
y_selected_0=y_0.iloc[ind_random_0]

y_all=np.concatenate((y_selected_0, y_1), axis=0)



########################################
############## genreating input outputs from data #######
########################################

TargetTensors=FnCreateTargetImages(y_all)
Y_all_tensors=TargetTensors
#plt.imshow(Y_all_tensors[11,:,:,:])
#print(y_all[11])
#plt.imshow(Y_all_tensors[8000,:,:,:])
#print(y_all[8000])


######################################################################################
################## NEtwork Architecture ##################################################
##########################################################################################


In_shape=X_all.shape[1]
In = Input(shape=(In_shape,))
fc1 = Dense(2*In_shape, kernel_initializer='normal', activation='linear')(In)
fc2 = Dense(In_shape, kernel_initializer='normal', activation='linear')(fc1)

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
callback_1 = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=30, restore_best_weights=True)
callback_2 = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
model_tensorization.compile(loss=custom_loss_2, optimizer=OPTIMIZER_2)
model_tensorization.save_weights('SavedInitialWeights_tensors.h5')

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
        
        
        
        History = model_tensorization.fit(X_train_here_4Net, Y_train_here_4Net,
         validation_split=0.1,  epochs=a.max_epochs, batch_size=a.BatchSize,
          verbose=1 , callbacks=[callback_1, callback_2] )# , callbacks=[callback_1, callback_2]


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
            plt.savefig(a.output+'Fold_'+str(FoldCounter)+ '_Test_'+str(i))
            plt.close('all')


