# -*- coding: utf-8 -*-
"""
Created on Wed May  1 10:28:19 2019

@author: User
"""

# Multilayer Perceptron
import pandas
import numpy
import tensorflow as tf
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# from tensorflow import set_random_seed
# set_random_seed(2)

import math
import argparse
#from tensorflow.keras.utils.np_utils import to_categorical

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
from sklearn.model_selection import train_test_split

####################
####### Functions #############
####################


def custom_loss_2 (y_true, y_pred):
    A = tf.keras.losses.mean_absolute_error(y_true, y_pred)
    #B = keras.losses.categorical_crossentropy(y_true[:,-4:], y_pred[:,-4:])
    return A


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


####################
###### End of functions ##############    
####################    


####################
###### Reading input arguments ##############        
#########################################################
######### Hyper paramters configurations ##################
########### ##########################################################

parser = argparse.ArgumentParser()
parser.add_argument("--output", )
parser.add_argument("--max_epochs",  )
parser.add_argument("--BatchSize", )
parser.add_argument("--n_DeepEnsembles", )
a = parser.parse_args()

# comment following lines if you want to use parser
a.n_DeepEnsembles=5
a.max_epochs=500
a.BatchSize=1000
a.output='./1/'
###########


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


##########################################################################################
################## Training and testing ###################################################
##########################################################################################


from designed_network import create_model

X_train, X_test, y_train, y_test = train_test_split(X_all, Y_all_tensors, test_size=0.33, random_state=42)

list_models = []

OPTIMIZER_2= tf.keras.optimizers.Adam(lr=0.001, beta_1=0.99, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
callback_1 = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=30, restore_best_weights=True)
callback_2 = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

ModelCounter=0
for i in range(a.n_DeepEnsembles):
        
        Model_here = create_model(input_shape=X_all.shape[1])
        Model_here.compile(loss=custom_loss_2, optimizer=OPTIMIZER_2)
        ModelCounter=ModelCounter+1   
        
        Y_train_here_4Net=y_train
        X_train_here_4Net=X_train
        #print(X_train_here_4Net.shape)
        
        print('---Model No:  ', ModelCounter)        
        
        History = Model_here.fit(X_train_here_4Net, Y_train_here_4Net, shuffle=True,
         validation_split=0.1,  epochs=a.max_epochs, batch_size=a.BatchSize,
          verbose=0 , callbacks=[callback_1, callback_2] )# , callbacks=[callback_1, callback_2]

        # summarize history for loss
        plt.plot(History.history['loss'])
        plt.plot(History.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.grid()
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(a.output+'Model_'+str(ModelCounter)+'_History.png')
        plt.close()

        list_models.append(Model_here)

list_y_pred = []    
for m in range(a.n_DeepEnsembles):
    model_here = list_models[m]
    y_pred_test_here = model_here.predict(X_test)
    list_y_pred.append(y_pred_test_here)

arr = np.array(list_y_pred)
#print(arr.shape)

for i in range(len(y_test)):
    pred = arr[:,i,:,:,:]           
    pred = np.squeeze(pred)    

    pre_avg_i = np.mean(pred, axis=0)
    pre_std_i = np.std(pred, axis=0)
    #print(pre_avg_i.shape)
    #print(np.max(pre_avg_i), np.min(pre_avg_i) )
    #print(np.max(pre_std_i), np.min(pre_std_i) )

    plt.figure(1)
    plt.subplot(131)
    plt.imshow(y_test[i,:,:,:],vmin=0, vmax=1)    
    ax = plt.gca()
    ax.set_title('Target')
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.subplot(132)
    plt.imshow(pre_avg_i,vmin=0, vmax=1)
    ax = plt.gca()
    ax.set_title('Avg. of Preds')
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.subplot(133)
    plt.imshow(pre_std_i,vmin=0, vmax=1)
    ax = plt.gca()
    ax.set_title('STD. of Preds')
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.savefig(a.output+'_Test_'+str(i))
    plt.close('all')    
