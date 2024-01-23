# -*- coding: utf-8 -*-
"""
Created on Jan  2024

@author: Mohammad Eslami
"""

import pandas
import numpy
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
from sklearn.model_selection import train_test_split

import math
import argparse
#from tensorflow.keras.utils.np_utils import to_categorical
import tensorflow as tf
import matplotlib.pyplot as plt

import numpy as np

####################
####### Functions #############
####################


def custom_loss_2 (y_true, y_pred):
    A = tf.keras.losses.mean_absolute_error(y_true, y_pred)
    return A


def FnCreateTargetImages(Labels_DRlevel,Labels_quality,Labels_clarity):
    # Neon color
    OutputImages=np.zeros(shape=(len(Labels_DRlevel),23,23,3))    
    
    for i in range(len(Labels_DRlevel)):
          
        if Labels_DRlevel[i]==0:
            OutputImages[i,6:17,6:17,0]=0
            OutputImages[i,6:17,6:17,1]=1
            OutputImages[i,6:17,6:17,2]=0            
        elif Labels_DRlevel[i]==1:
            OutputImages[i,6:17,6:17,0]=0.5
            OutputImages[i,6:17,6:17,1]=0.8
            OutputImages[i,6:17,6:17,2]=0
        elif Labels_DRlevel[i]==2:
            OutputImages[i,6:17,6:17,0]=1
            OutputImages[i,6:17,6:17,1]=1
            OutputImages[i,6:17,6:17,2]=0            
        elif Labels_DRlevel[i]==3:
            OutputImages[i,6:17,6:17,0]=1
            OutputImages[i,6:17,6:17,1]=0.5
            OutputImages[i,6:17,6:17,2]=0
        elif Labels_DRlevel[i]==4:
            OutputImages[i,6:17,6:17,0]=1
            OutputImages[i,6:17,6:17,1]=0
            OutputImages[i,6:17,6:17,2]=0

            
        if Labels_quality[i]==0:
            OutputImages[i,2:4,2:21,0]=0.2
            OutputImages[i,2:4,2:21,1]=0.2
            OutputImages[i,2:4,2:21,2]=0.2            
        elif Labels_quality[i]==1:
            OutputImages[i,2:4,2:21,0]=1
            OutputImages[i,2:4,2:21,1]=1
            OutputImages[i,2:4,2:21,2]=1            

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
a.BatchSize=100
InputFolder='./data/resizedversion_256x256_drid/'
a.output='./v4_DE/'

import os
try:
    os.stat(a.output)
except:
    os.mkdir(a.output) 



####################
###### Reading Data ##############    
####################    

AllDataset = pandas.read_csv('./data/data.csv', low_memory=False)
AllDataset=AllDataset.dropna()


from skimage import io


X_all=np.zeros((len(AllDataset),256,256,3),dtype=np.uint8)
y_all_DRLevel=np.zeros((len(AllDataset),1), dtype=int)
y_all_clarity=np.zeros((len(AllDataset),1), dtype=int)
y_all_quality=np.zeros((len(AllDataset),1), dtype=int)
X_all_names=[]

for i in range(len(AllDataset)):
    FileName=AllDataset.values[i,0]
    FileName=FileName+'.jpg'
    #print(FileName)    
    IMG = io.imread(InputFolder+ FileName)
    X_all[i,:,:,:]=IMG
    y_all_DRLevel[i,0]=AllDataset.values[i,1]
    y_all_clarity[i,0]=AllDataset.values[i,3]
    y_all_quality[i,0]=AllDataset.values[i,2]    
#    X_all_names[i]=AllDataset.values[i,0]
    X_all_names.append(AllDataset.values[i,0])
    # plt.imshow(X_all[i,:,:,:])
    
X_all_names=np.array(X_all_names)    

########################################
############## genreating outputs #######
########################################

TargetTensors=FnCreateTargetImages(y_all_DRLevel,y_all_quality,y_all_clarity)
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


##########################################################################################
################## Training on folding ###################################################
##########################################################################################


from designed_network import create_model
indices = np.arange(len(X_all))
X_train, X_test, y_train, y_test, indices_train, indices_test, = train_test_split(X_all, Y_all_tensors, indices, test_size=0.33, random_state=42)
X_test_names = X_all_names[indices_test]


shape_input=(X_all.shape[1],X_all.shape[2],X_all.shape[3])

list_models = []

OPTIMIZER_2= tf.keras.optimizers.Adam(lr=0.001, beta_1=0.99, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
callback_1 = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=30, restore_best_weights=True)
callback_2 = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

ModelCounter=0

for i in range(a.n_DeepEnsembles):
        
        Model_here = create_model(shape_input=shape_input)
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
    plt.savefig(a.output+'_Test_Img_'+X_test_names[i],dpi=300)
    plt.close('all')    
