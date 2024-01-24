# -*- coding: utf-8 -*-
"""
Created on Jan 2024

@author: Mohammad Eslami, Solale Tabarestani
"""

# Multilayer Perceptron
import pandas
import numpy
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

import math
import argparse
#from tensorflow.keras.utils.np_utils import to_categorical
import tensorflow as tf
import matplotlib.pyplot as plt


import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr, spearmanr
from skimage import io

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
        for j in range (4):
            if Labels.iloc[i,j]=='NL':
                OutputImages[i,:,j*5:(j+1)*5,0]=0
                OutputImages[i,:,j*5:(j+1)*5,1]=1
                OutputImages[i,:,j*5:(j+1)*5,2]=0
                
            elif Labels.iloc[i,j]=='MCI':
                OutputImages[i,:,j*5:(j+1)*5,0]=0
                OutputImages[i,:,j*5:(j+1)*5,1]=0
                OutputImages[i,:,j*5:(j+1)*5,2]=1
                
            elif Labels.iloc[i,j]=='MCI to Dementia':
                OutputImages[i,:,j*5:(j+1)*5,0]=1
                OutputImages[i,:,j*5:(j+1)*5,1]=0
                OutputImages[i,:,j*5:(j+1)*5,2]=1
                
            elif Labels.iloc[i,j]=='Dementia':
                OutputImages[i,:,j*5:(j+1)*5,0]=1
                OutputImages[i,:,j*5:(j+1)*5,1]=0
                OutputImages[i,:,j*5:(j+1)*5,2]=0
                
            # plt.figure()
            # plt.imshow(OutputImages[19,:,:,:])
    return OutputImages

    
####################
###### End of functions ##############    
####################    

n_splits=10

parser = argparse.ArgumentParser()
parser.add_argument("--output", )
parser.add_argument("--max_epochs",  )
parser.add_argument("--BatchSize", )
parser.add_argument("--n_DeepEnsembles", )
a = parser.parse_args()

# comment following lines if you want to use parser
a.n_DeepEnsembles=5
a.max_epochs=4000
a.BatchSize=500
a.output='./v5-4000epochs/'
outputFolderChild=a.output+'/Images/'

import os
try:
    os.stat(a.output)
    os.stat(outputFolderChild)
except:
    os.mkdir(a.output) 
    os.mkdir(outputFolderChild)




####################
###### Reading Data ##############    
####################    

# AllDataset = pandas.read_csv('Data_XY_BLD_CutOff.csv', low_memory=False)
AllDataset = pandas.read_csv('Data_XY_BLD_v0.csv', low_memory=False)
AllDataset = AllDataset.set_index(AllDataset.RID)

AllDataset.columns

###################### MRI ######################
MRI_X = AllDataset.loc[:,['Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp', 'ICV']]
p=MRI_X.values

print(np.nanmin(MRI_X.ICV, axis=0))
print(np.nanmean(MRI_X.ICV, axis=0))
print(np.nanmax(MRI_X.ICV, axis=0))
MRI_X.ICV.isnull().sum()

MRI_Y = AllDataset.loc[:, ['DX_BLD', 'DX_6','DX_12', 'DX_24']]#
MRI_RID = AllDataset.RID
# normalize data
MRI_X = (MRI_X - MRI_X.mean())/ (MRI_X.max() - MRI_X.min())
#MRI_X=MRI_X+1
MRI_X = MRI_X.fillna(0)

###################### PET ######################
PET_X = AllDataset.loc[:,['FDG', 'PIB', 'AV45']]
PET_Y = AllDataset.loc[:, ['DX_BLD', 'DX_6','DX_12', 'DX_24']]#
PET_RID = AllDataset.RID

print(np.nanmin(PET_X.AV45, axis=0))
print(np.nanmean(PET_X.AV45, axis=0))
print(np.nanmax(PET_X.AV45, axis=0))
PET_X.AV45.isnull().sum()

# normalize data
PET_X = (PET_X - PET_X.mean()) / (PET_X.max() - PET_X.min())
#PET_X=PET_X+1
PET_X=PET_X.fillna(0)
###################### COG ######################
COG_X = AllDataset.loc[:, ['RAVLTimmediate', 'RAVLTlearning', 'RAVLTforgetting', 
                           'RAVLTpercforgetting','FAQ', 'MOCA',
                'EcogPtMem', 'EcogPtLang', 'EcogPtVisspat', 'EcogPtPlan',
                'EcogPtOrgan', 'EcogPtDivatt', 'EcogPtTotal',
                'EcogSPMem', 'EcogSPLang', 'EcogSPVisspat', 
                'EcogSPPlan', 'EcogSPOrgan', 'EcogSPDivatt', 'EcogSPTotal']]




COG_Y = AllDataset.loc[:, ['DX_BLD', 'DX_6','DX_12', 'DX_24']]#
COG_RID = AllDataset.RID


print(np.nanmin(COG_X.EcogSPTotal, axis=0))
print(np.nanmean(COG_X.EcogSPTotal, axis=0))
print(np.nanmax(COG_X.EcogSPTotal, axis=0))
COG_X.EcogSPTotal.isnull().sum()




# normalize data
COG_X = (COG_X - COG_X.mean()) / (COG_X.max() - COG_X.min())
#COG_X=COG_X+1
COG_X=COG_X.fillna(0)
###################### CSF ######################
CSF_X = AllDataset.loc[:,['ABETA', 'PTAU', 'TAU']]
CSF_Y = AllDataset.loc[:, ['DX_BLD', 'DX_6','DX_12', 'DX_24']]#
CSF_RID = AllDataset.RID

print(np.nanmin(CSF_X.ABETA, axis=0))
print(np.nanmean(CSF_X.ABETA, axis=0))
print(np.nanmax(CSF_X.ABETA, axis=0))
CSF_X.ABETA.isnull().sum()

# normalize data
CSF_X = (CSF_X - CSF_X.mean()) / (CSF_X.max() - CSF_X.min())
#CSF_X=CSF_X+1
CSF_X=CSF_X.fillna(0)
###################### Risk Factor ######################
RF_X_1 = AllDataset.loc[:,['AGE','PTEDUCAT']] #, 'APOE4', 'PTGENDER']]
RF_Y = AllDataset.loc[:, ['DX_BLD', 'DX_6','DX_12', 'DX_24']]#
RF_RID = AllDataset.RID

# normalize data
RF_X_1 = (RF_X_1 - RF_X_1.mean()) / (RF_X_1.max() - RF_X_1.min())
#RF_X_1=RF_X_1+1
RF_X_1=RF_X_1.fillna(0)

RF_X_A = AllDataset.loc[:,['APOE4']]# ,'PTEDUCAT']] 
print(np.nanmin(RF_X_A.APOE4, axis=0))
print(np.nanmean(RF_X_A.APOE4, axis=0))
print(np.nanmax(RF_X_A.APOE4, axis=0))
RF_X_A.APOE4.isnull().sum()
RF_X_A=RF_X_A-1
RF_X_A=RF_X_A.fillna(0)

RF_X_sex = AllDataset.loc[:,['PTGENDER']] #, 'APOE4', 'PTGENDER']]
RF_X_sex[RF_X_sex=='Male']=-1
RF_X_sex[RF_X_sex=='Female']=1
RF_X_sex=RF_X_sex.fillna(0)


import pandas as pd
RF_X = pd.concat([RF_X_1, RF_X_A, RF_X_sex], axis=1)#, RF_X_sex

##############################################


########################################
############## dataset cleaning #######
########################################

Labels_all_4class=COG_Y.iloc[:,-4:]

Labels_all_3class=Labels_all_4class.copy()
Labels_all_3class=Labels_all_3class.replace('MCI to Dementia','Dementia')

########################################
############## dataset balancing #######
########################################


Finall_Labels=Labels_all_3class

# Finall_Labels.to_csv('Final_Labels_3class.csv')



Counts_BLD=Finall_Labels['DX_BLD'].value_counts()
Counts_6=Finall_Labels['DX_6'].value_counts()
Counts_12=Finall_Labels['DX_12'].value_counts()
Counts_24=Finall_Labels['DX_24'].value_counts()

Balancing_value=np.max([Counts_BLD.MCI,Counts_6.Dementia,Counts_12.MCI,Counts_24.MCI])



########################################
############## genreating input outputs from data #######
########################################

TargetTensors=FnCreateTargetImages(Finall_Labels)
#plt.imshow(TargetTensors[11,:,:,:])
#print(Labels_all_4class.iloc[11])


X_all=[MRI_X.values, PET_X.values, COG_X.values, 
       CSF_X.values, RF_X.values]

YTrain = COG_Y
YTrain1 = YTrain.reset_index()
Y_all_tensors=TargetTensors
X_all_RIDs=RF_RID.values


##########################################################################################
################## Training Models ###################################################
##########################################################################################

from sklearn.model_selection import train_test_split
from designed_network import create_model
indices = np.arange(len(Y_all_tensors))
X_train, X_test, y_train, y_test, indices_train, indices_test, = train_test_split(X_all[1], Y_all_tensors, indices, test_size=0.1, random_state=0)

X_train=[X_all[0][indices_train], 
                           X_all[1][indices_train], X_all[2][indices_train],
                           X_all[3][indices_train],X_all[4][indices_train]]
X_test=[X_all[0][indices_test], 
                           X_all[1][indices_test], X_all[2][indices_test],
                           X_all[3][indices_test],X_all[4][indices_test]]

X_train_names=X_all_RIDs[indices_train]
X_test_names=X_all_RIDs[indices_test]

# Show the results of the split
print("Training set has {} samples.".format(y_train.shape[0]))
print("Testing set has {} samples.".format(y_test.shape[0]))

shape_input_1 = MRI_X.shape[1]
shape_input_2 = PET_X.shape[1]
shape_input_3 = COG_X.shape[1]
shape_input_4 = CSF_X.shape[1]
shape_input_5 = RF_X.shape[1]

list_models = []

# OPTIMIZER_1=tensorflow.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
OPTIMIZER_2=tf.keras.optimizers.Adam(lr=0.001, beta_1=0.99, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# OPTIMIZER_3=tensorflow.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)


import time
callback_1 = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=500, restore_best_weights=True)
callback_2 = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=800, restore_best_weights=True)

ModelCounter=0
for i in range(a.n_DeepEnsembles):
        Model_here = create_model(shape_input_1, shape_input_2, shape_input_3, 
                                  shape_input_4, shape_input_5,)
        Model_here.compile(loss=custom_loss_2, optimizer=OPTIMIZER_2)
        ModelCounter=ModelCounter+1   
        
        Y_train_here_4Net=y_train
        X_train_here_4Net=X_train

        print('---Model No:  ', ModelCounter)    

        start_time = time.time()
        History = Model_here.fit(X_train_here_4Net, Y_train_here_4Net, 
        validation_split=0.1,  epochs=a.max_epochs, batch_size=a.BatchSize,
         verbose=0 , callbacks=[callback_1, callback_2]) # , callbacks=[callback_1, callback_2]
        elapsed_time = time.time() - start_time
        print('----- train elapsed time:', elapsed_time)

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


# save test results
list_y_pred = []    
for m in range(a.n_DeepEnsembles):
    model_here = list_models[m]
    y_pred_test_here = model_here.predict(X_test)
    list_y_pred.append(y_pred_test_here)
arr = np.array(list_y_pred)

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
    #ax.set_xticks([])
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
        
    plt.savefig(a.output+'_Test_RID_'+str(X_test_names[i]), dpi=300)
    plt.close('all')   
        
#save some train samples
    
    