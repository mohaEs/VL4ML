# -*- coding: utf-8 -*-
"""
@author: Mohammad Eslami
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np

####################
####### Functions #############
####################

def custom_loss_2 (y_true, y_pred):
    A = tf.keras.losses.mean_absolute_error(y_true, y_pred)
    return A


def FnCreateTargetImages(Labels):
    OutputImages=np.zeros(shape=(len(Labels),45,45,3))    
    
    lables_int=Labels.astype(int)
    for i in range(len(Labels)):
                
        OutputImages[i,10:35,:lables_int[i],0]=0
        OutputImages[i,10:35,:lables_int[i],1]=1
        OutputImages[i,10:35,:lables_int[i],2]=1
            

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
a.max_epochs=300
a.BatchSize=300
a.output='./1/'

import os
try:
    os.stat(a.output)
except:
    os.mkdir(a.output) 



####################
###### Preparing Data ##############    
###### code from https://github.com/daniel-codes/hospital-los-predictor ##############    
#################### 


###### Reading Data     
####################    

# Primary Admissions information
df = pandas.read_csv('./data/ADMISSIONS.csv')

# Patient specific info such as gender
df_pat = pandas.read_csv('./data/PATIENTS.csv')

# Diagnosis for each admission to hospital
df_diagcode = pandas.read_csv('./data/DIAGNOSES_ICD.csv')

# Intensive Care Unit (ICU) for each admission to hospital
df_icu = pandas.read_csv('./data/ICUSTAYS.csv')

################# ADMISSIONS.csv Exploration
#################

df.info()
print('Dataset has {} number of unique admission events.'.format(df['HADM_ID'].nunique()))
print('Dataset has {} number of unique patients.'.format(df['SUBJECT_ID'].nunique()))


# Convert admission and discharge times to datatime type
df['ADMITTIME'] = pandas.to_datetime(df['ADMITTIME'])
df['DISCHTIME'] = pandas.to_datetime(df['DISCHTIME'])

# Convert timedelta type into float 'days', 86400 seconds in a day
df['LOS'] = (df['DISCHTIME'] - df['ADMITTIME']).dt.total_seconds()/86400

# Verify
df[['ADMITTIME', 'DISCHTIME', 'LOS']].head()
df['LOS'].describe()
# Look at what is happening with negative LOS values
df[df['LOS'] < 0]

# Drop rows with negative LOS, usually related to a time of death before admission
df['LOS'][df['LOS'] > 0].describe()
# Drop LOS < 0 
df = df[df['LOS'] > 0]
## Plot LOS Distribution
# plt.hist(df['LOS'], bins=200, color = '#55a868')
# plt.xlim(0, 50)
# plt.title('Distribution of LOS for all hospital admissions \n incl. deceased')
# plt.ylabel('Count')
# plt.xlabel('Length-of-Stay (days)')
# plt.tick_params(top=False, right=False) 
# plt.show()

# Pre-emptively drop some columns that I don't need anymore
df.drop(columns=['DISCHTIME', 'ROW_ID', 
                'EDREGTIME', 'EDOUTTIME', 'HOSPITAL_EXPIRE_FLAG',
                'HAS_CHARTEVENTS_DATA'], inplace=True)    


    # Mark admissions where patients died in boolean column
df['DECEASED'] = df['DEATHTIME'].notnull().map({True:1, False:0})
print("{} of {} patients died in the hospital".format(df['DECEASED'].sum(),df['SUBJECT_ID'].nunique()))
# Look at statistics less admissions resulting in death
df['LOS'].loc[df['DECEASED'] == 0].describe()

# Hospital LOS metrics for later comparison
actual_mean_los = df['LOS'].loc[df['DECEASED'] == 0].mean() 
actual_median_los = df['LOS'].loc[df['DECEASED'] == 0].median() 
print(actual_mean_los)
print(actual_median_los)

    
# plt.hist(df['LOS'].loc[df['DECEASED'] == 0], bins=200, color = '#55a868')
# plt.xlim(0, 50)
# plt.title('Distribution of LOS for hospital admissions')
# plt.ylabel('Count')
# plt.xlabel('Length-of-Stay (days)')
# plt.tick_params(top=False, right=False) 
# plt.show()


df['ETHNICITY'].value_counts()
# Compress the number of ethnicity categories
df['ETHNICITY'].replace(regex=r'^ASIAN\D*', value='ASIAN', inplace=True)
df['ETHNICITY'].replace(regex=r'^WHITE\D*', value='WHITE', inplace=True)
df['ETHNICITY'].replace(regex=r'^HISPANIC\D*', value='HISPANIC/LATINO', inplace=True)
df['ETHNICITY'].replace(regex=r'^BLACK\D*', value='BLACK/AFRICAN AMERICAN', inplace=True)
df['ETHNICITY'].replace(['UNABLE TO OBTAIN', 'OTHER', 'PATIENT DECLINED TO ANSWER', 
                         'UNKNOWN/NOT SPECIFIED'], value='OTHER/UNKNOWN', inplace=True)
df['ETHNICITY'].loc[~df['ETHNICITY'].isin(df['ETHNICITY'].value_counts().nlargest(5).index.tolist())] = 'OTHER/UNKNOWN'
df['ETHNICITY'].value_counts()


df['RELIGION'].value_counts()
# Reduce categories to terms of religious or not
# I tested with and without category reduction, with little change in R2 score
df['RELIGION'].loc[~df['RELIGION'].isin(['NOT SPECIFIED', 'UNOBTAINABLE'])] = 'RELIGIOUS'
print(df['RELIGION'].value_counts())
print(df['RELIGION'].value_counts()[0]/len(df['RELIGION']))
print(df['RELIGION'].value_counts()[1]/len(df['RELIGION']))
print(df['RELIGION'].value_counts()[2]/len(df['RELIGION']))


df['ADMISSION_TYPE'].value_counts()
df['INSURANCE'].value_counts()
df['MARITAL_STATUS'].value_counts(dropna=False)
# Fix NaNs and file under 'UNKNOWN'
df['MARITAL_STATUS'] = df['MARITAL_STATUS'].fillna('UNKNOWN (DEFAULT)')
df['MARITAL_STATUS'].value_counts(dropna=False)



################# DIAGNOSES_ICD.csv Exploration
#################


df_diagcode.info()
print('There are {} unique ICD9 codes in this dataset.'.format(df_diagcode['ICD9_CODE'].value_counts().count()))
# Filter out E and V codes since processing will be done on the numeric first 3 values
df_diagcode['recode'] = df_diagcode['ICD9_CODE']
df_diagcode['recode'] = df_diagcode['recode'][~df_diagcode['recode'].str.contains("[a-zA-Z]").fillna(False)]
df_diagcode['recode'].fillna(value='999', inplace=True)

# https://stackoverflow.com/questions/46168450/replace-specific-range-of-values-in-data-frame-pandas
df_diagcode['recode'] = df_diagcode['recode'].str.slice(start=0, stop=3, step=1)
df_diagcode['recode'] = df_diagcode['recode'].astype(int)


# ICD-9 Main Category ranges
icd9_ranges = [(1, 140), (140, 240), (240, 280), (280, 290), (290, 320), (320, 390), 
               (390, 460), (460, 520), (520, 580), (580, 630), (630, 680), (680, 710),
               (710, 740), (740, 760), (760, 780), (780, 800), (800, 1000), (1000, 2000)]

# Associated category names
diag_dict = {0: 'infectious', 1: 'neoplasms', 2: 'endocrine', 3: 'blood',
             4: 'mental', 5: 'nervous', 6: 'circulatory', 7: 'respiratory',
             8: 'digestive', 9: 'genitourinary', 10: 'pregnancy', 11: 'skin', 
             12: 'muscular', 13: 'congenital', 14: 'prenatal', 15: 'misc',
             16: 'injury', 17: 'misc'}

# Re-code in terms of integer
for num, cat_range in enumerate(icd9_ranges):
    df_diagcode['recode'] = np.where(df_diagcode['recode'].between(cat_range[0],cat_range[1]), 
            num, df_diagcode['recode'])
    
# Convert integer to category name using diag_dict
df_diagcode['recode'] = df_diagcode['recode']
df_diagcode['cat'] = df_diagcode['recode'].replace(diag_dict)
# Verify
df_diagcode.head()

# Create list of diagnoses for each admission
hadm_list = df_diagcode.groupby('HADM_ID')['cat'].apply(list).reset_index()
hadm_list.head()
# Convert diagnoses list into hospital admission-item matrix
hadm_item = pandas.get_dummies(hadm_list['cat'].apply(pandas.Series).stack()).sum(level=0)
hadm_item.head()

# Join back with HADM_ID, will merge with main admissions DF later
hadm_item = hadm_item.join(hadm_list['HADM_ID'], how="outer")
hadm_item.head()

# Merge with main admissions df
df = df.merge(hadm_item, how='inner', on='HADM_ID')
# Verify Merge
df.head()


# Look at the median LOS by diagnosis category
diag_cat_list = ['skin', 'infectious',  'misc', 'genitourinary', 'neoplasms', 'blood', 'respiratory', 
                  'congenital','nervous', 'muscular', 'digestive', 'mental', 'endocrine', 'injury',
                 'circulatory', 'prenatal',  'pregnancy']

results = []
for variable in diag_cat_list:
    results.append(df[[variable, 'LOS']].groupby(variable).median().reset_index().values[1][1])
    
    
# import seaborn as sns    

# sns.set(style="whitegrid")
# fig, ax = plt.subplots(figsize=(7,5))
# ind = range(len(results))
# ax.barh(ind, results, align='edge', color = '#55a868', alpha=0.8)
# ax.set_yticks(ind)
# ax.set_yticklabels(diag_cat_list)
# ax.set_xlabel('Median Length of Stay (days)')
# ax.tick_params(left=False, right=False, top=False) 
# ax.set_title('Comparison of Diagnoses'.format(variable))
# plt.show()




################# Patients.csv Exploration
#################

df_pat.head()
df_pat['GENDER'].value_counts()
# Convert to datetime type
df_pat['DOB'] = pandas.to_datetime(df_pat['DOB'])
df_pat = df_pat[['SUBJECT_ID', 'DOB', 'GENDER']]
df_pat.head()

df = df.merge(df_pat, how='inner', on='SUBJECT_ID')

# Find the first admission time for each patient
df_age_min = df[['SUBJECT_ID', 'ADMITTIME']].groupby('SUBJECT_ID').min().reset_index()
df_age_min.columns = ['SUBJECT_ID', 'ADMIT_MIN']
df_age_min.head()
df = df.merge(df_age_min, how='outer', on='SUBJECT_ID')
# Verify merge
df.head()

# Age is decode by finding the difference in admission date and date of birth
df['ADMIT_MIN'] = pandas.to_datetime(df['ADMIT_MIN']).dt.date
df['DOB'] = pandas.to_datetime(df['DOB']).dt.date
df['age'] = df.apply(lambda e: (e['ADMIT_MIN'] - e['DOB']).days/365, axis=1)
#df['age'] = (df['ADMIT_MIN'] - df['DOB']).dt.days // 365
df['age'] = np.where(df['age'] < 0, 90, df['age'])
#df['age'] = np.where(df['age'] == -0, 0, df['age'])
df['age'].isnull().sum()


## Note that no ‘middle’ patients show up - this reflects the fact that MIMIC-III does not contain data from pediatric patients.
# plt.hist(df['age'], bins=20, color='#c44e52')
# plt.ylabel('Count')
# plt.xlabel('Age (years)')
# plt.title('Distribution of Age in MIMIC-III')
# plt.tick_params(left=False, bottom=False, top=False, right=False) 
# plt.show()


# plt.scatter(df['age'], df['LOS'], alpha=0.005)
# #plt.yscale('sqrt')
# plt.ylabel('LOS (days)')
# plt.xlabel('Age (years)')
# plt.title('Age versus Length-of-stay')
# plt.ylim(1, 50)

# https://en.wikipedia.org/wiki/List_of_ICD-9_codes
age_ranges = [(0, 13), (13, 36), (36, 56), (56, 100)]
for num, cat_range in enumerate(age_ranges):
    df['age'] = np.where(df['age'].between(cat_range[0],cat_range[1]), 
            num, df['age'])
    
age_dict = {0: 'newborn', 1: 'young_adult', 2: 'middle_adult', 3: 'senior'}
df['age'] = df['age'].replace(age_dict)
df.age.value_counts()



df['GENDER'].replace({'M': 0, 'F':1}, inplace=True)




################# ICUSTAYS.csv Exploration
#################

# Intensive Care Unit (ICU) for each admission to hospital
df_icu.info()
df_icu['HADM_ID'].nunique()

df_icu.groupby('FIRST_CAREUNIT').median()
# Based on above statistics, reduce to just ICU and NICU groups
df_icu['FIRST_CAREUNIT'].replace({'CCU': 'ICU', 'CSRU': 'ICU', 'MICU': 'ICU',
                                  'SICU': 'ICU', 'TSICU': 'ICU'}, inplace=True)
df_icu['cat'] = df_icu['FIRST_CAREUNIT']
icu_list = df_icu.groupby('HADM_ID')['cat'].apply(list).reset_index()
icu_list.head()


df_icu['FIRST_CAREUNIT'].value_counts()
# Create admission-ICU matrix
icu_item = pandas.get_dummies(icu_list['cat'].apply(pandas.Series).stack()).sum(level=0)
icu_item[icu_item >= 1] = 1
icu_item = icu_item.join(icu_list['HADM_ID'], how="outer")
icu_item.head()

print("Number of admissions to ICU {}.".format(icu_item.ICU.sum()))
print("Number of admissions to NICU {}.".format(icu_item.NICU.sum()))

# Merge ICU data with main dataFrame
df = df.merge(icu_item, how='outer', on='HADM_ID')

# Replace NaNs with 0
df['ICU'].fillna(value=0, inplace=True)
df['NICU'].fillna(value=0, inplace=True)
# Verify NaN fix
print(df.ICU.value_counts(dropna=False))
print(df.NICU.value_counts(dropna=False))




################# data preprocessing
#################

# Look at what is no longer needed in the DataFrame
df.info()
# Remove deceased persons as they will skew LOS result
df = df[df['DECEASED'] == 0]
# Remove LOS with negative number, likely entry form error
df = df[df['LOS'] > 0]

# Drop unused or no longer needed columns
df.drop(columns=['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'ADMISSION_LOCATION',
                'DISCHARGE_LOCATION', 'LANGUAGE', 'ADMIT_MIN', 'DOB',
                'DIAGNOSIS', 'DECEASED',  'DEATHTIME'], inplace=True)
df.info()


# Create dummy columns for categorical variables
prefix_cols = ['ADM', 'INS', 'REL', 'ETH', 'AGE', 'MAR', 'RELIGION']
dummy_cols = ['ADMISSION_TYPE', 'INSURANCE', 'RELIGION',
             'ETHNICITY', 'age', 'MARITAL_STATUS', 'RELIGION']
df = pandas.get_dummies(df, prefix=prefix_cols, columns=dummy_cols)
df.info()
# Verify
df.head()

# Check for any remaining NaNs
df.isnull().values.sum()



################# 
#################

## since my image is 45x45, remove data with los more than 40
df = df[df['LOS'] < 45]

# Target Variable (Length-of-Stay)
y_all = df['LOS'].values
# Prediction Features
X_all = df.drop(columns=['LOS'])
# plt.hist(y_all, bins=20, color='#c44e52')
# plt.show()

####################
###### End of Preparing Data ##############    
######  ##############    
#################### 


########################################
############## genreating input outputs from data #######
########################################

TargetTensors=FnCreateTargetImages(y_all)
Y_all_tensors=TargetTensors
#plt.imshow(Y_all_tensors[11,:,:,:])
#print(y_all[11])
#plt.imshow(Y_all_tensors[8000,:,:,:])
#print(y_all[8000])


##############################################################################
################## End NEtwork Architecture ##########################################
######################################################################################


##########################################################################################
################## Training on folding ###################################################
##########################################################################################

from sklearn.model_selection import train_test_split
from designed_network import create_model
indices = np.arange(len(X_all))
X_train, X_test, y_train, y_test, indices_train, indices_test, = train_test_split(X_all, Y_all_tensors, indices, test_size=0.1, random_state=0)
# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))

shape_input = X_all.shape[1]
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

        print('---Model No:  ', ModelCounter)       


        History = Model_here.fit(X_train_here_4Net, Y_train_here_4Net, 
        validation_split=0.1,  epochs=a.max_epochs, batch_size=a.BatchSize, 
        verbose=0, callbacks=[callback_1, callback_2])#250-250

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

plt.rcParams["axes.grid"] = False

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
    #ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    #ax.set_xticks([])
    ax.set_yticks([])

    plt.subplot(132)
    plt.imshow(pre_avg_i,vmin=0, vmax=1)
    ax = plt.gca()
    ax.set_title('Avg. of Preds')
    #ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    #ax.set_xticks([])
    ax.set_yticks([])

    plt.subplot(133)
    plt.imshow(pre_std_i,vmin=0, vmax=1)
    ax = plt.gca()
    ax.set_title('STD. of Preds')
    #ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    #ax.set_xticks([])
    ax.set_yticks([])
        
    plt.savefig(a.output+'_Test_'+str(i),dpi=300)
    plt.close('all')   
        
       
            
