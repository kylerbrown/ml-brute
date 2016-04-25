# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 20:28:37 2016

@author: wmcfadden
"""

import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from sklearn.cross_validation import train_test_split

CATEGORICAL_COLUMNS = ["Make", "Model", "OrdCat", "NVCat"] + ["Cat" + str(i) for i in range(1, 13)] + ["NVVar" + str(i) for i in range(1, 5)]
myTraining = pd.read_csv('train.csv',index_col=0)
myTest = pd.read_csv('test.csv',index_col=0)

print "Loaded dataset...."

#Fix some columns for reduced dimensionality
myTraining['ModelAge'] = myTraining['CalendarYear']-myTraining['ModelYear']
myTest['ModelAge'] = myTest['CalendarYear']-myTest['ModelYear']

#Remove meaningless rows and response from training set for matching
myTraining = myTraining.drop(['CalendarYear'], axis=1)
myTest = myTest.drop(['CalendarYear'], axis=1)

n_tr = myTraining.shape[0]
df_agg = pd.concat([myTraining.drop(['Response'],1),myTest])
label = myTraining['Response']

#Make what should be categorical variables into categorical variables
for col in CATEGORICAL_COLUMNS:
    df_agg[col] = df_agg[col].astype('category')

#Convert all categoricas into 
X_a = pd.get_dummies(df_agg)
training = X_a[:n_tr]
truetest = X_a[n_tr:]
  
#Chop up training data to build our own evaluation scheme  
modtrain = X_a[:n_tr]
modtest = X_a[:n_tr]
#modtrain, modtest, labeltrain, labeltest = train_test_split(training, label, test_size=0.01, stratify=label)

print "Converted dataset...."

#Stitch training data back together and send them out to 
modtrain['Response']=label
modtest['Response']=label
truetest['Response']=-1

modtrain.to_csv('modeltrain.csv')
pd.concat([modtest,truetest]).to_csv('modeltest.csv')
label.to_csv('ans.csv')

print "Saved new datasets...."