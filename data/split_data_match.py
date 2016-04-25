# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 20:28:37 2016

@author: wmcfadden
"""

import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegressionCV
import matplotlib.pylab as plt

CATEGORICAL_COLUMNS = ["Make", "Model", "OrdCat", "NVCat"] + ["Cat" + str(i) for i in range(1, 13)]
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
myTest['Response']=-1
df_agg = pd.concat([myTraining,myTest])

#Make what should be categorical variables into categorical variables
for col in CATEGORICAL_COLUMNS:
    df_agg[col] = df_agg[col].astype('category')

#Convert all categoricas into 
X_a = pd.get_dummies(df_agg)

matchlabel = np.concatenate((np.zeros([n_tr,1]),np.ones([X_a.shape[0]-n_tr,1])))

matchreg = LogisticRegressionCV()
matchreg.fit(X_a.drop('Response',1),matchlabel)

#Chop up training data to build our own evaluation scheme  

estim= matchreg.predict_proba(X_a.drop('Response',1))[:,1]
plt.plot(estim,'.')
print "Converted dataset...."

X_a['match']=estim
truetest = X_a[n_tr:]
training = X_a[:n_tr].sort('match',ascending=False)[:50000]
truetest.drop('Response',1)


#Stitch training data back together and send them out to 


training.to_csv('matchtrain.csv')
truetest.to_csv('matchtest.csv')

print "Saved new datasets...."