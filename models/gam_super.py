#!/usr/bin/python
import sys, os
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
import time
from pyearth import Earth
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from matplotlib import pyplot
import pandas as pd
import numpy as np


train = pd.read_csv('../data/modeltrain.csv',index_col=0)
test = pd.read_csv('../data/modeltest.csv',index_col=0)
label = train['Response'].values


featextra= pd.read_csv('../feat/improve.csv',index_col=0)
train = pd.concat([train,featextra.loc[train.index]],axis=1)
test = pd.concat([test,featextra.loc[test.index]],axis=1)

featextra= pd.read_csv('../feat/duplicate.csv',index_col=0)
train = pd.concat([train,featextra.loc[train.index]],axis=1)
test = pd.concat([test,featextra.loc[test.index]],axis=1)


feat = train.columns.drop('Response',1)
#Build an Earth model with a logisticregression pipeline
earth_pipe = Pipeline([('earth',Earth(use_fast=True,allow_missing=True,penalty=0.5,max_degree=3)),('log',LogisticRegression())])
earth_pipe.fit(train[feat],label)

#Parameter tuning

#param_grid = {'earth__penalty': np.arange(1,11,2),'earth__max_degree': range(1,4)}
#
#gs1 = GridSearchCV(earth_pipe,param_grid,n_jobs=1,pre_dispatch=1,cv=StratifiedKFold(label, n_folds=5, shuffle=True),scoring='log_loss',verbose=2)
#
#
#gs1.fit(train[feat],label)
#
#print gs1.best_params_
#print gs1.best_score_
#
##----------------------------------------------------------
#
#pipe2 = gs1.best_estimator_
#
#bp = gs1.best_params_['earth__penalty']
#
#param_grid = {'earth__penalty': np.arange(bp-1,bp+1,0.5)}
#
#gs2 = GridSearchCV(pipe2,param_grid,n_jobs=1,pre_dispatch=1,cv=StratifiedKFold(label, n_folds=5, shuffle=True),scoring='log_loss',verbose=2)
#
#gs2.fit(train[feat],label)
#
#print gs2.best_params_
#print gs2.best_score_
#
#
##----------------------------------------------------------
#
#pipe3 = gs2.best_estimator_
#
#param_grid = {'earth__endspan_alpha': [0.01,0.03,0.05],
#              'earth__minspan_alpha': [0.01,0.03,0.05]}
#
#gs3 = GridSearchCV(pipe3,param_grid,n_jobs=1,pre_dispatch=1,cv=StratifiedKFold(label, n_folds=5, shuffle=True),scoring='log_loss',verbose=2)
#
#gs3.fit(train[feat],label)
#
#print gs3.best_params_
#print gs3.best_score_
#
#
#
##Evaluate best model
#
#print gs3.best_estimator_

out_test = earth_pipe.predict_proba(test[feat])[:,1]


df_out = pd.DataFrame({'RowID':test.index,'ProbabilityOfResponse':out_test})
df_out.to_csv('../res_test/gam_super_'+time.strftime("%j-%H-%M")+'.csv',index=False)



