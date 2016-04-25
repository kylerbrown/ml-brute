#!/usr/bin/python
#import sys, os
#sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
import pandas as pd
import numpy as np
import time

import matplotlib.pylab as plt

from sklearn.ensemble import RandomForestClassifier

from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV

### load data in do training
train = pd.read_csv('../data/modeltrain.csv',index_col=0)
test = pd.read_csv('../data/modeltest.csv',index_col=0)

featextra= pd.read_csv('../feat/improve.csv',index_col=0)
train = pd.concat([train,featextra.loc[train.index]],axis=1)
test = pd.concat([test,featextra.loc[test.index]],axis=1)

featextra= pd.read_csv('../feat/duplicate.csv',index_col=0)
train = pd.concat([train,featextra.loc[train.index]],axis=1)
test = pd.concat([test,featextra.loc[test.index]],axis=1)

train.fillna(0,inplace=True)
test.fillna(0,inplace=True)

label = train['Response'].values

feat = train.columns.drop('Response',1)


erf= RandomForestClassifier(n_estimators=2000,
 n_jobs=12,bootstrap=True)
erf.fit(train[feat],label)
#grid_param = {'n_estimators':[700,800,900,1000]}
#
#print ('running cross validation')
#gs1 = GridSearchCV(erf_model, grid_param, 'log_loss',n_jobs=2,pre_dispatch='n_jobs',
#                   cv=StratifiedKFold(label, n_folds=10, shuffle=True))
#
#gs1.fit(train[feat], label)
#erf=gs1.best_estimator_
#print gs1.best_score_
#print erf
#
##-----------------------------------------------------------------------------
#
#grid_param = {'max_features':np.arange(0.05,0.3,0.05)}
#
#gs1 = GridSearchCV(erf, grid_param, 'log_loss', n_jobs=1, verbose=3,pre_dispatch=1,
#                   cv=StratifiedKFold(label, n_folds=10, shuffle=True))
#
#gs1.fit(train[feat], label)
#erf=gs1.best_estimator_
#print gs1.best_score_
#print erf
#
##-----------------------------------------------------------------------------

#grid_param = {'n_estimators':[300,400,500,600,700]}
#
#gs1 = GridSearchCV(erf, grid_param, n_jobs=1, verbose=3,
#                   cv=StratifiedKFold(label, n_folds=10, shuffle=True))
#
#gs1.fit(train[feat], label)
#
#erf=gs1.best_estimator_
#print gs1.best_score_
#print erf

#-----------------------------------------------------------------------------

#bn = gs1.best_params_['n_estimators']
#grid_param = {'n_estimators':np.arange(bn-50,bn+50,10)}
#
#gs1 = GridSearchCV(erf, grid_param, n_jobs=2, verbose=3,pre_dispatch='n_jobs',
#                   cv=StratifiedKFold(label, n_folds=10, shuffle=True))
#
#gs1.fit(train[feat], label)
#
#erf=gs1.best_estimator_
#print gs1.best_score_
#print erf

#-----------------------------------------------------------------------------

#print erf.score(train[feat],label)
#print erf.best_params_
#print erf.best_estimator_

sol = erf.predict_proba(test[feat])[:,1]
df_out = pd.DataFrame({'RowID':test.index,'ProbabilityOfResponse':sol})
df_out.to_csv('../res_test/rf2000_sol'+time.strftime("%j-%H-%M")+'.csv',index=False)



