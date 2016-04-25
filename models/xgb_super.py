#!/usr/bin/python
import sys, os
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
import pandas as pd
import numpy as np
import xgboost as xgb
import time
import glob
import matplotlib.pylab as plt

from sklearn.cross_validation import *
from sklearn.grid_search import GridSearchCV

from sklearn import metrics
def modelfit(alg, dtrain, label,useTrainCV=True, cv_folds=10, early_stopping_rounds=25):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain, label=label)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          stratified=True, metrics='logloss', early_stopping_rounds=early_stopping_rounds)
        print cvresult
        print "optimal number of estimators"
        print cvresult['test-logloss-mean'].idxmin()
        alg.set_params(n_estimators=cvresult['test-logloss-mean'].idxmin())
    
    #Fit the algorithm on the data
    alg.fit(dtrain, label,eval_metric='logloss')
        
    #Predict training set:
    dtrain_predprob = alg.predict_proba(dtrain)[:,1]
        
    #Print model report:
    print "\nModel Report"
    print "LogLoss (Train): %f" % metrics.log_loss(label, dtrain_predprob)
                    
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    print feat_imp[0:20]
#    feat_imp.plot(kind='bar', title='Feature Importances')
#    plt.ylabel('Feature Importance Score')
    
### load data in do training
train = pd.read_csv('../data/modeltrain.csv',index_col=0)


test = pd.read_csv('../data/modeltest.csv',index_col=0)

featextra= pd.read_csv('../feat/improve.csv',index_col=0)
train = pd.concat([train,featextra.loc[train.index]],axis=1)
test = pd.concat([test,featextra.loc[test.index]],axis=1)

featextra= pd.read_csv('../feat/duplicate.csv',index_col=0)
train = pd.concat([train,featextra.loc[train.index]],axis=1)
test = pd.concat([test,featextra.loc[test.index]],axis=1)


label = train['Response'].values

feat = train.columns.drop('Response',1)


#xgb_model = xgb.XGBClassifier(learning_rate =0.05,
# n_estimators=1000,
# max_depth=5,
# min_child_weight=1,
# gamma=0,
# colsample_bytree=0.8,
# objective= 'binary:logistic',
# nthread=4,
# scale_pos_weight=1)
#
##-----------------------------------------------------------------------
#
#modelfit(xgb_model,train[feat],label)
#
##-----------------------------------------------------------------------
#
#grid_param = {'max_depth':range(4,11,2),'min_child_weight':range(1,8,2)}
#
#gs1 = GridSearchCV(xgb_model, grid_param, 'log_loss',n_jobs=2,pre_dispatch='n_jobs', 
#                   cv=StratifiedKFold(label, n_folds=10, shuffle=True))
#
#gs1.fit(train[feat], label)
#
#print gs1.best_params_
#print gs1.best_score_
#print gs1.best_estimator_
##-----------------------------------------------------------------------
#
#bmd = gs1.best_params_['max_depth']
#bmcw = gs1.best_params_['min_child_weight']
#
#grid_param = {'max_depth':range(bmd-1,bmd+2,1),'min_child_weight':range(bmcw-1,bmcw+2,1)}
#
#gs2 = GridSearchCV(xgb_model, grid_param, 'log_loss',n_jobs=2,pre_dispatch='n_jobs', 
#                   cv=StratifiedKFold(label, n_folds=10, shuffle=True))
#
#gs2.fit(train[feat], label)
#
#print gs2.best_params_
#print gs2.best_score_
#print gs2.best_estimator_
#-----------------------------------------------------------------------

#xgb_improve = gs2.best_estimator_
##xgb_improve = xgb.XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=0.8,
##       gamma=0, learning_rate=0.05, max_delta_step=0, max_depth=4,
##       min_child_weight=6, missing=None, n_estimators=118, nthread=4,
##       objective='binary:logistic', reg_alpha=0, reg_lambda=1,
##       scale_pos_weight=1, seed=0, silent=True, subsample=1)
#xgb_improve.set_params(n_estimators=xgb_improve.get_params()['n_estimators']*10)
#modelfit(xgb_improve,train[feat],label)
#
##-----------------------------------------------------------------------
#
#grid_param = {
# 'subsample':np.arange(0.6,0.9,0.1),
# 'colsample_bytree':np.arange(0.6,0.9,0.1)
#}
#
#gs3 = GridSearchCV(xgb_improve, grid_param, 'log_loss',n_jobs=2,pre_dispatch='n_jobs', 
#                   cv=StratifiedKFold(label, n_folds=10, shuffle=True))
#
#gs3.fit(train[feat], label)
#
#print gs3.best_params_
#print gs3.best_score_
#print gs3.best_estimator_
#
##-----------------------------------------------------------------------
#
#bss = gs3.best_params_['subsample']
#bcs = gs3.best_params_['colsample_bytree']
#
#grid_param = {
# 'subsample':np.arange(bss-0.05,bss+0.05,0.025),
# 'colsample_bytree':np.arange(bcs-0.05,bcs+0.05,0.025)
#}
#
#gs4 = GridSearchCV(xgb_improve, grid_param, 'log_loss',n_jobs=2,pre_dispatch='n_jobs', 
#                   cv=StratifiedKFold(label, n_folds=10, shuffle=True))
#
#gs4.fit(train[feat], label)
#
#print gs4.best_params_
#print gs4.best_score_
#print gs4.best_estimator_
#
#xgb_improve = gs4.best_estimator_

#-----------------------------------------------------------------------

xgb_improve = xgb.XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=0.55,
       gamma=0, learning_rate=0.05, max_delta_step=0, max_depth=4,
       min_child_weight=6, missing=None, n_estimators=155, nthread=4,
       objective='binary:logistic', reg_alpha=0, reg_lambda=1,
       scale_pos_weight=1, seed=0, silent=True, subsample=0.825)

grid_param = {
 'learning_rate':np.arange(0.01,0.3,0.05),
 }

gs5 = GridSearchCV(xgb_improve, grid_param, 'log_loss',n_jobs=2,pre_dispatch='n_jobs', 
                   cv=StratifiedKFold(label, n_folds=10, shuffle=True))

gs5.fit(train[feat], label)

print gs5.best_params_
print gs5.best_score_
print gs5.best_estimator_
#-----------------------------------------------------------------------

xgb_evenbetter = gs5.best_estimator_
xgb_evenbetter.set_params(n_estimators=xgb_improve.get_params()['n_estimators']*100)
modelfit(xgb_evenbetter,train[feat],label)

#-----------------------------------------------------------------------

blr = gs5.best_params_['learning_rate']

grid_param = {
 'learning_rate':np.arange(blr-0.02,blr+0.02,0.01),
 }

gs6 = GridSearchCV(xgb_improve, grid_param, 'log_loss',n_jobs=2,pre_dispatch='n_jobs', 
                   cv=StratifiedKFold(label, n_folds=10, shuffle=True))

gs6.fit(train[feat], label)

print gs6.best_params_
print gs6.best_score_

print gs6.best_estimator_
#-----------------------------------------------------------------------

xgb_final = gs6.best_estimator_
xgb_final.set_params(n_estimators=xgb_evenbetter.get_params()['n_estimators']*100)
modelfit(xgb_final,train[feat],label)

print xgb_final
#-----------------------------------------------------------------------

sol = xgb_final.predict_proba(test[feat])[:,1]
df_out = pd.DataFrame({'RowID':test.index,'ProbabilityOfResponse':sol})
df_out.to_csv('../res_test/xgb_super_sol_'+time.strftime("%d-%H-%M")+'.csv',index=False)

