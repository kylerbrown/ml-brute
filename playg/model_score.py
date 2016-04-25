#!/usr/bin/python
import glob
import pandas as pd
import numpy as np
import time

import matplotlib.pylab as plt
plt.style.use('ggplot')

from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import log_loss
### load data in do training

label = pd.read_csv('../data/ans.csv',index_col=0,header=None)
n_tr = label.shape[0]  

path ='../res_test' # use your path
allFiles = glob.glob(path + "/*.csv")
frame = pd.DataFrame()
concat_d = []
concat_t = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col='RowID')
    if('ProbabilityOfResponse' in df):
        data = df.loc[label.index]
        test = df.drop(label.index)
        test.to_csv('../submissions/singlemodel_'+file_.split('/')[-1]+'.csv')
        print 'Mean for '+ file_+' '+str(data['ProbabilityOfResponse'].mean())
        print 'Score for '+file_+' '+str(log_loss(label.values,data['ProbabilityOfResponse'].values))
        concat_d.append(data)
        concat_t.append(test)

train = pd.concat(concat_d,axis=1)
test = pd.concat(concat_t,axis=1)

m_train = train.mean(1)
print 'Score for '+'combined'+' '+str(log_loss(label.values,m_train.values))
m_sub = pd.DataFrame(test.mean(1))
m_sub.columns=['ProbabilityOfResponse']
m_sub.to_csv('../submissions/averagemodels_.csv')
