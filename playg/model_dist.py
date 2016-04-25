#!/usr/bin/python
import glob
import pandas as pd
import numpy as np
import time

import matplotlib.pylab as plt
plt.style.use('ggplot')

from sklearn.metrics import log_loss
### load data in do training

label = pd.read_csv('../data/ans.csv',index_col=0,header=None)
n_tr = label.shape[0]  

path ='../res_test' # use your path
allFiles = glob.glob(path + "/*.csv")
frame = pd.DataFrame(index=label.index)
concat_d = []
concat_t = []
i=1
n = np.ceil(len(allFiles)/2.)
plt.figure(figsize=[8,2*n])
for file_ in allFiles:
    df = pd.read_csv(file_,index_col='RowID')
    if('ProbabilityOfResponse' in df):
        data = df.loc[label.index]
        test = df.drop(label.index)
        test.to_csv('../submissions/singlemodel_'+file_.split('/')[-1]+'.csv')
        plt.subplot(n,2,i)   
        plt.hist(data['ProbabilityOfResponse'].values[label.values.ravel()==0],25,normed=True,alpha=0.5)
        plt.hist(data['ProbabilityOfResponse'].values[label.values.ravel()==1],25,normed=True,alpha=0.5)
#        plt.hist(test['ProbabilityOfResponse'].values,25)
        plt.title(file_.split('/')[-1])        
        plt.xlim([0,1])
        i+=1        
        concat_d.append(data.rename(columns={'ProbabilityOfResponse': file_.split('/')[-1]}))
        concat_t.append(test.rename(columns={'ProbabilityOfResponse': file_.split('/')[-1]}))
plt.show()

train = pd.concat(concat_d,axis=1)
plt.figure()
plt.hist((train.iloc[:,0]*train.iloc[:,1]).values[label.values.ravel()==0],50,normed=True,alpha=0.5)
plt.hist((train.iloc[:,0]*train.iloc[:,1]).values[label.values.ravel()==1],50,normed=True,alpha=0.5)
test = pd.concat(concat_t,axis=1)
