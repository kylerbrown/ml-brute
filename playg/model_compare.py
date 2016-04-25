#!/usr/bin/python
#%%
import glob
import pandas as pd
import numpy as np
import time

import matplotlib.pylab as plt
plt.style.use('ggplot')

from sklearn.metrics import log_loss
#%% load data in do training

label = pd.read_csv('../data/ans.csv',index_col=0,header=None)
n_tr = label.shape[0]  
binsize=60

path ='../res_test' # use your path
allFiles = glob.glob(path + "/*.csv")
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
        plt.hist(data['ProbabilityOfResponse'].values[label.values.ravel()==0],binsize,normed=True,alpha=0.5)
        plt.hist(data['ProbabilityOfResponse'].values[label.values.ravel()==1],binsize,normed=True,alpha=0.5)
#        plt.hist(test['ProbabilityOfResponse'].values,25)
        plt.title(file_.split('/')[-1])        
        plt.xlim([0,1])
        i+=1        
        concat_d.append(data.rename(columns={'ProbabilityOfResponse': file_.split('/')[-1]}))
        concat_t.append(test.rename(columns={'ProbabilityOfResponse': file_.split('/')[-1]}))
plt.show()
#%% 
train = pd.concat(concat_d,axis=1)
test = pd.concat(concat_t,axis=1)
plt.figure()
plt.hist(train.mean(1).values[label.values.ravel()==0],binsize,normed=True,alpha=0.5)
plt.hist(train.mean(1).values[label.values.ravel()==1],binsize,normed=True,alpha=0.5)
plt.xlim([0,1])
#%%
p1 = 'erf200_sol115-12-47.csv'
p2= 'xgb_super_sol_23-18-55.csv'
plt.figure()
H, xedges, yedges = np.histogram2d(train[p2].values[label.values.ravel()==0],train[p1].values[label.values.ravel()==0],binsize,[[0, 1], [0, 1]])
plt.imshow(H, interpolation='nearest', origin='low', extent=[0, 1, 0, 1],cmap='Reds',alpha=1)
H, xedges, yedges = np.histogram2d(train[p2].values[label.values.ravel()==1],train[p1].values[label.values.ravel()==1],binsize,[[0, 1], [0, 1]])
plt.imshow(H, interpolation='nearest', origin='low', extent=[0, 1, 0, 1],cmap='Blues',alpha=0.6)
plt.xlabel('erf')
plt.ylabel('xgb')

#%%
plt.figure()
plt.scatter(train[p1].values[label.values.ravel()==0],train[p2].values[label.values.ravel()==0],c='red',alpha=0.05)
plt.scatter(train[p1].values[label.values.ravel()==1],train[p2].values[label.values.ravel()==1],c='blue',alpha=0.05)
plt.xlabel('erf')
plt.ylabel('xgb')
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim([0,1])
plt.ylim([0,1])

#%% trial
plt.figure()
t1 = train[p2]
t2 = train[p1]
sqrd = np.sqrt(t1*t2)
r1 = r1=sqrd*0.28/sqrd.mean()
r1 = pd.DataFrame((t1+t2)/2)
plt.hist(r1.values[label.values.ravel()==0],binsize,normed=True,alpha=0.5)
plt.hist(r1.values[label.values.ravel()==1],binsize,normed=True,alpha=0.5)
plt.xlim([0,1])
r1.index.rename('RowID',inplace=True)
r1.columns=['ProbabilityOfResponse']
r1.to_csv('../submissions/arithmean_.csv')

#%%
t1 = test[p2]
t2 = test[p1]
sqrd = np.sqrt(t1*t2)
r1 = r1=sqrd*0.28/sqrd.mean()
r1 = pd.DataFrame((t1+t2)/2)
plt.hist(r1.values,binsize,normed=True)
r1.index.rename('RowID',inplace=True)
r1.columns=['ProbabilityOfResponse']
r1.to_csv('../submissions/arithmean_.csv')


#%%
plt.figure()
plt.scatter(test[p1].values,test[p2].values,c='red',alpha=0.5)
plt.xlabel('erf')
plt.ylabel('xgb')
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim([0,1])
plt.ylim([0,1])

#%%
plt.figure()
H, xedges, yedges = np.histogram2d(test[p2].values,test[p1].values,binsize,[[0, 1], [0, 1]])
plt.imshow(np.log(H), interpolation='nearest', origin='low', extent=[0, 1, 0, 1],cmap='Reds',alpha=1)


#%%
plt.figure()
sqrd = np.sqrt(t1*t2)
r1 = r1=sqrd*0.28/sqrd.mean()
r1 = pd.DataFrame((t1+t2)/2)
plt.hist(t1.values,binsize,normed=True)
plt.hist(t2.values,binsize,normed=True)
plt.hist(r1.values,binsize,normed=True,)
plt.xlim([0,1])
r1.index.rename('RowID',inplace=True)
r1.columns=['ProbabilityOfResponse']
r1.to_csv('../submissions/geomean_.csv')


#%%
tnyshift = t1
tnyshift[(t1>0.0075) & (t1<0.125) & (t2<0.4)] = tnyshift[(t1>0.0075) & (t1<0.125) & (t2<0.4)]-0.005
plt.figure()
plt.hist(tnyshift.values,50,normed=True)