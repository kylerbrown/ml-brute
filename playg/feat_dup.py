# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np


import matplotlib.pylab as plt
plt.style.use('ggplot')

train = pd.read_csv('../data/train.csv',index_col=0)
test = pd.read_csv('../data/test.csv',index_col=0)
#
test['Response']=-1
data = pd.concat([train,test])
data['ModelAge']=data['CalendarYear']-data['ModelYear']
feat = data.columns.drop(['CalendarYear','Response'],1)
feat_list = {'noyear_dup':data[feat].columns,
             'cat_dup':data[feat].filter(regex='Cat|Make|Mod').columns,
             'var_dup':data[feat].filter(regex='Var').columns
}
newdata = pd.DataFrame(index=data.index)
for f in feat_list:
    feat_ = feat_list[f]
    newdata[f]=data.duplicated(feat_,keep=False)*1

#newdata.to_csv('../feat/duplicate.csv')

d2=data.drop('CalendarYear',1)
#groupon = d2.drop(['ModelYear','Response'],1).columns.tolist()

def func(x):
    x['cnt'] = len(x)
    return x
groupon = d2.filter(regex='Cat').columns.tolist()
gr = d2.groupby(groupon)  
cnt = gr.apply(func)
d2['cnt_1']=cnt['cnt']

#groupon = d2.filter(regex='Cat|NVVar').columns.tolist()
#gr = d2.groupby(groupon)  
#d2 = gr.apply(func)
#d2['cnt_2']=d2['cnt']
#
#groupon = d2.filter(regex='NV').columns.tolist()
#gr = d2.groupby(groupon)  
#d2 = gr.apply(func)
#d2['cnt_3']=d2['cnt']
#
#groupon = d2.drop(['Response','ModelYear'],1).columns.tolist()
#gr = d2.groupby(groupon)  
#d2 = gr.apply(func)
#d2['cnt_4']=d2['cnt']
#
#groupon = d2.drop(['Response'],1).columns.tolist()
#gr = d2.groupby(groupon)  
#d2 = gr.apply(func)
#d2['cnt_5']=d2['cnt']
#d2.drop('cnt',1)


#d2.to_csv('../feat/duplicate.csv')


#v=list(gr.groups.values())
#vlong = []
#for i in range(len(v)):
#    vlong.append([i,len(v[i])])
#vl = np.array(vlong)
#vl = vl[vl[:,1].argsort(),:]
#plt.hist(vl[:,1],50)