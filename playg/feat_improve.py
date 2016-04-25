# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV


import matplotlib.pylab as plt
plt.style.use('ggplot')

train = pd.read_csv('../data/train.csv',index_col=0)
test = pd.read_csv('../data/test.csv',index_col=0)

data = pd.concat([train.drop('Response',1),test])

feat_cat = data.filter(regex='Cat|Make|Mod|Year').columns
newdata = pd.DataFrame(index=data.index)
for f in feat_cat:
    newdata[f+'_freq']=data.groupby(f)[f].transform(lambda x: len(x))/data.shape[0]

newdata.to_csv('../feat/improve.csv',index=False)


