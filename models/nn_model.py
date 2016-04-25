#!/usr/bin/python

from sknn.mlp import Classifier, Layer
import pandas as pd

train = pd.read_csv('../data/modeltrain.csv',index_col=0)
test = pd.read_csv('../data/modeltest.csv',index_col=0)
label = train['Response'].values

feat = train.columns.drop('Response',1)

nn = Classifier(
    layers=[
        Layer("Rectifier",units=10),
        Layer("Softmax")],
    learning_rate=0.001,
    n_iter=25)
    
nn.fit(train[feat].values, label)

from sklearn.metrics import log_loss
log_loss(label,nn.predict_proba(train[feat].values)[:,1])