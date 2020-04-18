# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 20:26:16 2018

@author: HL
"""

#import matplotlib.pyplot as plt
import numpy as np
from bayesian_optimization import BayesianOptimization
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
import time

train = pd.read_csv('loan/rftrain.csv')
train_label = train.loc[:,'label']
train = train.drop(['Unnamed: 0','label','emp_length','earliest_cr_line'],axis=1)

test = pd.read_csv('loan/rftest.csv')
test_label = test.loc[:,'label']
test = test.drop(['Unnamed: 0','label','emp_length','earliest_cr_line'],axis=1)

#train.head()
#train.info()
#train = train.drop('emp_length',axis=1)
#train.loc[train.loc[:,'emp_length'] == '< 1 year','emp_length'] = '0 year'

le = LabelEncoder()
catogory_cols = ['initial_list_status','addr_state','purpose','verification_status','grade','term']
for col in catogory_cols:
    train.loc[:,col]= train.loc[:,col].astype('category')
    test.loc[:,col]=test.loc[:,col].astype('category')
    train.loc[:,col] = le.fit_transform(train.loc[:,col])
    test.loc[:,col] = le.transform(test.loc[:,col])


#X_train, X_dev, Y_train, Y_dev= train_test_split(train, train_label, test_size=0.1)

kf = KFold(n_splits=5)
kf.get_n_splits(train)


start = time.time()
print('start fitting rfc....')
loss=0 #1.6468912720703779
for train_index, test_index in kf.split(train):
    print(np.shape(train_index), np.shape(test_index))
    X_train, X_test = train.iloc[train_index,:], train.iloc[test_index,:]
    y_train, y_test = train_label.iloc[train_index], train_label.iloc[test_index]
    rfc = RandomForestClassifier(n_estimators=400, max_features=7, n_jobs=-1)
    rfc.fit(X_train, y_train)
    loss += log_loss(y_test,rfc.predict_proba(X_test)[:,1])   
end = time.time()
rfc_runtime = (end-start)/60
print('rfc fitting time: {:.2f}'.format(rfc_runtime))
#print('rfc predicting...')



''' =============== Bayesian Optimization =============== '''

def target_func(n_estimators, max_features):
    n_estimators = int(np.exp(n_estimators))
    loss = 0
    for train_index, test_index in kf.split(train):
        X_train, X_test = train.iloc[train_index,:], train.iloc[test_index,:]
        y_train, y_test = train_label.iloc[train_index], train_label.iloc[test_index]
        rf = RandomForestClassifier(n_estimators=int(n_estimators), max_features=min(max_features, 0.999), n_jobs=-1)
        rf.fit(X_train, y_train)
        loss += log_loss(y_test,rf.predict_proba(X_test)[:,1])   
    return -loss

rfcBO= BayesianOptimization(target_func,{'n_estimators':[np.log(200),np.log(800)],
                                          'max_features':(0.1,0.3)})
start= time.time()
rfcBO.maximize(init_points=2, acq='ei',n_iter=5)
end=time.time()
rfBO_runtime = (end-start)/60
print('rfBO runtime: {:.2f}'.format(rfBO_runtime))

print('RFC: %f' % rfcBO.res['max']['max_val'])

rf = RandomForestClassifier(n_estimators=800, max_features=0.1, n_jobs=-1)
rf.fit(train, train_label)
loss_bo = log_loss(test_label, rf.predict_proba(test)[:,1])


