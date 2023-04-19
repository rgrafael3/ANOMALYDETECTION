# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 16:24:40 2023

@author: rgraf
"""




# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay

X_trainNormal = pd.read_csv("C://THESIS//DATA//resultadosmicroservice_good.csv",sep=",", header=0)
X_trainNotNormal = pd.read_csv("C://THESIS//DATA//resultadosmicroservice_bad.csv",sep=",", header=0)

AllX_train=pd.concat([X_trainNormal, X_trainNotNormal])
AllX_train.head()

X_train=AllX_train[["name","resultv","latency","inlen","outlen","mem","tprocess2","tprocess3","tkafka","kafkalen","tdatabase","resulsetlen","numthread","tranrate","loadcpu","freemem"]]
X_trainNormal["y"]=0
X_trainNotNormal["y"]=1

XY_train=pd.concat([X_trainNormal, X_trainNotNormal])

y_train=XY_train["y"]

mmx_scaler = MinMaxScaler()

# Create scaled datasets 
# X_train_M = pd.DataFrame(mmx_scaler.fit_transform(X_train))
# X_test_M  = pd.DataFrame(mmx_scaler.transform(X_train))

#X_train_M = pd.DataFrame(preprocessing.normalize(X_train.values))
#X_test_M  = pd.DataFrame(preprocessing.normalize(X_train.values))

X_train_M = pd.DataFrame(preprocessing.StandardScaler().fit(X_train).transform(X_train))
X_test_M  = pd.DataFrame(preprocessing.StandardScaler().fit(X_train).transform(X_train))


print(X_test_M.head())

#if 1==1:
#   exit() 

from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.under_sampling import RandomUnderSampler,NearMiss
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold 

# Specify different values for the tunning process
StratifiedKFold = RepeatedStratifiedKFold(n_splits     = 5, 
                                          n_repeats    = 2, 
                                          random_state = 99)

base_estimator     = [LogisticRegression(), DecisionTreeClassifier(), SVC()]
n_estimators       = [5, 10, 50]
bootstrap          = [True, False]
warm_start         = [True, False]
sampling_strategy  = ['auto']
sampler            = [RandomUnderSampler(), NearMiss()]

#Create parameter grid
balanced_grid = [{'estimator'    : base_estimator,
                  'n_estimators'      : n_estimators, 
                  'bootstrap'         : bootstrap,
                  'warm_start'        : warm_start, 
                  'sampling_strategy' : sampling_strategy,
                  'sampler'           : sampler}]

#Create Balanced Bagging object
balanced_model  = BalancedBaggingClassifier()

#Grid Search CV
#balanced_search = GridSearchCV(balanced_model, 
#                           balanced_grid, 
#                           scoring= 'f1', 
#                           cv = StratifiedKFold, 
#                           verbose= True).fit(X_train_M, y_train)

#balanced_search.best_params_

from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.under_sampling import RandomUnderSampler,NearMiss
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


models = []

models.append(('Logistic Regression weak-learner',BalancedBaggingClassifier(base_estimator = LogisticRegression(),
                                          bootstrap      = False,
                                          n_estimators   = 1,
                                          sampler        = RandomUnderSampler(),
                                          sampling_strategy = 'auto',
                                          warm_start        = False)))
models.append(('Decision Tree weak-learner',BalancedBaggingClassifier(base_estimator = DecisionTreeClassifier(),
                                          bootstrap      = True,
                                          n_estimators   = 1,
                                          sampler        = RandomUnderSampler(),
                                          sampling_strategy = 'auto',
                                          warm_start        = False)))
models.append( ('Support Vector Classifier weak-learner',BalancedBaggingClassifier(base_estimator = SVC(),
                                          bootstrap      = True,
                                          n_estimators   = 1,
                                          sampler        = RandomUnderSampler(),
                                          sampling_strategy = 'auto',
                                          warm_start        = False)))
import time
start_time = time.time()

for name, model in models:
    fitting  = model.fit(X_train_M, y_train)
    y_pred   = fitting.predict(X_test_M)
    accuracy = accuracy_score(y_train, y_pred) 
    svc_disp = RocCurveDisplay.from_estimator(fitting, X_train_M, y_train)
    plt.show(name)
    print('Accuracy: ',round(accuracy, 5), name,"--- %s seconds ---" % (time.time() - start_time) )
    
for name, model in models:
    fitting   = model.fit(X_train_M, y_train)
    y_pred    = fitting.predict(X_test_M)
    precision = precision_score(y_train, y_pred)   
    print('Precision: ',round(precision, 5), name,"--- %s seconds ---" % (time.time() - start_time) )
    
for name, model in models:
    fitting   = model.fit(X_train_M, y_train)
    y_pred    = fitting.predict(X_test_M)
    recall    = recall_score(y_train, y_pred)   
    print('Sensitivity: ',round(recall, 5), name,"--- %s seconds ---" % (time.time() - start_time) )
    
for name, model in models:
    fitting    = model.fit(X_train_M, y_train)
    y_pred     = fitting.predict(X_test_M)
    pre_recall = f1_score(y_train, y_pred)   
    print('F1-Score: ',round(pre_recall, 5), name,"--- %s seconds ---" % (time.time() - start_time) )


