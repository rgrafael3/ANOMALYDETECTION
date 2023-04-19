# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 23:29:44 2023

@author: rgraf
"""

from sklearn.utils import shuffle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
from sklearn.semi_supervised import LabelPropagation
import pomegranate as pg



def label_prop_test(kernel, params_list, X_train, X_test, y_train, y_test):
    plt.figure(figsize=(20,10))
    n, g = 0, 0
    roc_scores = []
    if kernel == 'rbf':
        for g in params_list:
            lp = LabelPropagation(kernel=kernel, n_neighbors=n, gamma=g, max_iter=100000000, tol=0.0001)
            lp.fit(X_train, y_train)
            roc_scores.append(roc_auc_score(y_test, lp.predict_proba(X_test)[:,1]))
    if kernel == 'knn':
        for n in params_list:
            lp = LabelPropagation(kernel=kernel, n_neighbors=n, gamma=g, max_iter=10000000, tol=0.0001)
            lp.fit(X_train, y_train)
            roc_scores.append(roc_auc_score(y_test, lp.predict_proba(X_test)[:,1]))
    plt.figure(figsize=(16,8));
    plt.plot(params_list, roc_scores)
    print (roc_scores)
    plt.title('Label Propagation ROC AUC with ' + kernel + ' kernel')
    print('Best metrics value is at {}'.format(params_list[np.argmax(roc_scores)]))
    plt.show()
    


X_trainNormal = pd.read_csv("C://THESIS//DATA//resultadosmicroservice_good.csv",sep=",", header=0)
X_trainNotNormal = pd.read_csv("C://THESIS//DATA//resultadosmicroservice_bad.csv",sep=",", header=0)


#droproc name of transaction and categorical vaule tran result
X_trainNormal=X_trainNormal.drop(['resultv','name'], axis=1)
X_trainNotNormal=X_trainNotNormal.drop(['resultv','name'], axis=1)


AllX=pd.concat([X_trainNormal, X_trainNotNormal])
AllX=pd.DataFrame(preprocessing.StandardScaler().fit(AllX).transform(AllX))






X_trainNormal["y"]=0
X_trainNotNormal["y"]=1




Ally=pd.concat([X_trainNormal['y'], X_trainNotNormal['y']])


AllXY=pd.concat([AllX,Ally.reset_index()],axis=1,ignore_index=True)




training_dataXY, testing_dataXY = train_test_split(AllXY, test_size=0.2, random_state=25)


AllXY=shuffle(AllXY)



normalisedTrainingDataX=training_dataXY.drop([15], axis=1)

X=AllXY.drop([15], axis=1)
y=AllXY.loc[:,15]



X_1, X_2, X_3  = np.split(X, [int(.2*len(X)), int(.7*len(X))])
y_1, y_2, y_3  = np.split(y, [int(.2*len(y)), int(.7*len(y))])
y_1_2 = np.concatenate((y_1, y_2.apply(lambda x: -1)))
X_1_2 = np.concatenate((X_1, X_2))

index = ['Algorithm', 'ROC AUC']
results = pd.DataFrame(columns=index)
logreg = LogisticRegression(random_state=1, class_weight='balanced')
logreg.fit(X_1, y_1)
results = results.append(pd.Series(['Logistic Regression', roc_auc_score(y_3, logreg.predict_proba(X_3)[:,1])], 
                                   index=index), ignore_index=True)
#print (results)

gammas = [9e-6, 1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5,1e-3,1e-1,0.3,0.5,1
          ]
label_prop_test('rbf', gammas, X_1_2, X_3, y_1_2, y_3)

ns = np.arange(50,500)
#label_prop_test('knn', ns, X_1_2, X_3, y_1_2, y_3)





nb = pg.NaiveBayes.from_samples(pg.NormalDistribution, X_1_2, y_1_2, verbose=True)
print (X_3.shape)



results = results.append(pd.Series(['Naive Bayes ICD Prior', 
                                   roc_auc_score(y_3, nb.predict_proba(X_3)[:,1])], index=index), ignore_index=True)

print (results)