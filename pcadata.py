# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 22:43:38 2023

@author: rgraf
"""

from sklearn.decomposition import PCA


import numpy
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sn


X_trainNormal = pd.read_csv("C://THESIS//DATA//resultadosmicroservice_good.csv",sep=",", header=0)
X_trainNotNormal = pd.read_csv("C://THESIS//DATA//resultadosmicroservice_bad.csv",sep=",", header=0)

AllX_train=pd.concat([X_trainNormal, X_trainNotNormal])
AllX_train.head()

#X_train=AllX_train[["name","resultv","latency","inlen","outlen","mem","tprocess2","tprocess3","tkafka","kafkalen","tdatabase","resulsetlen","numthread","tranrate","loadcpu","freemem"]]
X_train=AllX_train[["latency","inlen","outlen","mem","tprocess2","tprocess3","tkafka","kafkalen","tdatabase","resulsetlen","numthread","tranrate","loadcpu","freemem"]]
X_trainNormal["y"]=0
X_trainNotNormal["y"]=1

XY_train=pd.concat([X_trainNormal, X_trainNotNormal])

print (XY_train.tail())

y_train=XY_train["y"]

print(y_train.head())

print (y_train.shape)

mmx_scaler = MinMaxScaler()

X_train_M = pd.DataFrame(preprocessing.StandardScaler().fit(X_train).transform(X_train))

pca_data = PCA(n_components=2)


principal_data=pca_data.fit_transform(X_train_M)

principal_data_Df = pd.DataFrame(data = principal_data
             , columns = ['principalcomponent1', 'principalcomponent2'])


corrMatrix=X_train_M.corr()
print(corrMatrix)

sn.heatmap(corrMatrix, annot=True)
plt.show()

sn.pairplot(X_train_M)

plt.show()

print(principal_data_Df.tail())
print(y_train.tail())
print (principal_data_Df.tail())
print (principal_data_Df.shape)

print('Explained variation per principal component: {}'.format(pca_data.explained_variance_ratio_))

plt.figure()
plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Principal Component - 1',fontsize=20)
plt.ylabel('Principal Component - 2',fontsize=20)
plt.title("Principal Component Analysis of Metrics data ",fontsize=20)
targets = [1, 0]
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = XY_train['y'] == target
    principal_data_Df=principal_data_Df.reset_index(drop=True)
    indicesToKeep=indicesToKeep.reset_index(drop=True)
    print(indicesToKeep.shape)
    print (indicesToKeep.head())
    print(pd.concat([principal_data_Df, indicesToKeep], axis=1).tail())
    print(principal_data_Df.loc[indicesToKeep,'principalcomponent1'].tail())
    plt.scatter(principal_data_Df.loc[indicesToKeep,'principalcomponent1']
               , principal_data_Df.loc[indicesToKeep,'principalcomponent2'], c = color, s = 50)

