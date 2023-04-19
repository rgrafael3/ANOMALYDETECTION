# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 17:59:26 2023

@author: rgraf
"""

import numpy
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 

X_trainNormal = pd.read_csv("C://THESIS//DATA//resultadosmicroservice_good.csv",sep=",", header=0)
X_trainNotNormal = pd.read_csv("C://THESIS//DATA//resultadosmicroservice_bad.csv",sep=",", header=0)


#drop name of transaction and categorical vaule tran result
X_trainNormal=X_trainNormal.drop(['resultv','name'], axis=1)
X_trainNotNormal=X_trainNotNormal.drop(['resultv','name'], axis=1)


AllX=pd.concat([X_trainNormal, X_trainNotNormal])
AllX=pd.DataFrame(preprocessing.StandardScaler().fit(AllX).transform(AllX))






X_trainNormal["y"]=0
X_trainNotNormal["y"]=1




Ally=pd.concat([X_trainNormal['y'], X_trainNotNormal['y']])


AllXY=pd.concat([AllX,Ally.reset_index()],axis=1,ignore_index=True)




training_dataXY, testing_dataXY = train_test_split(AllXY, test_size=0.2, random_state=25)



normalisedTrainingDataX=training_dataXY.drop([15], axis=1)



fitting = LogisticRegression(random_state=0,C=100,solver='saga',max_iter=1000000000).fit(normalisedTrainingDataX, training_dataXY[15])



y_pred = fitting.predict(testing_dataXY.drop([15], axis=1))
y=testing_dataXY[15]


accuracy=accuracy_score(y, y_pred)
precission=precision_score(y, y_pred)
f1score=f1_score(y, y_pred)
recall=recall_score(y, y_pred)
print ("Accuracy on test",accuracy)
print ("Recall on test", recall)
print ("Precission on test" , precission)
print ("F1 score",f1score)