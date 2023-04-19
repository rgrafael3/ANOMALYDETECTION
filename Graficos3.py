# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 23:26:59 2023

@author: rgraf
"""
# Common imports
import numpy as np # numpy is THE toolbox for scientific computing with python
import pandas as pd # pandas provides THE data structure and data analysis tools for data scientists 
from sklearn.svm import SVC


# maximum number of columns
pd.set_option("display.max_rows", 101)
pd.set_option("display.max_columns", 101)

# To plot pretty figures
#matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

from sklearn import datasets

df_iris         = pd.DataFrame(datasets.load_iris().data)
df_iris.columns = datasets.load_iris().feature_names
y_iris          = datasets.load_iris().target
# bring target variable into dataframe:
df_iris.insert(df_iris.shape[1], 'target', y_iris, True)

# for later use:
cols = ['petal length (cm)', 'petal width (cm)']
X    = np.array(df_iris[cols]) #feature matrix
y    = df_iris['target'].values # target

from sklearn.svm import SVC#, LinearSVC 
# LinearSVC: faster
# SVC: polynomial....

# SVM Classifier model
svm_clf = SVC(random_state=0,C=0.001,max_iter=1000000000,kernel="linear") # C regularization Regularization = 1/alpha
# LinearSVC()
svm_clf.fit(X, y)

def plot_svc_decision_boundary(svm_clf, xmin, xmax):
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]

    x0 = np.linspace(xmin, xmax, 200)
    # see above formula!
    decision_boundary = -w[0]/w[1] * x0 - b/w[1]

    margin = 1/w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin

    svs = svm_clf.support_vectors_
    plt.scatter(svs[:, 0], svs[:, 1])
    plt.plot(x0, decision_boundary, "k-", linewidth=2)
    plt.plot(x0, gutter_up, "k--", linewidth=2)
    plt.plot(x0, gutter_down, "k--", linewidth=2)
   



from warnings import filterwarnings
from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X = scaler.fit_transform(X)
svm_clf = SVC(kernel="linear", C=10**9)
svm_clf.fit(X, y)

plt.figure(figsize=(12,7.2))
plot_svc_decision_boundary(svm_clf, 0, 5.5)
plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs")
plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo")
plt.xlabel("X", fontsize=14)
plt.ylabel("Y", fontsize=14)
plt.axis([0,5.5,0,2])
plt.show() 