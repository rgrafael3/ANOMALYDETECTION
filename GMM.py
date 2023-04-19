# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 22:43:38 2023

@author: rgraf
"""

from sklearn.decomposition import PCA


import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.cluster import KMeans
import matplotlib as mpl

from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
from matplotlib.colors import LogNorm

def plot_gaussian_mixture(clusterer, X, resolution=1000, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = -clusterer.score_samples(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z,
                 norm=LogNorm(vmin=1.0, vmax=30.0),
                 levels=np.logspace(0, 2, 12))
    plt.contour(xx, yy, Z,
                norm=LogNorm(vmin=1.0, vmax=30.0),
                levels=np.logspace(0, 2, 12),
                linewidths=1, colors='k')

    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z,
                linewidths=2, colors='r', linestyles='dashed')
    
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)
    plot_centroids(clusterer.means_, clusterer.weights_)

    plt.xlabel("$x_1$", fontsize=14)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)

def plot_clusters_coloured(principal_data_Df, XY_train):
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
                   , principal_data_Df.loc[indicesToKeep,'principalcomponent2'], c = color, s = 2)




def plot_clusters(X, y=None):
    plt.scatter(X[:, 0], X[:, 1], s=2)
    plt.xlabel("$x_1$", fontsize=14)
    plt.ylabel("$x_2$", fontsize=14, rotation=0)
    plt.title('Artificial dataset created with make_blobs')

def plot_data(X):
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)

def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=13, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=15, linewidths=15,
                color=cross_color, zorder=11, alpha=1)

def plot_decision_boundaries(clusterer, X, XY_train, resolution=1000, show_centroids=True,
                             show_xlabels=True, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    
    
    
    
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')
    plot_clusters_coloured(X,XY_train)
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)

    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)
    
X_trainNormal = pd.read_csv("C://THESIS//DATA//resultadosmicroservice_good.csv",sep=",", header=0)
X_trainNotNormal = pd.read_csv("C://THESIS//DATA//resultadosmicroservice_bad.csv",sep=",", header=0)

AllX_train=pd.concat([X_trainNormal, X_trainNotNormal])
#AllX_train.head()

X_train=AllX_train[["latency","inlen","outlen","mem","tprocess2","tprocess3","tkafka","kafkalen","tdatabase","resulsetlen","numthread","tranrate","loadcpu","freemem"]]
#X_train=X_trainNormal[["latency","inlen","outlen","mem","tprocess2","tprocess3","tkafka","kafkalen","tdatabase","resulsetlen","numthread","tranrate","loadcpu","freemem"]]
X_trainNormal["y"]=0
X_trainNotNormal["y"]=1

XY_train=pd.concat([X_trainNormal, X_trainNotNormal])



#y_train=XY_train["y"]



mmx_scaler = MinMaxScaler()

X_train_M = pd.DataFrame(preprocessing.StandardScaler().fit(X_train).transform(X_train))


#db = DBSCAN(algorithm='auto', eps=0.15
#            , leaf_size=30, metric='euclidean',
#       metric_params=None, min_samples=10, n_jobs=None, p=None)

#db.fit(X_train_M)

#gmm = GaussianMixture(n_components = 5)
#gmm.fit(X_train_M)
#labels = gmm.predict(X_train_M)


pca_data = PCA(n_components=2)


principal_data=pca_data.fit_transform(X_train_M)

principal_data_Df = pd.DataFrame(data = principal_data
             , columns = ['principalcomponent1', 'principalcomponent2'])
principal_data_Df=principal_data_Df.reset_index(drop=True)

gmm = GaussianMixture(n_components = 4)
gmm.fit(principal_data_Df)
labels = gmm.predict(principal_data_Df)


print (principal_data_Df)


print('Converged:',gmm.converged_)
print(gmm.n_iter_)
means = gmm.means_
covariances = gmm.covariances_

print (means)
#print (covariances)

principal_data_Df['labels']=labels

print (labels)
print (labels.size)

d0 = principal_data_Df[principal_data_Df['labels']== 0]
d1 = principal_data_Df[principal_data_Df['labels']== 1]
d2 = principal_data_Df[principal_data_Df['labels']== 2]
d3 = principal_data_Df[principal_data_Df['labels']== 3]
d4 = principal_data_Df[principal_data_Df['labels']== 4]

plt.scatter(d0.iloc[:,0], d0.iloc[:,1], c ='r')
plt.scatter(d1.iloc[:,0], d1.iloc[:,1], c ='g')
plt.scatter(d2.iloc[:,0], d2.iloc[:,1], c ='b')
plt.scatter(d3.iloc[:,0], d3.iloc[:,1], c ='k')
plt.scatter(d4.iloc[:,0], d4.iloc[:,1], c ='y')
plt.show()

plt.figure(figsize=(12, 4))
plot_gaussian_mixture(gmm, principal_data_Df.to_numpy())
plt.show()

densities = gmm.score_samples(principal_data_Df.drop('labels', axis=1).to_numpy())
density_threshold = np.percentile(densities, 5)
print('Density threshold at 4%: ', density_threshold)
anomalies = principal_data_Df.to_numpy()[densities < density_threshold]

plt.figure(figsize=(12, 4))

plot_gaussian_mixture(gmm, principal_data_Df.drop('labels', axis=1).to_numpy())
plt.scatter(anomalies[:, 0], anomalies[:, 1], color='r', marker='*')
plt.ylim(top=2.1)
plt.show()

gms_per_k = [GaussianMixture(n_components=k, n_init=10, random_state=42).fit(principal_data_Df.drop('labels', axis=1).to_numpy())for k in range(1, 20)]
bics = [model.bic(principal_data_Df.drop('labels', axis=1).to_numpy()) for model in gms_per_k]
aics = [model.aic(principal_data_Df.drop('labels', axis=1).to_numpy()) for model in gms_per_k]

plt.figure(figsize=(12, 3))
plt.plot(range(1, 20), bics, "bo-", label="BIC")
plt.plot(range(1, 20), aics, "go--", label="AIC")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Information Criterion", fontsize=14)
plt.axis([1, 19.5, np.min(aics) - 50, np.max(aics) + 50])
#plt.annotate('Minimum',
#             xy=(4, bics[3]),
#             xytext=(0.35, 0.6),
#             textcoords='figure fraction',
#             fontsize=14,
#             arrowprops=dict(facecolor='black', shrink=0.1)
#           )
plt.legend()
plt.show()