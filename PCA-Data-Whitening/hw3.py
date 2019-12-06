#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 00:24:51 2018

@author: rossspencer
"""

import numpy as np
import matplotlib.pyplot as plt

#generate dataset and plot
mean = [1,100] #mean of class 1
cov  = [[13,-2], [-2,.001]] #diagonal covariance
numPnt = 100 #number of points for class1
data = np.random.multivariate_normal(mean, cov, numPnt) #generate points for class 1
fig = plt.figure()
plt.scatter(data[:,0],data[:,1])
plt.xlim(min(data[:,0]) - 1, max(data[:,0]) + 1)
plt.ylim(min(data[:,1]) - 1, max(data[:,1]) + 1)
plt.show()


'''
whiten : bool, optional (default False)
When True (False by default) the components_ vectors are multiplied by the square root of n_samples and then divided by the singular values to ensure uncorrelated outputs with unit component-wise variances.
Whitening will remove some information from the transformed signal (the relative variance scales of the components) but can sometime improve the predictive accuracy of the downstream estimators by making their data respect some hard-wired assumptions.
'''
from sklearn.decomposition import PCA

pca = PCA(whiten=False)
data_nowhite = pca.fit_transform(data)
fig = plt.figure()
plt.xlim(min(data_nowhite[:,0]) - 1, max(data_nowhite[:,0]) + 1)
plt.ylim(min(data_nowhite[:,1]) - 1, max(data_nowhite[:,1]) + 1)
plt.scatter(data_nowhite[:,0],data_nowhite[:,1])
plt.show()
pca.get_covariance()
np.cov(data_nowhite,rowvar=False)

pca = PCA(whiten=True)
data_white = pca.fit_transform(data)
fig = plt.figure()
plt.xlim(min(data_white[:,0]) - 1, max(data_white[:,0]) + 1)
plt.ylim(min(data_white[:,1]) - 1, max(data_white[:,1]) + 1)
plt.scatter(data_white[:,0],data_white[:,1])
plt.show()
pca.get_covariance()
np.cov(data_white,rowvar=False)


#x = [[15,4], [32,2]]
#np.linalg.svd(x)[1] #eigenvalues
#np.linalg.svd(x)[0] #eigenvectors
#np.dot(35.5159986,x)
#U, s, Vt = np.linalg.svd(x, full_matrices=False)
#X_white = np.dot(U, Vt)
#np.dot(X_white, X_white)

##found time to implement PCA whitening without relying upon scikit-learn
#### many thanks to ali_m on StackOverflow for helping me fix an issue with my covariance
def PCA_whitening(X):
   #first subtract the mean to center columns at 0
   M = np.mean(X, axis=0)
   X_centered = X - M
   #find the covariance matrix
   Xcov = np.cov(X_centered, rowvar = False)
   #get the set of eigenvalues aka weights w and the set of eigenvectors v
   # eigh returns:	  w : (…, M) ndarray -- The eigenvalues in ascending order, each repeated according to its multiplicity.
   #            v : {(…, M, M) ndarray, (…, M, M) matrix} -- The column v[:, i] is the normalized eigenvector corresponding to the eigenvalue w[i]. 
   w, v = np.linalg.eigh(Xcov)
   #put the eigenvalues in a diagonal matrix
   L = np.diag(1 / np.sqrt(w))
   #find VLV'
   U = np.dot(np.dot(v, L), v.T)
   #find XVDV' to get our final whitened dataset
   X_new = np.dot(X_centered, U)

   return X_new


data_whitened = PCA_whitening(data)
fig = plt.figure()
plt.scatter(data_whitened[:,0],data_whitened[:,1])
plt.xlim(min(data_whitened[:,0]) - 1, max(data_whitened[:,0]) + 1)
plt.ylim(min(data_whitened[:,1]) - 1, max(data_whitened[:,1]) + 1)
plt.show()
