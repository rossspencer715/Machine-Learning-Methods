# -*- coding: utf-8 -*-
"""
File:   hw02.py
Author: Ross Spencer
Date:   09/22/2018
Desc:   Homework 2 Source Code
    
"""


""" =======================  Import dependencies ========================== """

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


plt.close('all') #close any open plots

""" =======================  Import DataSet ========================== """


Train_2D = np.loadtxt('2dDataSetforTrain.txt')
Train_7D = np.loadtxt('7dDataSetforTrain.txt')
Train_HS = np.loadtxt('HyperSpectralDataSetforTrain.txt')

labels_2D = Train_2D[:,Train_2D.shape[1]-1]
labels_7D = Train_7D[:,Train_7D.shape[1]-1]
labels_HS = Train_HS[:,Train_HS.shape[1]-1]

Train_2D = np.delete(Train_2D,Train_2D.shape[1]-1,axis = 1)
Train_7D = np.delete(Train_7D,Train_7D.shape[1]-1,axis = 1)
Train_HS = np.delete(Train_HS,Train_HS.shape[1]-1,axis = 1)

Test_2D = np.loadtxt('2dDataSetforTest.txt')
Test_7D = np.loadtxt('7dDataSetforTest.txt')
Test_HS = np.loadtxt('HyperSpectralDataSetforTest.txt')

""" ======================  Function definitions ========================== """


"""
===============================================================================
===============================================================================
======================== Probabilistic Generative Classfier ===================
===============================================================================
===============================================================================
"""

Train = Train_7D
labels = labels_7D
Classes = np.sort(np.unique(labels))
X_train_class = []
    
#sets random state then splits our data so 33% of it is saved for validation
M = 20
X_train, X_valid, label_train, label_valid = train_test_split(Train, labels, test_size = 0.33, random_state = M)

#breaks our training data into classes so that the ith class is in X_train_class[i-1]
for j in range(Classes.shape[0]):
    jth_class = X_train[label_train == Classes[j],:]
    X_train_class.append(jth_class)
    
#calculates the estimators for mu, full cov, and probability of each class
mu1 = np.mean(X_train_class[0], axis=0)
mu2 = np.mean(X_train_class[1], axis=0)
cov1 = np.cov(X_train_class[0].T)
cov2 = np.cov(X_train_class[1].T)
pC1 = len(X_train_class[0])/(len(X_train_class[0]) + len(X_train_class[1]))
pC2 = len(X_train_class[1])/(len(X_train_class[0]) + len(X_train_class[1]))


#used to generate textfile w the cov1 of 7D in state M=21
#np.savetxt('cov1_7D.txt', cov1)


#prints data in the 2D case to visualize what's happening, credit to lecture notes
#fig = plt.figure()
#ax = fig.add_subplot(*[1,1,1])
#ax.scatter(X_train_class[0][:,0], X_train_class[0][:,1], c='r') 
#ax.scatter(X_train_class[1][:,0], X_train_class[1][:,1], c='b')  
##ax.scatter(Train_2D[:,0], Train_2D[:,1]) 
#plt.show()

X = X_valid
#finds the partial density function value at each Xi in our validation set for each class' estimated parameters
y1 = multivariate_normal.pdf(X, mean=mu1, cov=cov1, allow_singular=True); 
y2 = multivariate_normal.pdf(X, mean=mu2, cov=cov2, allow_singular=True); 

#calculates the posterior for each class
pos1 = (y1*pC1)/(y1*pC1 + y2*pC2);
pos2 = (y2*pC2)/(y1*pC1 + y2*pC2);

#predicts based off which posterior probability is larger, if pos1<pos2 is true it should be class 1 and if it's false it should be class 0
predictions_PG = pos1<pos2
#converts from boolean values to 1s and 0s
predictions_PG = np.where(predictions_PG, 1, 0)

''' diagonal: '''

#calculates diagonal cov matrix assuming the different variables are independent and thus have cov=0
numVars = (int)(len(X_train_class[0].T))
cov1 = np.zeros((numVars, numVars))
cov2 = np.zeros((numVars, numVars))
for i in range(numVars):    #cov(x,x) = var(x)
    cov1[i][i] = np.var(X_train_class[0][:,i])      #use s*(n-1)/n aka regular population variance for MLE
    cov2[i][i] = np.var(X_train_class[1][:,i])

#finds the partial density function value at each Xi in our validation set taking diagonal cov in account
y1 = multivariate_normal.pdf(X, mean=mu1, cov=cov1, allow_singular=True); 
y2 = multivariate_normal.pdf(X, mean=mu2, cov=cov2, allow_singular=True); 

#calculates the posterior with the new y values of the pdfs and predicts based off the larger
pos1 = (y1*pC1)/(y1*pC1 + y2*pC2);
pos2 = (y2*pC2)/(y1*pC1 + y2*pC2);
predictions_PGdiag = pos1<pos2
predictions_PGdiag = np.where(predictions_PGdiag, 1, 0)




''' 
=================================================================================================================
special case: for hyperspectral data since my old method was hard-coded to only take 2 classes into consideration 
'''

# reset our train and label variables
Train = Train_HS
labels = labels_HS
Classes = np.sort(np.unique(labels))
X_train_class = []
    
#sets random state then splits our data so 33% of it is saved for validation
M = 20
X_train, X_valid, label_train, label_valid = train_test_split(Train, labels, test_size = 0.33, random_state = M)

#breaks our training data into classes so that the ith class is in X_train_class[i-1]
for j in range(Classes.shape[0]):
    jth_class = X_train[label_train == Classes[j],:]
    X_train_class.append(jth_class)
    
    
#calculates the estimators for mu, full cov, and probability of each class
mu1 = np.mean(X_train_class[0], axis=0)
mu2 = np.mean(X_train_class[1], axis=0)
mu3 = np.mean(X_train_class[2], axis=0)
mu4 = np.mean(X_train_class[3], axis=0)
mu5 = np.mean(X_train_class[4], axis=0)

cov1 = np.cov(X_train_class[0].T)
cov2 = np.cov(X_train_class[1].T)
cov3 = np.cov(X_train_class[2].T)
cov4 = np.cov(X_train_class[3].T)
cov5 = np.cov(X_train_class[4].T)

pC1 = len(X_train_class[0])/(len(X_train))
pC2 = len(X_train_class[1])/(len(X_train))
pC3 = len(X_train_class[2])/(len(X_train))
pC4 = len(X_train_class[3])/(len(X_train))
pC5 = len(X_train_class[4])/(len(X_train))

#finds the partial density function value at each Xi in our validation set for each class' estimated parameters
X = X_valid
y1 = multivariate_normal.pdf(X, mean=mu1, cov=cov1, allow_singular=True); 
y2 = multivariate_normal.pdf(X, mean=mu2, cov=cov2, allow_singular=True); 
y3 = multivariate_normal.pdf(X, mean=mu3, cov=cov3, allow_singular=True); 
y4 = multivariate_normal.pdf(X, mean=mu4, cov=cov4, allow_singular=True); 
y5 = multivariate_normal.pdf(X, mean=mu5, cov=cov5, allow_singular=True); 

#calculates the posterior for each class
pos1 = (y1*pC1)/(y1*pC1 + y2*pC2 + y3*pC3 + y4*pC4 + y5*pC5);
pos2 = (y2*pC2)/(y1*pC1 + y2*pC2 + y3*pC3 + y4*pC4 + y5*pC5);
pos3 = (y3*pC3)/(y1*pC1 + y2*pC2 + y3*pC3 + y4*pC4 + y5*pC5);
pos4 = (y4*pC4)/(y1*pC1 + y2*pC2 + y3*pC3 + y4*pC4 + y5*pC5);
pos5 = (y5*pC5)/(y1*pC1 + y2*pC2 + y3*pC3 + y4*pC4 + y5*pC5);

#predicts based off which posterior probability is largest, classification should be the "most likely" aka class w the max posterior
predictions_PG = []
for i in range(len(pos1)):
    temp = max(pos1[i], pos2[i], pos3[i], pos4[i], pos5[i])
    if temp == pos1[i]:
        predictions_PG.append(0)
    elif temp == pos2[i]:
        predictions_PG.append(1)
    elif temp == pos3[i]:
        predictions_PG.append(2)
    elif temp == pos4[i]:
        predictions_PG.append(3)
    else:
        predictions_PG.append(4)

''' diagonal: '''


#calculates diagonal cov matrix assuming the different variables are independent and thus have cov=0
numVars = (int)(len(X_train_class[0].T))
cov1 = np.zeros((numVars, numVars))
cov2 = np.zeros((numVars, numVars))
cov3 = np.zeros((numVars, numVars))
cov4 = np.zeros((numVars, numVars))
cov5 = np.zeros((numVars, numVars))
for i in range(numVars):    #cov(x,x) = var(x)
    cov1[i][i] = np.var(X_train_class[0][:,i])      #use s*(n-1)/n aka regular population variance for MLE
    cov2[i][i] = np.var(X_train_class[1][:,i])
    cov3[i][i] = np.var(X_train_class[1][:,i])
    cov4[i][i] = np.var(X_train_class[1][:,i])
    cov5[i][i] = np.var(X_train_class[1][:,i])
    
    
#finds the partial density function value at each Xi in our validation set taking diagonal cov in account
y1 = multivariate_normal.pdf(X, mean=mu1, cov=cov1, allow_singular=True); 
y2 = multivariate_normal.pdf(X, mean=mu2, cov=cov2, allow_singular=True); 
y3 = multivariate_normal.pdf(X, mean=mu3, cov=cov3, allow_singular=True); 
y4 = multivariate_normal.pdf(X, mean=mu4, cov=cov4, allow_singular=True); 
y5 = multivariate_normal.pdf(X, mean=mu5, cov=cov5, allow_singular=True); 

#calculates the posterior with the new y values of the pdfs
pos1 = (y1*pC1)/(y1*pC1 + y2*pC2 + y3*pC3 + y4*pC4 + y5*pC5);
pos2 = (y2*pC2)/(y1*pC1 + y2*pC2 + y3*pC3 + y4*pC4 + y5*pC5);
pos3 = (y3*pC3)/(y1*pC1 + y2*pC2 + y3*pC3 + y4*pC4 + y5*pC5);
pos4 = (y4*pC4)/(y1*pC1 + y2*pC2 + y3*pC3 + y4*pC4 + y5*pC5);
pos5 = (y5*pC5)/(y1*pC1 + y2*pC2 + y3*pC3 + y4*pC4 + y5*pC5);

#predicts based off which posterior probability is largest, classification should be the "most likely" aka class w the max posterior
predictions_PGdiag = []
for i in range(len(pos1)):
    temp = max(pos1[i], pos2[i], pos3[i], pos4[i], pos5[i])
    if temp == pos1[i]:
        predictions_PGdiag.append(0)
    elif temp == pos2[i]:
        predictions_PGdiag.append(1)
    elif temp == pos3[i]:
        predictions_PGdiag.append(2)
    elif temp == pos4[i]:
        predictions_PGdiag.append(3)
    else:
        predictions_PGdiag.append(4)




"""
===============================================================================
===============================================================================
============================ KNN Classifier ===================================
===============================================================================
===============================================================================
"""

""" Here you can write functions to achieve your KNN classifier. """
# KNN classifier credit to scikitlearn and Dr. Zare's lecture notes

#from matplotlib.colors import ListedColormap
from sklearn import neighbors

""" ============  Generate Training and validation Data =================== """

""" Here is an example for 2D DataSet, you can change it for 7D and HS 
    Also, you can change the random_state to get different validation data """

# Here you can change your data set
Train = Train_7D
labels = labels_7D
Classes = np.sort(np.unique(labels))

# Here you can change M to get different validation data
M = 15
X_train, X_valid, label_train, label_valid = train_test_split(Train, labels, test_size = 0.33, random_state = M)

# breaks data into classes where the ith class is in X_train_class[i-1]
X_train_class = []
for j in range(Classes.shape[0]):
    jth_class = X_train[label_train == Classes[j],:]
    X_train_class.append(jth_class)


#Visualization of first two dimension of the dataset for the 2D case
#for j in range(Classes.shape[0]):
#    plt.scatter(X_train_class[j][:,0],X_train_class[j][:,1])

""" ========================  Train the Classifier ======================== """

""" Here you can train your classifier with your training data """

# number of features = number cols in each row
# print(len(Train[0,:]))

n_neighbors = 300
clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
''' indented code, mostly straight from the lecture, allowed me to visualize how this classifier performed in the 2D case '''
            #classifiers.append(neighbors.KNeighborsClassifier(n_neighbors, weights='uniform'))
            #classifiers.append(neighbors.KNeighborsClassifier(n_neighbors, weights='distance'))
            #names = ['K-NN_Uniform']#, 'K-NN_Weighted']
            
            #Put together datasets
            #n_samples = len(X_train_class[1][:]) + len(X_train_class[0][:])
            #makes random data for a classification problem::
            #X, y = make_classification(n_samples, n_features=len(Train[0,:]), n_redundant=0, n_informative=len(Train[0,:]),
            #                           random_state=0, n_clusters_per_class=1)
            
            
            ###where we're at in the lecture code lol
            #plt.close('all')
            ##set up meshgrid for figure
            #x_min, x_max = Train[:, 0].min() - .5, Train[:, 0].max() + .5
            #y_min, y_max = Train[:, 1].min() - .5, Train[:, 1].max() + .5
            #xx, yy = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02))
            
            # just plot the dataset first
            #cm = plt.cm.RdBu
            #cm_bright = ListedColormap(['#FF0000', '#0000FF'])
            #ax = plt.subplot(1, 1, 1)
            ## Plot the training points
            #ax.scatter(X_train_class[1][:,0], X_train_class[1][:,1], c='b', cmap=cm_bright)
            #ax.scatter(X_train_class[0][:,0], X_train_class[0][:,1], c='r', cmap=cm_bright)
            ## and testing points
            ##ax.scatter(X_test[:, 0], X_test[:, 1], marker='+', c=y_test, cmap=cm_bright, alpha=0.6)
            #ax.set_xlim(xx.min(), xx.max())
            #ax.set_ylim(yy.min(), yy.max())
            #ax.set_xticks(())
            #ax.set_yticks(())
            
            ##figure.subplots_adjust(left=.02, right=.98)
            #plt.show()
            #
            #plt.close('all')
            
            
clf.fit(X_train, label_train)
# score = clf.score(X_valid, label_valid) #KNeighborsClassifier includes a scoring/accuracy method of its own
predictions_KNN = clf.predict(X_valid)



""" ======================== Cross Validation ============================= """


""" Here you should test your parameters with validation data """
#defined my own accuracy score before I realized it was already imported...
#def accuracy_score(label_valid, predictions):
#    accuracy = sum(label_valid == predictions)/len(predictions)
#    return accuracy


# The accuracy for your validation data
accuracy_PG = accuracy_score(label_valid, predictions_PG)
print('\nThe accuracy of Probabilistic Generative classifier is: ', accuracy_PG*100, '%')
accuracy_PGdiag = accuracy_score(label_valid, predictions_PGdiag)
print('\nThe accuracy of Probabilistic Generative (Diagonal) classifier is: ', accuracy_PGdiag*100, '%')
accuracy_KNN = accuracy_score(label_valid, predictions_KNN)
print('\nThe accuracy of KNN classifier is: ', accuracy_KNN*100, '%')

""" ========================  Test the Model ============================== """

""" This is where you should test the testing data with your classifier """

# Classifier of choice: KNN given its superiority in the 2D and Hyperspectral cases and the simplicity in my coded solution 
n_neighbors = 3
clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
clf.fit(Train_2D, labels_2D)
predictions_KNN = clf.predict(Test_2D)
np.savetxt('2DforTestLabels.txt', predictions_KNN)

# Classifier of choice: probabilistic generator since it did slightly better than KNN in cross validation for 7D
#breaks our entire training dataset into classes so that the ith class is in X_train_class[i-1]
Classes = np.sort(np.unique(labels_7D))
X_train_class = []
for j in range(Classes.shape[0]):
    jth_class = Train_7D[labels_7D == Classes[j],:]
    X_train_class.append(jth_class)
    
#calculates the estimators for mu, full cov, and probability of each class
mu1 = np.mean(X_train_class[0], axis=0)
mu2 = np.mean(X_train_class[1], axis=0)
cov1 = np.cov(X_train_class[0].T)
cov2 = np.cov(X_train_class[1].T)
pC1 = len(X_train_class[0])/(len(X_train_class[0]) + len(X_train_class[1]))
pC2 = len(X_train_class[1])/(len(X_train_class[0]) + len(X_train_class[1]))

#finds the pdf of our test data under our estimated params
y1 = multivariate_normal.pdf(Test_7D, mean=mu1, cov=cov1); 
y2 = multivariate_normal.pdf(Test_7D, mean=mu2, cov=cov2); 

#calculates the posterior for each class
pos1 = (y1*pC1)/(y1*pC1 + y2*pC2);
pos2 = (y2*pC2)/(y1*pC1 + y2*pC2);

#predicts based off which posterior probability is larger, if pos1<pos2 is true it should be class 1 and if it's false it should be class 0
predictions_PG = pos1<pos2
#converts from boolean values to 1s and 0s
predictions_PG = np.where(predictions_PG, 1, 0)
np.savetxt('7DforTestLabels.txt', predictions_KNN)

''' Old method was to use KNN for 7D as well, but through cross validation decided to use PG classifier
n_neighbors = 3
clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
clf.fit(Train_7D, labels_7D)
predictions_KNN = clf.predict(Test_7D)
np.savetxt('7DforTestLabels.txt', predictions_KNN)
'''

# Classifier of choice: KNN given its superiority in the 2D and Hyperspectral cases and the simplicity in my coded solution
n_neighbors = 3
clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
clf.fit(Train_HS, labels_HS)
predictions_KNN = clf.predict(Test_HS)
np.savetxt('HyperSpectralforTestLabels.txt', predictions_KNN)
