#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

clf = SVC(kernel='rbf',C=10000)

t0 = time()
clf.fit(features_train,labels_train)
print "training time:", round(time()-t0, 3), "s"

t0 = time()
pred =  clf.predict(features_test)
print "predicting time:", round(time()-t0, 3), "s"

acc = accuracy_score(labels_test, pred)

print "Accuracy:", acc
print "Answer:", np.sum(pred == 1)



#########################################################
### your code goes here ###

#########################################################


