#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys

from scipy.constants import precision
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)
features_train,features_test,labels_train,labels_test = train_test_split(features,labels, test_size = 0.3, random_state = 42)

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred= clf.predict(features_test)
prec_score = precision_score(labels_test, pred)
recall = recall_score(labels_test, pred)
# print("Accuracy", accuracy_score(labels_test, pred))
print(prec_score)
print(recall)

predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
print(precision_score(true_labels, predictions))
print(recall_score(true_labels, predictions))

### your code goes here 


