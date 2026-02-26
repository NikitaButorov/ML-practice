#!/usr/bin/python
import math
import sys
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

# ==============================
# Task 1: Select features
# ==============================
features_list = ['poi', 'bonus_salary_ratio', 'poi_total_messages']

# ==============================
# Load dataset
# ==============================
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# ==============================
# Filter out records with NaN in important features
# ==============================



def safe_float(x):
    try:
        return float(x)
    except:
        return 0.0
outliers = [
    "TOTAL",
    "THE TRAVEL AGENCY IN THE PARK",
    "LOCKHART EUGENE E",
    "BELFER ROBERT",
    "BHATNAGAR SANJAY",
    "LAY KENNETH L",
    "SKILLING JEFFREY K"


]
for name in outliers:
    if name in data_dict:
        del data_dict[name]

new_dict = {}

for name, person in data_dict.items():
    salary = person.get('salary')
    bonus = person.get('bonus')
    from_poi = person.get('from_poi_to_this_person')
    to_poi = person.get('from_this_person_to_poi')


    def safe_float(x):
        try:
            val = float(x)
            if np.isinf(val) or np.isnan(val):
                return 0.0
            return val
        except:
            return 0.0

    bonus_salary_ratio = safe_float(salary) + safe_float(bonus)
    poi_total_messages = safe_float(from_poi) + safe_float(to_poi)


    if bonus_salary_ratio == 0 or poi_total_messages == 0:
        continue

    person['bonus_salary_ratio'] = bonus_salary_ratio
    person['poi_total_messages'] = poi_total_messages
    new_dict[name] = person

my_dataset = new_dict





# ==============================
# Extract features and labels
# ==============================
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

# ==============================
# Step 1: Temporary model to find outliers
# ==============================

# probs = temp_clf.predict_proba(features)[:, 1]
# errors = np.abs(probs - labels)
#
# num_outliers = 7  # number of worst points to remove
# worst_indices = np.argsort(errors)[::-1][:num_outliers]
#
# sorted_keys = sorted(my_dataset.keys())
# worst_people = [sorted_keys[i] for i in worst_indices]
#
# for name in worst_people:
#     print("Removing outlier:", name)
#     del my_dataset[name]

# ==============================
# Step 2: Re-extract features and labels after outlier removal
# ==============================
# data_clean = featureFormat(my_dataset, features_list, sort_keys=True)
# labels_clean, features_clean = targetFeatureSplit(data_clean)
# features_clean = np.array(features_clean)
# labels_clean = np.array(labels_clean)

# ==============================
# Step 3: Split data into train/test for final model
# ==============================
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.3, random_state=42)

clf=SVC(C=0.01, kernel='linear',gamma='auto',class_weight='balanced')
clf.fit(features_train, labels_train)


pred = clf.predict(features_test)
print(precision_score(labels_test,pred))





ratios = [f[0] for f in features]
messages = [f[1] for f in features]
poi_labels = labels

X = np.array(ratios)
Y = np.array(messages)
labels_arr = np.array(poi_labels)

x_min, x_max = X.min() - 0.01, X.max() + 0.01
y_min, y_max = Y.min() - 1, Y.max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
plt.contour(xx, yy, Z, colors='k', linewidths=1)

for i in range(len(X)):
    if labels_arr[i] == 1:
        plt.scatter(X[i], Y[i], color='red', label='POI' if i == 0 else "")
    else:
        plt.scatter(X[i], Y[i], color='blue', label='Non-POI' if i == 0 else "")

plt.xlabel("Bonus / Salary Ratio")
plt.ylabel("To POI messages")
plt.title("Decision Boundary + Scatter Plot (Cleaned Data)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()







# ==============================
# Step 6: Dump classifier, dataset, and features_list
# ==============================
dump_classifier_and_data(clf, my_dataset, features_list)
