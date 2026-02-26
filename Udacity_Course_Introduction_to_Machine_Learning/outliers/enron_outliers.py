#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
from matplotlib import pyplot as plt

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### your code below
salaries = []
bonuses = []

for point in data:
    salary = point[0]
    bonus = point[1]
    salaries.append(salary)
    bonuses.append(bonus)


plt.scatter(salaries, bonuses)
plt.xlabel("salary")
plt.ylabel("bonus")
plt.title("Salary vs Bonus")
plt.show()


