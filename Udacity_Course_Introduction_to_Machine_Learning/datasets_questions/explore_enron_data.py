#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = "../final_project/final_project_dataset.pkl"
enron_file_handler = open(enron_data, "r")
enron = pickle.load(enron_file_handler)
enron_file_handler.close()


count_poi = 0
for person in enron.values():
    if person.get('poi') == True:
        count_poi += 1

count_email = 0
for person in enron.values():
    if person.get('email_address') != 'NaN':
        count_email += 1

print(count_poi)
print(len(enron))
print(enron)