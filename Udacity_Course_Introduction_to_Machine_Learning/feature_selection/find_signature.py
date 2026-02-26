# -*- coding: utf-8 -*-
#!/usr/bin/python

import pickle
import cPickle as pickle
import numpy
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

numpy.random.seed(42)


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../feature_selection/word_data_overfit.pkl"
authors_file = "../feature_selection/email_authors_overfit.pkl"
word_data = pickle.load( open(words_file, "r"))
authors = pickle.load( open(authors_file, "r") )



### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()


### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]

clf = DecisionTreeClassifier()
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(labels_test,pred)
print(acc)
imps = clf.feature_importances_
indices = np.where(imps > 0.2)
print (imps[imps > 0.2])
print(indices)
features = vectorizer.get_feature_names()
print(features[18853])
# for i, v in enumerate(clf.feature_importances_):
#     if v > 0.2:
#         print(i, v)
### your code goes here







# with open(words_file, "rb") as f:
#     data = pickle.load(f)
#
# # Проверяем тип
# if isinstance(data, list):
#     cleaned = []
#     for text in data:
#         if isinstance(text, basestring):  # str или unicode
#             # удаляем слово sshacklensf
#             text_cleaned = " ".join([w for w in text.split() if w != "cgermannsf"])
#             cleaned.append(text_cleaned)
#         else:
#             cleaned.append(text)  # оставляем как есть, если не строка
# else:
#     raise TypeError("Ожидался список, а не %s" % type(data))
#
# # Сохраняем обратно
# with open(words_file, "wb") as f:
#     pickle.dump(cleaned, f)
#
# print("Готово! Слово удалено из всех писем.")


