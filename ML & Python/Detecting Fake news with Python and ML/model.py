# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 23:54:11 2022

@author: 91709
"""

import pandas as pd
import numpy as np
import itertools

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


df = pd.read_csv("news.csv")

print(df.shape)
print(df.columns)

y  = df.label
x = df.text
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2,random_state=0)

tfidf = TfidfVectorizer(stop_words='english',max_df=0.7)

x_train = tfidf.fit_transform(x_train)
x_test = tfidf.transform(x_test)

pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(x_train, y_train)

y_pred = pac.predict(x_test)

acc = accuracy_score(y_test, y_pred)
print(f"accuracy for 100 iterations is {acc}")

print(confusion_matrix(y_test, y_pred,labels=['FAKE','REAL']))
