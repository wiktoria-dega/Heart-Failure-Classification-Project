# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 18:35:20 2024

@author: Wiktoria
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report


def read_from_db(database_name='heart_disease_database', collection_name='heart_data'):
    
    client = MongoClient('mongodb://localhost:27017/')
    db = client[database_name]
    collection = db[collection_name]
    
    data = list(collection.find())
    
    df = pd.DataFrame(data)
    
    return df

df = read_from_db()

df.head()

df = df.drop(columns=['_id'])

#przygotowanie danych do ML
X = df.drop(columns=['HeartDisease'])
y = df['HeartDisease']

#podział na dane treningowe i testowe (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

#skalowanie
min_max_scaler = MinMaxScaler()

X_train_norm = min_max_scaler.fit_transform(X_train)
X_test_norm = min_max_scaler.transform(X_test)

y_test.value_counts()

#Budowa i ewaluacja modeli

#Drzewo decyzyjne
tree = DecisionTreeClassifier()

tree.fit(X_train_norm, y_train)

tree_preds_train = tree.predict(X_train_norm)
tree_preds_test = tree.predict(X_test_norm)

#Ocena skutecznosci

#Confusion matrix (macierz pomyłek)

print('train')
conf_train = confusion_matrix(y_train, tree_preds_train)
print(conf_train)

print(classification_report(y_train, tree_preds_train))

print('test')
conf_test = confusion_matrix(y_test, tree_preds_test)
print(conf_test)

print(classification_report(y_test, tree_preds_test))



