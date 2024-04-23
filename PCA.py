# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 09:13:47 2024

@author: Wiktoria
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pymongo import MongoClient
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r"C:\Users\Wiktoria\Desktop\Python Basics\Projekt2_Klasyfikacja\heart_data.csv")

df.isna().sum()


df['HeartDisease'].value_counts()

#label encoding
sex_mapping = {'M': 0, 'F': 1}
df['Sex'] = df['Sex'].map(sex_mapping)

exAngina_mapping = {'N': 0, 'Y': 1}
df['ExerciseAngina'] = df['ExerciseAngina'].map(exAngina_mapping)

#one hot encoding
df = pd.get_dummies(df, columns=['ChestPainType', 'RestingECG', 'ST_Slope'], dtype=float)

print(df)

df.info()

pd.set_option('display.max_columns', None)

df.describe()

df.plot.hist()

corr = df.corr()
sns.heatmap(corr, annot=True) 

#skalowanie
scaler = StandardScaler()

scaled_values = scaler.fit_transform(df)

scaled_df = pd.DataFrame(scaled_values, columns=df.columns)

print(scaled_df)

scaled_df.describe()

df.plot.hist()

scaled_df.plot.hist()


def perform_PCA(scaled_values, n_components):
    pca = PCA(n_components=n_components)
    X_transformed = pca.fit_transform(scaled_values)

    print("Przed transformacją:", scaled_values[0, :])
    print("Po transformacji:", X_transformed[0, :])
    print("Wymiary danych przed transformacją:", scaled_values.shape)
    print("Wymiary danych po transformacji:", X_transformed.shape)

    print("Wariancja wyjaśniona:", pca.explained_variance_)
    n_components_range = np.arange(pca.n_components_) + 1
    print("Wariancja wyjaśniona dla każdej komponenty:", pca.explained_variance_ratio_)
    print("Suma wariancji wyjaśnionej:", sum(pca.explained_variance_ratio_))

    plt.figure()
    plt.plot(n_components_range, pca.explained_variance_ratio_, 'x-', linewidth=3)
    plt.title('Wykres osypiska')
    plt.xlabel('Numer komponentu (składowej)')
    plt.ylabel('Wariancja wyjaśniona [*100%]')
    plt.show()

    cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
    plt.figure()
    plt.plot(n_components_range, cumsum_variance, 'x-', linewidth=3)
    plt.title('Wykres wariancji skumulowanej')
    plt.xlabel('Numer komponentu (składowej)')
    plt.ylabel('Wariancja wyjaśniona [*100%]')
    plt.show()

    return X_transformed

#domyslnie tyle składowych ile cech+output
X_transformed_from_PCA = perform_PCA(scaled_values, 19)

#sprawdzenie dla 9 składowych
X_transformed_from_PCA = perform_PCA(scaled_values, 9)

sns.pairplot(scaled_df)

#główne składowe (pierwsza i druga)
P1 = X_transformed_from_PCA[:, 0]
P2 = X_transformed_from_PCA[:, 1]

plt.figure()
plt.scatter(P1, P2)
plt.title('Składowe główne P1 i P2')
plt.xlabel('Pierwsza składowa główna P1')
plt.ylabel('Druga składowa główna P2')
plt.show()

