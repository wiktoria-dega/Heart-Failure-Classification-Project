# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 19:55:32 2024

@author: Wiktoria
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from imblearn.over_sampling import SMOTE

df = pd.read_csv(r"C:\Users\Wiktoria\Desktop\Python Basics\Projekt2_Klasyfikacja\heart_data.csv")

df.info()

df.isna().sum()

pd.set_option('display.max_columns', None)

desc = df.describe()
desc

#balans klasowy
df['HeartDisease'].value_counts()
df['HeartDisease'].value_counts().plot(kind='bar')

#SMOTE
oversampler = SMOTE(random_state=25)

#X, y = oversampler.fit_resample(df.drop(columns='HeartDisease'), df['HeartDisease'])
#dane - cechy X nie są w pełni liczbowe

#sex, chestpaintype, restingECG, ExerciseAngina, ST_slope

#label encoding
sex_mapping = {'M': 0, 'F': 1}
df['Sex'] = df['Sex'].map(sex_mapping)

exAngina_mapping = {'N': 0, 'Y': 1}
df['ExerciseAngina'] = df['ExerciseAngina'].map(exAngina_mapping)


#one hot encoding
df = pd.get_dummies(df, columns=['ChestPainType', 'RestingECG', 'ST_Slope'], dtype=float)

df.info()

X, y = oversampler.fit_resample(df.drop(columns='HeartDisease'), df['HeartDisease'])

#sprawdzanie zbalansowania danych
plt.figure()
y.value_counts().plot(kind='bar')
#dane są zbalansowane

#boxplot cech
plt.figure()
plt.title('Boxplot analizowanej ramki danych')
X.boxplot()

#X + y
df_resampled = X
df_resampled['HeartDisease'] = y

plt.figure()
df_resampled['RestingBP'].plot.box()

plt.figure()
df_resampled['Cholesterol'].plot.box()

restBP_0 = (df_resampled['RestingBP'] == 0).sum() #1 wartosc odstajaca
chol_0 = (df_resampled['Cholesterol'] == 0).sum() #173 wartosci
chol_over_500 = (df_resampled['Cholesterol'] > 500).sum() # 4 wartosci
 
#wyfiltrowanie outliers
df_resampled = df_resampled[(df_resampled['RestingBP'] > 0) & (df_resampled['Cholesterol'] < 500)]

plt.figure()
plt.title('Boxplot analizowanej ramki po odrzuceniu outliers')
df_resampled.boxplot()

#pairplot
plt.figure(figsize=(14,14))
sns.pairplot(df_resampled, hue='HeartDisease')

#Korelacja
corr = df_resampled.corr()
corr

#Macierz korelacji
sns.heatmap(corr, annot=True)





