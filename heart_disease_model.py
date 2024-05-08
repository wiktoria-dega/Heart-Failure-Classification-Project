import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import model_evaluation
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import save_read_model_scaler


def read_from_db(database_name='heart_disease_database', 
                 collection_name='heart_data'):
    
    client = MongoClient('mongodb://localhost:27017/')
    db = client[database_name]
    collection = db[collection_name]
    
    data = list(collection.find())
    
    df = pd.DataFrame(data)
    
    return df

df = read_from_db()

df.head()

df = df.drop(columns=['_id', 'ChestPainType_TA', 'ST_Slope_Flat',
                      'ChestPainType_ATA', 'RestingECG_ST', 'ChestPainType_NAP',
                      'RestingECG_LVH', 'ST_Slope_Down'])

#preparation of data for ML
X = df.drop(columns=['HeartDisease'])
y = df['HeartDisease']

#split into training and test data (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, random_state=10)

#scaling
min_max_scaler = MinMaxScaler()

X_train_norm = min_max_scaler.fit_transform(X_train)
X_test_norm = min_max_scaler.transform(X_test)

y_test.value_counts()

#model construction and evaluation
#decision tree model
tree = DecisionTreeClassifier(random_state=42)

tree.fit(X_train_norm, y_train)

model_evaluation.evaluate(tree, X_train_norm, y_train, "Decision Tree - Train")
model_evaluation.evaluate(tree, X_test_norm, y_test, "Decision Tree - Test")

for i,j in zip(tree.feature_importances_, X_train.columns):
    print(i, j)

#decision tree by GSCV
params = {
    'max_depth': [None, 3, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5]
    }

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='macro'),
    'recall': make_scorer(recall_score, average='macro'),
    'f1': make_scorer(f1_score, average='macro')
    }

gridSearch = GridSearchCV(tree, params, scoring=scoring, 
                          verbose=2, cv=4, refit='accuracy' )


gridSearch.fit(X_train_norm, y_train)

model_evaluation.evaluate(gridSearch, X_train_norm, y_train,
                          'Decision Tree by GridSearchCV - Train',
                          use_best_estimator=True)
model_evaluation.evaluate(gridSearch, X_test_norm, y_test,
                          'Decision Tree by GridSearchCV - Test',
                          use_best_estimator=True)

#random forest
rnd_forest = RandomForestClassifier()

rnd_forest.fit(X_train_norm, y_train)

model_evaluation.evaluate(rnd_forest, X_train_norm, y_train, 'Random Forest - Train')
model_evaluation.evaluate(rnd_forest, X_test_norm, y_test, 'Random Forest - Test')

#random forest by GSCV
params_forest = {
    'n_estimators': [100, 200, 300],
    'criterion': ['gini', 'entropy'],
    'max_depth': [4, 6, 8, 10],
    'max_features': ['auto', 'sqrt', 'log2']
    }

scoring_forest = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='macro'),
    'recall': make_scorer(recall_score, average='macro'),
    'f1': make_scorer(f1_score, average='macro')
    }

gs_forest = GridSearchCV(rnd_forest, params_forest, scoring=scoring_forest, 
                         verbose=2, cv=4, refit='accuracy')

gs_forest.fit(X_train_norm, y_train)

model_evaluation.evaluate(gs_forest, X_train_norm, y_train,
                          'Random Forest by GridSearchCV - Train',
                          use_best_estimator=True)
model_evaluation.evaluate(gs_forest, X_test_norm, y_test,
                          'Random Forest by GridSearchCV - Test',
                          use_best_estimator=True)

#logistic regression
modelLR = LogisticRegression()

modelLR.fit(X_train_norm, y_train)

model_evaluation.evaluate(modelLR, X_train_norm, 
                          y_train,'Logistic Regression - Train')
model_evaluation.evaluate(modelLR, X_test_norm, 
                          y_test,'Logistic Regression - Test')

# CV + KNN - nearest neighbor classifier
#crosvalidation settings
cv = KFold(n_splits=5, shuffle=True, random_state=42)

max_N = 10
mean_acc_list = []

for k in range(1, max_N + 1):
    kNN = KNeighborsClassifier(n_neighbors=k)
    print('Model: k=', k)
    # CV
    knn_scores = cross_val_score(kNN, 
                                 X_train_norm, 
                                 y_train, 
                                 cv=cv, 
                                 scoring='accuracy')
    mean_acc = knn_scores.mean()
    mean_acc_list.append(mean_acc)
    print(f'Mean classification accuracy kNN {mean_acc}')
    
    kNN.fit(X_train_norm, y_train)
    model_evaluation.evaluate(model = kNN, 
              x = X_test_norm, 
              y = y_test, 
              text = f'Model kNN, k={k}')
    
plt.plot(mean_acc_list)
plt.xlabel('K')
plt.ylabel('Mean accuracy')
plt.title('K vs Mean accuracy')
plt.grid()

#naive bayes
naive = GaussianNB()
naive.fit(X_train_norm, y_train)

model_evaluation.evaluate(naive, X_train_norm, y_train, 'Naive Bayes - Train')
model_evaluation.evaluate(naive, X_test_norm, y_test, 'Naive Bayes - Test')


nb_scores = cross_val_score(naive, 
                             X_train_norm, 
                             y_train, 
                             cv=cv, 
                             scoring='accuracy')

print(f'Mean classification accuracy NB: {nb_scores.mean()}')


#XGBoost
xgbc = xgb.XGBClassifier()
xgbc.fit(X_train_norm, y_train)

model_evaluation.evaluate(xgbc, X_train_norm, y_train,'XGBoost - Train')
model_evaluation.evaluate(xgbc, X_test_norm, y_test, 'XGBoost - Test')


#save model and scaler
save_read_model_scaler.save_model_scaler(tree, min_max_scaler)


