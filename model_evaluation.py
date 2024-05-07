# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 16:39:16 2024

@author: Wiktoria
"""
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, accuracy_score


def evaluate(model, x, y, text, use_best_estimator=False):
    print("Model:", text)
    
    #flag use_best_estimator (GridSearch or no GS)
    if use_best_estimator:
        model_preds = model.best_estimator_.predict(x)
    else:
        model_preds = model.predict(x)
    
    #confusion matrix
    conf_matrix = confusion_matrix(y, model_preds)
    print("Confusion matrix")
    print(conf_matrix)
    ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=model.classes_ if hasattr(model, 'classes_') else None).plot()

    #classification report
    print('Classification report')
    print(classification_report(y, model_preds))

    #accuracy
    accuracy = accuracy_score(y, model_preds)
    print('Accuracy:', accuracy)