import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef

def train_logistic_regression(X_train, X_test, y_train, y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
        "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, average='weighted', zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }
    
    try:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
            if len(np.unique(y_test)) == 2:
                metrics["AUC Score"] = roc_auc_score(y_test, y_proba[:, 1])
            else:
                metrics["AUC Score"] = roc_auc_score(y_test, y_proba, multi_class='ovr')
        else:
            metrics["AUC Score"] = 0.0
    except:
        metrics["AUC Score"] = 0.0
        
    return model, metrics
