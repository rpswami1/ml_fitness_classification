import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef

def calculate_metrics(y_true, y_pred, y_proba=None):
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "Recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, average='weighted', zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred)
    }
    
    if y_proba is not None:
        try:
            if len(np.unique(y_true)) == 2:
                metrics["AUC Score"] = roc_auc_score(y_true, y_proba[:, 1])
            else:
                metrics["AUC Score"] = roc_auc_score(y_true, y_proba, multi_class='ovr')
        except:
            metrics["AUC Score"] = 0.0
    else:
        metrics["AUC Score"] = 0.0
        
    return metrics

def train_logistic_regression(X_train, X_test, y_train, y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Train Metrics
    y_pred_train = model.predict(X_train)
    y_proba_train = model.predict_proba(X_train) if hasattr(model, "predict_proba") else None
    train_metrics = calculate_metrics(y_train, y_pred_train, y_proba_train)
    
    # Test Metrics
    y_pred_test = model.predict(X_test)
    y_proba_test = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    test_metrics = calculate_metrics(y_test, y_pred_test, y_proba_test)
    
    return model, train_metrics, test_metrics
