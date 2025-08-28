"""
Utility functions for Disease Prediction Toolkit.
Keep this minimal - most logic will be in notebooks.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

def load_and_create_target(file_path):
    """Load dataset and create target variable based on risk factors."""
    df = pd.read_csv(file_path)
    
    # Create risk-based target variable
    risk_score = 0
    risk_score += np.where(df['age'] > 50, 2, 0)
    risk_score += np.where(df['age'] > 65, 1, 0)
    risk_score += np.where(df['trestbps'] > 140, 2, 0)
    risk_score += np.where(df['chol'] > 240, 2, 0)
    risk_score += df['cp_typical angina'] * 3
    risk_score += df['cp_atypical angina'] * 2
    risk_score += df['cp_asymptomatic'] * 1
    risk_score += np.where(df['exang'] == True, 2, 0)
    risk_score += np.where(df['oldpeak'] > 1.0, 1, 0)
    risk_score += df['ca']
    
    # Binary target based on risk threshold
    threshold = np.median(risk_score) if len(risk_score) > 1 else 3
    df['target'] = (risk_score > threshold).astype(int)
    
    return df

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    return plt.gcf()

def plot_roc_curve(y_true, y_proba, title="ROC Curve"):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    return plt.gcf()

def print_model_metrics(y_true, y_pred, y_proba=None, model_name="Model"):
    """Print comprehensive model metrics."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    print(f"\n=== {model_name.upper()} METRICS ===")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1 Score:  {f1_score(y_true, y_pred, zero_division=0):.4f}")
    
    if y_proba is not None:
        print(f"ROC AUC:   {roc_auc_score(y_true, y_proba):.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, 
                              target_names=['No Disease', 'Disease'], 
                              zero_division=0))
