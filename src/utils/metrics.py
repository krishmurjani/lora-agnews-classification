import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_metrics(pred):
    """
    Compute evaluation metrics for model predictions.
    
    Args:
        pred: Prediction object with label_ids and predictions fields
        
    Returns:
        Dictionary of metrics
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    # Calculate various metrics
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='weighted')
    
    # Print class distribution for debugging
    print("\nPrediction distribution:")
    unique_preds, counts = np.unique(preds, return_counts=True)
    for i, count in zip(unique_preds, counts):
        print(f"  Class {i}: {count} ({count/len(preds)*100:.2f}%)")
    
    # Check if model is predicting a single class
    if np.unique(preds).size == 1:
        print("WARNING: Model is predicting only one class!")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
