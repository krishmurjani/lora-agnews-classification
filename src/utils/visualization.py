import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(trainer, dataset, class_names, output_dir="results/visualizations"):
    """
    Generate and save a confusion matrix visualization.
    
    Args:
        trainer: The model trainer
        dataset: The dataset to evaluate
        class_names: List of class names
        output_dir: Directory to save the visualization
        
    Returns:
        None (saves the visualization to disk)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get predictions
    predictions = trainer.predict(dataset)
    preds = predictions.predictions.argmax(-1)
    labels = predictions.label_ids
    
    # Create confusion matrix
    cm = confusion_matrix(labels, preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Plot normalized confusion matrix
    plt.figure(figsize=(10, 8))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_normalized.png'))
    plt.close()

def plot_training_history(history, output_dir="results/visualizations"):
    """
    Plot training and validation loss/metrics over epochs.
    
    Args:
        history: Dictionary with training history
        output_dir: Directory to save the visualizations
        
    Returns:
        None (saves visualizations to disk)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot training loss
    if 'loss' in history:
        plt.figure(figsize=(10, 6))
        plt.plot(history['loss'], label='Training Loss')
        if 'eval_loss' in history:
            plt.plot(history['eval_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'training_loss.png'))
        plt.close()
    
    # Plot accuracy
    if 'accuracy' in history:
        plt.figure(figsize=(10, 6))
        plt.plot(history['accuracy'], label='Training Accuracy')
        if 'eval_accuracy' in history:
            plt.plot(history['eval_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'training_accuracy.png'))
        plt.close()

def plot_class_distribution(labels, class_names, output_dir="results/visualizations"):
    """
    Plot the distribution of classes in a dataset.
    
    Args:
        labels: Array of labels
        class_names: List of class names
        output_dir: Directory to save the visualization
        
    Returns:
        None (saves the visualization to disk)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Count labels
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # Create DataFrame for easier plotting
    import pandas as pd
    df = pd.DataFrame({
        'Class': [class_names[label] for label in unique_labels],
        'Count': counts
    })
    
    # Plot class distribution
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Class', y='Count', data=df)
    plt.title('Class Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'))
    plt.close()
