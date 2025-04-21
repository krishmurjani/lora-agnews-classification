import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from peft import PeftModel
from tqdm import tqdm

from utils.data_processing import preprocess
from utils.metrics import compute_metrics
from utils.visualization import plot_confusion_matrix

def evaluate_model(model, dataset, id2label, labelled=True, batch_size=32, data_collator=None):
    """
    Evaluate a PEFT model on a dataset.
    
    Args:
        model: The model to evaluate
        dataset: The dataset to evaluate on
        id2label: Mapping from class indices to class names
        labelled: Whether the dataset has labels
        batch_size: Batch size for evaluation
        data_collator: Function to collate batches
        
    Returns:
        Dictionary of metrics (if labelled=True) or predictions (if labelled=False)
    """
    # Create the DataLoader
    eval_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    all_predictions = []
    all_labels = []
    all_probs = []  # Added to track prediction probabilities
    
    # Loop over the DataLoader
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        # Move each tensor in the batch to the device
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        
        # Get both predictions and probabilities
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)
        
        all_predictions.append(predictions.cpu())
        all_probs.append(probs.cpu())
        
        if labelled:
            # Expecting that labels are provided under the "labels" key.
            references = batch["labels"]
            all_labels.append(references.cpu())

    # Concatenate predictions and probabilities from all batches
    all_predictions = torch.cat(all_predictions, dim=0)
    all_probs = torch.cat(all_probs, dim=0)
    
    if labelled:
        all_labels = torch.cat(all_labels, dim=0)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        print(f"\nEvaluation Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Create confusion matrix directory if it doesn't exist
        os.makedirs("results/visualizations", exist_ok=True)
        
        # Create confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        class_names = list(id2label.values())
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, 
                   yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig('results/visualizations/confusion_matrix.png')
        plt.close()
        
        # Add error analysis for misclassified examples
        print("\nAnalyzing misclassifications...")
        misclassified_indices = torch.where(all_predictions != all_labels)[0]
        if len(misclassified_indices) > 0:
            sample_size = min(10, len(misclassified_indices))
            sample_indices = np.random.choice(misclassified_indices, sample_size, replace=False)
            
            print(f"\nSample of misclassified examples ({sample_size}/{len(misclassified_indices)}):")
            for idx in sample_indices:
                pred = all_predictions[idx].item()
                true = all_labels[idx].item()
                prob = all_probs[idx][pred].item()
                print(f"Example {idx}: Predicted {id2label[pred]} ({prob:.4f}), True {id2label[true]}")
        
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}, all_predictions, all_labels
    else:
        return all_predictions

def class_wise_performance(predictions, labels, id2label):
    """
    Calculate class-wise performance metrics.
    
    Args:
        predictions: Model predictions
        labels: True labels
        id2label: Mapping from class indices to class names
        
    Returns:
        DataFrame with class-wise metrics
    """
    results = []
    
    for idx, class_name in enumerate(id2label.values()):
        # Filter for examples of this class
        class_indices = torch.where(labels == idx)[0]
        class_preds = predictions[class_indices]
        class_true = labels[class_indices]
        class_accuracy = (class_preds == class_true).float().mean().item()
        class_examples = len(class_indices)
        correct_predictions = len(torch.where(class_preds == class_true)[0])
        
        results.append({
            'Class': class_name,
            'Accuracy': class_accuracy,
            'Correct': correct_predictions,
            'Total': class_examples
        })
    
    return pd.DataFrame(results)

def main():
    # Load the model and tokenizer
    model_path = "models/best_model"
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    
    # Load the dataset
    from datasets import load_dataset
    dataset = load_dataset('ag_news', split='test')
    
    # Get class mapping
    id2label = {i: label for i, label in enumerate(dataset.features["label"].names)}
    
    # Preprocess the dataset
    tokenized_dataset = dataset.map(
        lambda examples: preprocess(examples, tokenizer),
        batched=True, 
        remove_columns=["text"]
    )
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    
    # Create data collator
    from transformers import DataCollatorWithPadding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    
    # Evaluate the model
    metrics, predictions, labels = evaluate_model(
        model, 
        tokenized_dataset, 
        id2label,
        labelled=True, 
        batch_size=32, 
        data_collator=data_collator
    )
    
    # Calculate class-wise metrics
    class_metrics = class_wise_performance(predictions, labels, id2label)
    
    # Save metrics to a CSV file
    os.makedirs("results/metrics", exist_ok=True)
    class_metrics.to_csv("results/metrics/class_wise_performance.csv", index=False)
    
    print("\nClass-wise performance:")
    print(class_metrics)
    
    # Save overall metrics
    pd.DataFrame([metrics]).to_csv("results/metrics/overall_performance.csv", index=False)
    
    print("\nOverall metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

if __name__ == "__main__":
    main()
