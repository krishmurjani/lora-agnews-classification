import os
import argparse
import torch
import pandas as pd
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from datasets import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from utils.data_processing import clean_text, preprocess

def classify_text(model, tokenizer, text, id2label):
    """
    Classify a single text input.
    
    Args:
        model: The model to use for inference
        tokenizer: The tokenizer for preprocessing
        text: The text to classify
        id2label: Mapping from class indices to class names
        
    Returns:
        Predicted label and confidence score
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Clean the text first
    text = clean_text(text)
    
    # Update to match the preprocessing in training
    inputs = tokenizer(
        text, 
        truncation=True, 
        padding=True, 
        max_length=512,
        return_tensors="pt"
    ).to(device)
    
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        output = model(**inputs)
    
    # Get prediction scores and softmax probabilities
    logits = output.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    prediction = logits.argmax(dim=-1).item()
    confidence = probs[0][prediction].item()
    
    return id2label[prediction], confidence

def run_batch_inference(model, dataset, id2label, batch_size=32, data_collator=None):
    """
    Run inference on a dataset in batches.
    
    Args:
        model: The model to use for inference
        dataset: The dataset to run inference on
        id2label: Mapping from class indices to class names
        batch_size: Batch size for inference
        data_collator: Function to collate batches
        
    Returns:
        Numpy array of predictions
    """
    from torch.utils.data import DataLoader
    
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_probs = []
    
    # Run inference in batches
    for batch in tqdm(dataloader, desc="Running inference"):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)
        
        all_predictions.append(predictions.cpu())
        all_probs.append(probs.cpu())
    
    # Concatenate predictions from all batches
    all_predictions = torch.cat(all_predictions, dim=0)
    all_probs = torch.cat(all_probs, dim=0)
    
    return all_predictions.numpy(), all_probs.numpy()

def main():
    parser = argparse.ArgumentParser(description="Run inference with the trained model")
    parser.add_argument("--input", type=str, help="Path to input file (CSV or TXT)")
    parser.add_argument("--output", type=str, default="results/predictions/test_predictions.csv", 
                        help="Path to output file")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--model_path", type=str, default="models/best_model",
                        help="Path to the model directory")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Load model and tokenizer
    print(f"Loading model from {args.model_path}")
    tokenizer = RobertaTokenizer.from_pretrained(args.model_path)
    model = RobertaForSequenceClassification.from_pretrained(args.model_path)
    
    # Get class mapping
    id2label = model.config.id2label
    num_labels = len(id2label)
    
    # Load input data
    if args.input:
        print(f"Loading data from {args.input}")
        try:
            if args.input.endswith('.csv'):
                # Load as CSV
                df = pd.read_csv(args.input)
            elif args.input.endswith('.pkl'):
                # Load as pickle
                df = pd.read_pickle(args.input)
            elif args.input.endswith('.txt'):
                # Load as text file (one text per line)
                with open(args.input, 'r', encoding='utf-8') as f:
                    texts = [line.strip() for line in f.readlines()]
                df = pd.DataFrame({'text': texts})
            else:
                raise ValueError(f"Unsupported file format: {args.input}")
                
            # Create a Dataset object
            dataset = Dataset.from_pandas(df)
            
        except Exception as e:
            print(f"Error loading input file: {e}")
            return
            
        # Preprocess the dataset
        processed_dataset = dataset.map(
            lambda examples: preprocess(examples, tokenizer),
            batched=True, 
            remove_columns=["text"] if "text" in dataset.features else []
        )
        
        # Create data collator
        from transformers import DataCollatorWithPadding
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
        
        # Run inference
        print("Running inference...")
        predictions, probabilities = run_batch_inference(
            model, 
            processed_dataset, 
            id2label,
            batch_size=args.batch_size, 
            data_collator=data_collator
        )
        
        # Create output DataFrame
        df_output = pd.DataFrame({
            'ID': range(len(predictions)),
            'Label': predictions,
            'LabelText': [id2label[pred] for pred in predictions],
            'Confidence': [probabilities[i, pred] for i, pred in enumerate(predictions)]
        })
        
        # Save predictions to CSV
        df_output.to_csv(args.output, index=False)
        print(f"Inference complete. Predictions saved to {args.output}")
        
        # Create visualizations directory
        os.makedirs("results/visualizations", exist_ok=True)
        
        # Plot label distribution in predictions
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df_output, x='Label')
        plt.xticks(range(num_labels), list(id2label.values()), rotation=45)
        plt.title('Label Distribution in Predictions')
        plt.tight_layout()
        plt.savefig('results/visualizations/prediction_distribution.png')
        plt.close()
        
    else:
        # Interactive demo mode with example texts
        test_texts = [
            "Wall St. Bears Claw Back Into the Black. Short-sellers, Wall Street's dwindling band of ultra-cynics, are seeing green again.",
            "Kederis proclaims innocence. Olympic champion Kostas Kederis today left hospital ahead of his date with IOC inquisitors.",
            "US plans to send more troops to Iraq next year, despite calls to withdraw forces.",
            "NASA's new space telescope captures stunning images of distant galaxies."
        ]
        
        print("\nTesting model on example texts:")
        results = []
        for text in test_texts:
            pred_label, confidence = classify_text(model, tokenizer, text, id2label)
            print(f"\nText: {text}")
            print(f"Class: {pred_label}, Confidence: {confidence:.4f}")
            results.append({
                'Text': text,
                'Prediction': pred_label,
                'Confidence': confidence
            })
        
        # Save example results
        os.makedirs("results/predictions", exist_ok=True)
        pd.DataFrame(results).to_csv("results/predictions/example_predictions.csv", index=False)
        print("\nExample predictions saved to results/predictions/example_predictions.csv")

if __name__ == "__main__":
    main()
