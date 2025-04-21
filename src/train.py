import os
import random
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from torch.nn import functional as F

# Import custom modules
from utils.data_processing import clean_text, preprocess
from utils.metrics import compute_metrics
from utils.visualization import plot_confusion_matrix
from configs.lora_config import get_lora_config

# Set random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    
    print(f"\ntrainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.4f}")
    return trainable_params

def main():
    # Set seed
    set_seed(42)
    
    # Set output directory
    output_dir = "models"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load base model
    base_model = 'roberta-base'
    
    # Load dataset
    dataset = load_dataset('ag_news', split='train')
    tokenizer = RobertaTokenizer.from_pretrained(base_model)
    
    # Preprocess data
    tokenized_dataset = dataset.map(
        lambda examples: preprocess(examples, tokenizer),
        batched=True, 
        remove_columns=["text"]
    )
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    
    # Extract label information
    num_labels = dataset.features['label'].num_classes
    class_names = dataset.features["label"].names
    print(f"Number of labels: {num_labels}")
    print(f"The labels: {class_names}")
    
    # Create label mappings
    id2label = {i: label for i, label in enumerate(class_names)}
    label2id = {label: i for i, label in id2label.items()}
    
    # Create data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    
    # Load model
    model = RobertaForSequenceClassification.from_pretrained(
        base_model,
        id2label=id2label,
        label2id=label2id
    )
    
    # Split dataset
    split_datasets = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_datasets['train']
    eval_dataset = split_datasets['test']
    
    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(eval_dataset)}")
    
    # Apply LoRA config
    peft_config = get_lora_config()
    peft_model = get_peft_model(model, peft_config)
    
    # Print trainable parameters
    trainable_params = print_trainable_parameters(peft_model)
    assert trainable_params < 1000000, f"Trainable parameters ({trainable_params}) exceed 1M limit!"
    
    # Set up trainer
    training_args = TrainingArguments(
        output_dir="./models/checkpoints",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=4,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        logging_dir='./logs',
        logging_steps=100,
        report_to="none",
        warmup_ratio=0.1,
    )
    
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Train model
    print("Starting training...")
    result = trainer.train()
    print(f"Training completed. Training loss: {result.training_loss}")
    
    # Evaluate model
    eval_results = trainer.evaluate()
    print("\nEvaluation Results:")
    for key, value in eval_results.items():
        print(f"{key}: {value}")
    
    # Save evaluation metrics
    os.makedirs("results/metrics", exist_ok=True)
    pd.DataFrame([eval_results]).to_csv("results/metrics/model_performance.csv", index=False)
    
    # Create confusion matrix
    plot_confusion_matrix(trainer, eval_dataset, class_names, "results/visualizations")
    
    # Save the final model
    model_save_path = os.path.join(output_dir, "best_model")
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Print final parameter count
    print("\nFinal model details:")
    print_trainable_parameters(peft_model)
    print(f"Number of classes: {num_labels}")
    print(f"Class names: {class_names}")
    print(f"Final training metrics: {eval_results}")
    print("Training complete!")

if __name__ == "__main__":
    main()
