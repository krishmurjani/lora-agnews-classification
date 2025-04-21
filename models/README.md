# Models

This directory contains trained models and checkpoints for the AG News classification project.

## Directory Structure

- `best_model/`: The best performing model saved after training
- `checkpoints/`: Intermediate model checkpoints saved during training
  - `checkpoint-6750/`
  - `checkpoint-13500/`
  - `checkpoint-20250/`
  - `checkpoint-27000/`

## Model Architecture

The models in this directory use the following architecture:
- Base model: RoBERTa-base
- Fine-tuning method: Low-Rank Adaptation (LoRA)
- LoRA Configuration:
  - r = 36
  - alpha = 32
  - Dropout = 0.25
  - Target modules: Selected attention layers
  - Under 1M trainable parameters

## Usage

To load the best model for inference:

```python
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from peft import PeftModel

# Load tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained("models/best_model")
model = RobertaForSequenceClassification.from_pretrained("models/best_model")

# Example inference
text = "NASA's new space telescope captures stunning images of distant galaxies."
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
outputs = model(**inputs)
prediction = outputs.logits.argmax(dim=1).item()
```

## Performance

The best model achieves the following performance on the validation set:
- Accuracy: [Insert your accuracy here]
- F1 Score: [Insert your F1 score here]
- Precision: [Insert your precision here]
- Recall: [Insert your recall here]
