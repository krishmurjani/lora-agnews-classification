# LoRA AG News Classification

This repository contains our group project implementation of fine-tuning RoBERTa for AG News classification using Low-Rank Adaptation (LoRA).

## Project Overview

This project implements a text classification model for the AG News dataset using RoBERTa as the base model and LoRA for efficient fine-tuning. The model categorizes news articles into four classes: World, Sports, Business, and Sci/Tech.

## Setup and Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/lora-agnews-classification.git
cd lora-agnews-classification

# Install dependencies
pip install -r requirements.txt
```

## Dataset

The AG News corpus consists of news articles from more than 2,000 news sources. Each article is classified into one of four categories:
1. World
2. Sports
3. Business
4. Sci/Tech

The dataset is loaded using the Hugging Face datasets library.

## Model Architecture

- Base model: RoBERTa-base
- Fine-tuning method: Low-Rank Adaptation (LoRA)
- LoRA Configuration:
  - r = 36
  - alpha = 32
  - Dropout = 0.25
  - Target modules: Selected attention layers
  - Under 1M trainable parameters

## Training

To train the model:

```bash
python src/train.py
```

Alternatively, you can run the notebook:

```bash
jupyter notebook notebooks/model_training.ipynb
```

## Evaluation

The model achieves the following performance on the validation set:
- Accuracy: [Insert your accuracy here]
- F1 Score: [Insert your F1 score here]
- Precision: [Insert your precision here]
- Recall: [Insert your recall here]

## Inference

To run inference on new data:

```bash
python src/inference.py --input [path_to_input_file] --output [path_to_output_file]
```

## Results

Detailed results and visualizations can be found in the `results/` directory.

## Contributors

- [Team Member 1]
- [Team Member 2]
- [Team Member 3]
- [Team Member 4]

## License

[Insert your license here]
