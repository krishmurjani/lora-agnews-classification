# Data

This directory is intended for storing data files related to the AG News classification project.

## Dataset Description

The AG News corpus consists of news articles from more than 2,000 news sources. Each article is classified into one of four categories:
1. World
2. Sports
3. Business
4. Sci/Tech

The dataset is loaded using the Hugging Face datasets library with:
```python
from datasets import load_dataset
dataset = load_dataset('ag_news')
```

## Data Processing

The data processing pipeline includes:
1. Text cleaning (removing extra whitespace)
2. Tokenization with RoBERTa tokenizer
3. Truncation and padding to a maximum length of 512 tokens

## Directory Structure

This directory may contain:
- Custom datasets (if you're working with data beyond the standard AG News dataset)
- Processed datasets (if you save intermediate processing results)
- Test datasets for evaluation

## Note

The main AG News dataset is loaded directly through the Hugging Face datasets library and is not stored in this directory.
