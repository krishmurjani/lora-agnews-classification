def clean_text(text, apply_dropout=False, dropout_prob=0.05):
    """
    Clean text by removing extra whitespace and optionally applying word dropout.
    
    Args:
        text: The text to clean
        apply_dropout: Whether to apply word dropout
        dropout_prob: The probability of dropping a word
        
    Returns:
        Cleaned text
    """
    # Basic cleaning
    text = text.strip()
    text = ' '.join(text.split())
    
    # Word dropout is disabled by default
    # If enabled, it would randomly remove words from the text
    if apply_dropout:
        import random
        words = text.split()
        kept_words = [word for word in words if random.random() > dropout_prob]
        text = ' '.join(kept_words)
    
    return text

def preprocess(examples, tokenizer):
    """
    Preprocess examples by tokenizing cleaned text.
    
    Args:
        examples: The examples to preprocess
        tokenizer: The tokenizer to use
        
    Returns:
        Tokenized examples
    """
    # Apply text cleaning
    cleaned_texts = [clean_text(text, apply_dropout=False) 
                    for text in examples['text']]
    
    # Tokenize texts
    tokenized = tokenizer(
        cleaned_texts, 
        truncation=True, 
        padding='max_length',
        max_length=512,
        return_token_type_ids=False,
        return_attention_mask=True
    )
    
    return tokenized
