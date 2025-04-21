from peft import LoraConfig

def get_lora_config(
    r=36, 
    lora_alpha=32, 
    lora_dropout=0.25, 
    bias='none', 
    task_type="SEQ_CLS"
):
    """
    Create a LoRA configuration for fine-tuning.
    
    Args:
        r: LoRA attention dimension
        lora_alpha: LoRA alpha parameter
        lora_dropout: Dropout probability for LoRA layers
        bias: Bias type ('none', 'all', or 'lora_only')
        task_type: Type of task ('SEQ_CLS' for sequence classification)
        
    Returns:
        LoRA configuration object
    """
    # Target specific layers to reduce the number of trainable parameters
    target_modules = [
        "roberta.encoder.layer.0.attention.self.query",
        "roberta.encoder.layer.0.attention.self.key",
        "roberta.encoder.layer.5.attention.self.query",
        "roberta.encoder.layer.10.attention.self.query",
    ]
    
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        target_modules=target_modules,
        task_type=task_type,
    )
