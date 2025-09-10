def parse_forward_sft_key(sft_key: str):
    if not sft_key.startswith('temp'):
        raise ValueError(f"SFT key must start with 'temp', got: {sft_key}")
    
    parts = sft_key.split('_')
    if len(parts) != 2:
        raise ValueError(f"SFT key must be in format 'tempX_styleY', got: {sft_key}")
    
    temp_str = parts[0].replace('temp', '')
    style_str = parts[1].replace('style', '')
    
    try:
        temp = float(temp_str)
    except ValueError:
        raise ValueError(f"Could not parse temperature from: {temp_str}")
    
    return temp, style_str
    

def train_step_forward(model, tokenizer, batch_pairs, max_seq_length, device):
    """Perform one training step."""
    model.train()
    
    # Tokenize inputs and targets separately first to get input lengths
    input_texts = [pair["input"] for pair in batch_pairs]
    target_texts = [pair["target"] for pair in batch_pairs]
    
    # Get input lengths for masking (before adding targets)
    input_lengths = []
    for inp in input_texts:
        input_tokens = tokenizer(inp, add_special_tokens=False, return_tensors="pt")
        input_lengths.append(input_tokens['input_ids'].shape[1])
    
    # Create full texts (input + target)
    full_texts = [inp + tgt for inp, tgt in zip(input_texts, target_texts)]
    
    # Tokenize full texts
    tokenized = tokenizer(
        full_texts,
        max_length=max_seq_length,
        truncation=True,
        padding=True,
        return_tensors="pt",
        add_special_tokens = False
    )

    # Move to device
    input_ids = tokenized['input_ids'].to(device)
    attention_mask = tokenized['attention_mask'].to(device)
    
    # Create labels (mask input portion)
    labels = input_ids.clone()
    for i, inp_len in enumerate(input_lengths):
        assert tokenizer.decode(input_ids[i,:inp_len]) == input_texts[i]
        labels[i, :inp_len] = -100  # Ignore loss on input tokens
        
    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    
    return outputs.loss
