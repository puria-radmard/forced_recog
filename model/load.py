import torch
from typing import Type, Union
from model.base import BaseChatWrapper, HuggingFaceChatWrapper
from model.anthropic import AnthropicChatWrapper
from model.openai import OpenAIChatWrapper
from model.gemini import GeminiChatWrapper
from model.mock import load_mock_model
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
)


def load_model(
    model_name: str, 
    device: str = "auto",
    torch_dtype: torch.dtype = torch.float16
) -> BaseChatWrapper:
    """
    Load any model type and return a unified chat wrapper.
    
    Args:
        model_name: Name or path of the model to load
        device: Device to load the model on (for HuggingFace models)
        torch_dtype: PyTorch data type for model weights (for HuggingFace models)
        
    Returns:
        BaseChatWrapper instance with unified interface
        
    Example:
        >>> # HuggingFace model
        >>> chat_wrapper = load_model("meta-llama/Llama-2-7b-chat-hf")
        >>> # Anthropic model
        >>> chat_wrapper = load_model("claude-3-5-sonnet-20241022")
        >>> # OpenAI model
        >>> chat_wrapper = load_model("gpt-4o-mini")
        >>> # Gemini model
        >>> chat_wrapper = load_model("gemini-1.5-flash")
        >>> # Mock model for testing
        >>> chat_wrapper = load_model("mock")  # or "mock-random", "mock-alternating"
    """
    model_name_lower = model_name.lower()
    
    # Determine model type and return appropriate wrapper
    if model_name_lower.startswith('mock'):
        # Mock model for testing
        return load_mock_model(model_name)
    elif model_name_lower.startswith('claude-'):
        return AnthropicChatWrapper(model_name)
    elif model_name_lower.startswith('gemini-'):
        return GeminiChatWrapper(model_name)
    elif model_name_lower.startswith('gpt-'):
        return OpenAIChatWrapper(model_name)
    else:
        # HuggingFace model
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except TypeError:
            tokenizer = AutoTokenizer.from_pretrained(model_name, legacy = False, from_slow = False)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if device == 'auto':
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map = 'auto'
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
            ).to(device)
        
        return HuggingFaceChatWrapper(model, tokenizer)
