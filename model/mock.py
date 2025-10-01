"""
Mock Chat Wrapper for Testing

This module provides a mock implementation of BaseChatWrapper for testing the
experiment pipeline without making actual LLM calls. It generates deterministic
or random responses instantly, allowing for fast validation of:
- Conversation generation
- Logging functionality  
- Results processing
- File I/O operations

Usage:
    chat_wrapper = MockChatWrapper(mode="deterministic")  # or "random", "alternating"
    
Model naming in config:
    - "mock": Default deterministic mode (always chooses "1")
    - "mock-random": Random choices with seed
    - "mock-alternating": Alternates between "1" and "2"
"""

from typing import List, Dict, Optional, Any
import torch
from model.base import BaseChatWrapper


class MockTokenizer:
    """Mock tokenizer for compatibility with existing code."""
    
    def __init__(self):
        self.pad_token = "[PAD]"
        self.eos_token = "[EOS]"
        self.bos_token = "[BOS]"
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Mock encode - returns token IDs for "1" and "2"."""
        if text == "1":
            return [3]  # Token ID for "1"
        elif text == "2":
            return [4]  # Token ID for "2"
        else:
            # For other text, return dummy tokens
            return [5] * (len(text) // 4 + 1)
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Mock decode."""
        if 3 in token_ids:
            return "1"
        elif 4 in token_ids:
            return "2"
        else:
            return "[MOCK_TEXT]"


class MockOutputs:
    """Mock outputs object matching transformers output format."""
    
    def __init__(self, logits: torch.Tensor):
        self.logits = logits


class MockChatWrapper(BaseChatWrapper):
    """
    Mock chat wrapper for testing without LLM calls.
    
    This wrapper simulates model behavior by returning mock logits and responses
    based on the configured mode. It's designed for testing the experiment pipeline
    without requiring actual model inference.
    
    [TESTING] This wrapper uses MOCK LOGITS for testing purposes.
    The logits are artificially generated based on the selected mode and do not
    represent any real model behavior. Use only for pipeline testing.
    
    Modes:
        - deterministic: Always returns high probability for choice "1"
        - random: Returns random probabilities (seeded for reproducibility)
        - alternating: Alternates between choosing "1" and "2"
    """
    
    def __init__(self, mode: str = "deterministic", seed: int = 42):
        """
        Initialize the mock chat wrapper.
        
        Args:
            mode: Testing mode ("deterministic", "random", "alternating")
            seed: Random seed for reproducible testing
        """
        super().__init__(f"mock-{mode}")
        
        self.mode = mode
        self.seed = seed
        self.call_count = 0
        
        # Set random seed for reproducibility
        torch.manual_seed(seed)
        
        # Mock tokenizer
        self.tokenizer = MockTokenizer()
        self.device = "mock"
        
        print(f"[TESTING] MockChatWrapper initialized in '{mode}' mode")
        print(f"[TESTING]   This wrapper generates instant mock responses for testing")
        print(f"[TESTING]   No actual LLM calls will be made")
    
    def format_chat(
        self,
        *_,
        system_prompt: Optional[str] = None,
        in_context_questions: Optional[List[str]] = None,
        in_context_answers: Optional[List[str]] = None,
        user_message: Optional[str] = None,
        prefiller: Optional[str] = None,
        keep_bos: bool = False,
        skip_special_tokens: bool = False,
        **kwargs
    ) -> str:
        """
        Format chat by simply concatenating components.
        
        Returns a mock formatted string that captures the main elements.
        """
        parts = []
        
        if system_prompt:
            parts.append(f"[SYSTEM] {system_prompt[:50]}...")
        
        if in_context_questions and in_context_answers:
            for q, a in zip(in_context_questions, in_context_answers):
                parts.append(f"[USER] {q[:30]}...")
                parts.append(f"[ASSISTANT] {a[:30]}...")
        
        if user_message:
            parts.append(f"[USER] {user_message[:50]}...")
        
        if prefiller:
            parts.append(f"[PREFILL] {prefiller}")
        
        formatted = " | ".join(parts)
        return f"[MOCK_CHAT] {formatted}"
    
    def forward(
        self,
        chats: List[str],
        past_key_values: Optional[Any] = None,
        return_dict: bool = True,
        use_cache: bool = True,
        return_input_ids: bool = False,
        **forward_kwargs: Any
    ) -> MockOutputs:
        """
        Generate mock forward pass outputs.
        
        Returns mock logits based on the configured mode:
        - deterministic: Always favors choice "1" (80% vs 20%)
        - random: Random probabilities
        - alternating: Alternates between "1" and "2"
        
        Args:
            chats: List of formatted chat strings
            past_key_values: Ignored for mock
            return_dict: Ignored for mock
            use_cache: Ignored for mock
            return_input_ids: Ignored for mock
            
        Returns:
            MockOutputs with logits tensor
        """
        batch_size = len(chats)
        vocab_size = 1000  # Mock vocabulary size
        seq_len = 1  # Single token output
        
        # Create base logits tensor (all zeros)
        logits = torch.zeros((batch_size, seq_len, vocab_size))
        
        # Set logits for tokens at positions 3 (for "1") and 4 (for "2")
        for i in range(batch_size):
            if self.mode == "deterministic":
                # Always favor choice "1" with realistic-looking probabilities
                # This produces ~0.95 vs ~0.05 probabilities (always selects "1")
                logits[i, 0, 3] = 3.0  # High logit for "1" 
                logits[i, 0, 4] = 0.0  # Low logit for "2"
                
            elif self.mode == "random":
                # Random logits
                logits[i, 0, 3] = torch.randn(1).item() * 2
                logits[i, 0, 4] = torch.randn(1).item() * 2
                
            elif self.mode == "alternating":
                # Alternate based on call count
                if self.call_count % 2 == 0:
                    logits[i, 0, 3] = 3.0  # Favor "1"
                    logits[i, 0, 4] = 0.0
                else:
                    logits[i, 0, 3] = 0.0
                    logits[i, 0, 4] = 3.0  # Favor "2"
            
            else:
                raise ValueError(f"Unknown mock mode: {self.mode}")
        
        self.call_count += 1
        
        return MockOutputs(logits=logits)
    
    def generate(
        self,
        chats: List[str],
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        do_sample: bool = False,
        **kwargs
    ) -> List[str]:
        """
        Generate mock text responses.
        
        Returns mock responses based on the configured mode.
        
        Args:
            chats: List of formatted chat strings
            max_new_tokens: Ignored for mock
            temperature: Ignored for mock
            do_sample: Ignored for mock
            **kwargs: Ignored for mock
            
        Returns:
            List of mock response strings
        """
        responses = []
        
        for _ in chats:
            if self.mode == "deterministic":
                response = "1"
            elif self.mode == "random":
                response = "1" if torch.rand(1).item() > 0.5 else "2"
            elif self.mode == "alternating":
                response = "1" if self.call_count % 2 == 0 else "2"
            else:
                response = "1"
            
            responses.append(response)
        
        self.call_count += 1
        return responses


def load_mock_model(model_name: str = "mock") -> MockChatWrapper:
    """
    Load a mock model for testing.
    
    Args:
        model_name: Mock model specification
            - "mock" or "mock-deterministic": Always chooses "1"
            - "mock-random": Random choices
            - "mock-alternating": Alternates between "1" and "2"
            
    Returns:
        MockChatWrapper instance
    """
    # Parse mode from model name
    if model_name == "mock":
        mode = "deterministic"
    elif model_name.startswith("mock-"):
        mode = model_name.replace("mock-", "")
    else:
        mode = "deterministic"
    
    # Validate mode
    valid_modes = ["deterministic", "random", "alternating"]
    if mode not in valid_modes:
        raise ValueError(
            f"Invalid mock mode: {mode}. "
            f"Valid modes: {valid_modes}"
        )
    
    return MockChatWrapper(mode=mode)


