"""
Google Gemini API Integration for Assist Tag Recognition

This module provides a wrapper for Google Gemini API that inherits from
BaseChatWrapper for unified interface and logging support.
"""

import os
import json
import time
import random
from typing import List, Dict, Optional, Any, Union
import google.generativeai as genai
from dotenv import load_dotenv
from model.base import BaseChatWrapper

# Load environment variables
load_dotenv()

class GeminiChatWrapper(BaseChatWrapper):
    """
    Wrapper for Google Gemini API that inherits from BaseChatWrapper.
    
    This class provides Gemini API integration with unified interface
    and logging support.
    
    [WARNING] This wrapper uses MOCK LOGITS for compatibility.
    The logits returned by forward() are simulated based on text analysis
    and do not represent the model's actual internal probability distributions.
    """
    
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        """
        Initialize the Gemini wrapper.
        
        Args:
            model_name: The Gemini model to use (e.g., "gemini-1.5-flash", "gemini-1.5-pro")
        """
        super().__init__(model_name)
        
        # Configure Gemini API
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
        # Mock tokenizer for compatibility
        self.tokenizer = self._create_mock_tokenizer()
        self.device = "gemini"
        
        # Rate limiting configuration
        self.base_delay = 0.5  # Base delay in seconds (reduced for faster recovery)
        self.max_delay = 30.0  # Maximum delay in seconds (rate limits reset every minute)
        
    def _create_mock_tokenizer(self):
        """Create a mock tokenizer for compatibility with existing code."""
        class MockTokenizer:
            def __init__(self):
                self.pad_token = "<pad>"
                self.eos_token = "<eos>"
                
            def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
                """Mock encode method - returns simple token IDs."""
                # Simple mapping for choice tokens
                if text.strip() == "1":
                    return [1]
                elif text.strip() == "2":
                    return [2]
                else:
                    # For other text, return a simple hash-based token ID
                    return [hash(text) % 1000 + 10]
                    
            def decode(self, token_ids: List[int]) -> str:
                """Mock decode method."""
                if token_ids == [1]:
                    return "1"
                elif token_ids == [2]:
                    return "2"
                else:
                    return f"token_{token_ids[0]}"
        
        return MockTokenizer()
    
    def _is_rate_limit_error(self, error: Exception) -> bool:
        """
        Check if the error is a rate limit error.
        
        Args:
            error: The exception to check
            
        Returns:
            True if it's a rate limit error, False otherwise
        """
        error_str = str(error).lower()
        return any(phrase in error_str for phrase in [
            "rate limit", "rate_limit", "too many requests", "quota exceeded",
            "429", "throttle", "throttled", "limit exceeded", "quota",
            "resource_exhausted", "permission_denied", "quota_exceeded"
        ])
    
    def _calculate_delay(self, attempt: int) -> float:
        """
        Calculate exponential backoff delay with jitter.
        
        Args:
            attempt: The current attempt number (0-based)
            
        Returns:
            Delay in seconds
        """
        # Exponential backoff: base_delay * (2^attempt)
        delay = self.base_delay * (2 ** attempt)
        
        # Add jitter to prevent thundering herd
        jitter = random.uniform(0.1, 0.5) * delay
        
        # Cap at max_delay
        return min(delay + jitter, self.max_delay)
    
    def _make_api_call_with_retry(self, content: str, generation_config: Any, safety_settings: List[Dict]) -> Any:
        """
        Make API call with retry logic for rate limits.
        Retries indefinitely until success or non-rate-limit error.
        
        Args:
            content: The content to generate
            generation_config: Generation configuration
            safety_settings: Safety settings
            
        Returns:
            API response
            
        Raises:
            Exception: If non-rate-limit error occurs
        """
        attempt = 0
        
        while True:
            try:
                response = self.model.generate_content(
                    content,
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )
                return response
                
            except Exception as e:
                if self._is_rate_limit_error(e):
                    delay = self._calculate_delay(attempt)
                    print(f"⚠️  Rate limit hit (attempt {attempt + 1}). Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    attempt += 1
                    continue
                else:
                    # Non-rate-limit error, don't retry
                    raise e
    
    def format_chat(
        self, 
        *_,
        system_prompt: Optional[str] = None,
        in_context_questions: Optional[List[str]] = None,
        in_context_answers: Optional[List[str]] = None,
        user_message: Optional[str] = None,
        prefiller: Optional[str] = None,
        skip_special_tokens: bool = False,
        **kwargs
    ) -> str:
        """
        Format a chat conversation for Gemini API.
        
        Args:
            system_prompt: System instruction
            in_context_questions: List of user questions
            in_context_answers: List of assistant responses
            user_message: Final user message
            prefiller: Optional prefilled response
            skip_special_tokens: Ignored for Gemini
            
        Returns:
            Formatted conversation string
        """
        # Build the conversation content
        content_parts = []
        
        # Add system prompt
        if system_prompt:
            content_parts.append(f"System: {system_prompt}")
        
        # Add in-context Q&A pairs
        if in_context_questions and in_context_answers:
            for question, answer in zip(in_context_questions, in_context_answers):
                content_parts.append(f"Human: {question}")
                content_parts.append(f"Assistant: {answer}")
        
        # Add final user message
        if user_message:
            content_parts.append(f"Human: {user_message}")
        
        # Add prefiller if provided
        if prefiller:
            content_parts.append(f"Assistant: {prefiller}")
        
        # Join all parts
        formatted_content = "\n\n".join(content_parts)
        
        # Store for use in forward method
        self._last_content = formatted_content
        self._last_system_prompt = system_prompt
        
        return formatted_content
    
    def forward(
        self, 
        chats: List[str], 
        past_key_values: Optional[Any] = None,
        return_dict: bool = True,
        use_cache: bool = True,
        return_input_ids: bool = False,
        **forward_kwargs: Any
    ) -> Dict[str, Any]:
        """
        Run forward pass using Gemini API.
        
        Args:
            chats: List of formatted chat strings (ignored, uses last content)
            past_key_values: Ignored for Gemini
            return_dict: Whether to return dictionary output
            use_cache: Ignored for Gemini
            return_input_ids: Whether to return input IDs
            
        Returns:
            Dictionary with logits for choice tokens
        """
        if not hasattr(self, '_last_content'):
            raise ValueError("No content available. Call format_chat first.")
        
        try:
            # Add delay to avoid rate limiting
            import time
            time.sleep(2)  # 2 second delay between API calls
            
            # Create generation config for short responses
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=10,  # Short response for choice detection
                temperature=0.1,  # Low temperature for consistent responses
            )
            
            # Disable safety filters for testing
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH", 
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE"
                }
            ]
            
            # Make API call to Gemini with retry logic
            response = self._make_api_call_with_retry(
                self._last_content,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            # Extract response text with better error handling
            response_text = ""  # Initialize response_text
            try:
                response_text = response.text.strip() if response.text else ""
            except Exception as text_error:
                print(f"Gemini response text access error: {text_error}")
                # Check if response has candidates and their finish reasons
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'finish_reason'):
                        print(f"Finish reason: {candidate.finish_reason}")
                        if candidate.finish_reason == 2:  # SAFETY
                            print("⚠️  Safety warning - trying to extract text from parts")
                            # Try to extract text from parts even with safety warning
                            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                                for part in candidate.content.parts:
                                    if hasattr(part, 'text'):
                                        response_text = part.text.strip()
                                        print(f"✅ Extracted text from parts: '{response_text}'")
                                        break
                            else:
                                print("❌ No text found in parts")
                                response_text = ""
                        elif candidate.finish_reason == 3:  # RECITATION
                            print("Response blocked by recitation filters")
                            response_text = ""
                        else:
                            response_text = ""
                else:
                    response_text = ""
            
            # Create mock logits for choice tokens
            print(f"🔍 DEBUG: Creating logits for response_text: '{response_text}'")
            logits = self._create_choice_logits(response_text)
            
            # Create mock output similar to HuggingFace format
            class MockOutput:
                def __init__(self, logits):
                    self.logits = logits
            
            return MockOutput(logits)
            
        except Exception as e:
            print(f"Gemini API error: {e}")
            # Return neutral logits on error
            logits = self._create_choice_logits("")
            class MockOutput:
                def __init__(self, logits):
                    self.logits = logits
            return MockOutput(logits)
    
    def _create_choice_logits(self, response_text: str) -> Any:
        """
        Create mock logits for choice tokens based on response.
        
        Args:
            response_text: The response from Gemini
            
        Returns:
            Mock logits tensor
        """
        import torch
        
        # Print warning about mock logits
        print("\033[91m[WARNING] Using MOCK LOGITS for Gemini API model\033[0m")
        print("\033[91m   These are simulated probabilities based on text analysis only.\033[0m")
        print("\033[91m   They do not represent the model's actual confidence or internal state.\033[0m")
        print(f"\033[91m   Response text: '{response_text}'\033[0m")
        
        # Create logits for choice tokens [1, 2]
        # Higher probability for the choice that appears in response
        if "1" in response_text and "2" not in response_text:
            # Response contains "1", higher probability for choice 1
            logits = torch.tensor([[[2.0, 0.0]]])  # [batch, seq, vocab]
        elif "2" in response_text and "1" not in response_text:
            # Response contains "2", higher probability for choice 2
            logits = torch.tensor([[[0.0, 2.0]]])  # [batch, seq, vocab]
        else:
            # Neutral or unclear response
            logits = torch.tensor([[[0.5, 0.5]]])  # [batch, seq, vocab]
        
        # Ensure the logits have the right shape for the choice token extraction
        # The choice tokens are [1, 2], so we need logits at positions 1 and 2
        # Create a larger vocabulary tensor and place our choice logits at the right positions
        vocab_size = 1000  # Mock vocabulary size
        full_logits = torch.zeros((1, 1, vocab_size))  # [batch, seq, vocab]
        
        # Place choice logits at positions 1 and 2
        full_logits[0, 0, 1] = logits[0, 0, 0]  # Choice 1
        full_logits[0, 0, 2] = logits[0, 0, 1]  # Choice 2
        
        return full_logits
    
    def generate(
        self,
        chats: List[str],
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        **kwargs
    ) -> List[str]:
        """
        Generate text using Gemini API.
        
        Args:
            chats: List of formatted chat strings
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty
            **kwargs: Additional arguments
            
        Returns:
            List of generated text strings
        """
        if not hasattr(self, '_last_content'):
            raise ValueError("No content available. Call format_chat first.")
        
        try:
            # Create generation config
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_new_tokens,
                temperature=temperature,
            )
            
            # Disable safety filters for testing
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH", 
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE"
                }
            ]
            
            # Make API call to Gemini with retry logic
            response = self._make_api_call_with_retry(
                self._last_content,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            # Extract response text with better error handling
            try:
                response_text = response.text.strip() if response.text else ""
            except Exception as text_error:
                print(f"Gemini response text access error: {text_error}")
                # Check if response has candidates and their finish reasons
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'finish_reason'):
                        print(f"Finish reason: {candidate.finish_reason}")
                        if candidate.finish_reason == 2:  # SAFETY
                            print("Response blocked by safety filters")
                        elif candidate.finish_reason == 3:  # RECITATION
                            print("Response blocked by recitation filters")
                response_text = ""
            return [response_text]
            
        except Exception as e:
            print(f"Gemini API error: {e}")
            return ["Error generating response"]


def load_gemini_model(model_name: str = "gemini-1.5-flash") -> GeminiChatWrapper:
    """
    Load a Gemini model wrapper.
    
    Args:
        model_name: The Gemini model to use
        
    Returns:
        GeminiChatWrapper instance
    """
    return GeminiChatWrapper(model_name)
