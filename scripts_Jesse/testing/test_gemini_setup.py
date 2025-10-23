#!/usr/bin/env python3
"""
Test script to verify Google Gemini integration works correctly.
"""

import os
from dotenv import load_dotenv
from model.gemini import load_gemini_model

def test_gemini_setup():
    """Test that Gemini setup works correctly."""
    print("=== Testing Google Gemini Setup ===")
    
    # Load environment variables
    load_dotenv()
    
    # Check if API key is set
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY not found in environment variables")
        print("Please add your Google API key to the .env file:")
        print("GOOGLE_API_KEY=your_api_key_here")
        return False
    
    print(f"✅ API key found: {api_key[:10]}...")
    
    try:
        # Load the model
        print("Loading Gemini model...")
        chat_wrapper = load_gemini_model("gemini-1.5-flash")
        print("✅ Model loaded successfully")
        
        # Test format_chat
        print("Testing format_chat...")
        formatted = chat_wrapper.format_chat(
            system_prompt="You are a helpful assistant.",
            user_message="Hello, how are you?"
        )
        print("✅ format_chat works")
        
        # Test choice tokens
        print("Testing choice tokens...")
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from run_experiment import get_choice_tokens
        choice_tokens = get_choice_tokens(chat_wrapper)
        print(f"Choice tokens: {choice_tokens}")
        print("✅ Choice tokens work")
        
        # Test a simple forward pass
        print("Testing forward pass...")
        result = chat_wrapper.forward(["test"])
        print(f"Forward result logits shape: {result.logits.shape}")
        print("✅ Forward pass works")
        
        print("\n🎉 All tests passed! Gemini integration is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_gemini_setup()
    exit(0 if success else 1)
