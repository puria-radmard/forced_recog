#!/usr/bin/env python3
"""
Test script to verify all model integrations work correctly.
"""

import os
from dotenv import load_dotenv

def test_anthropic():
    """Test Anthropic Claude integration."""
    print("=== Testing Anthropic Claude ===")
    
    try:
        from model.anthropic import load_anthropic_model
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from run_experiment import get_choice_tokens
        
        # Check API key
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("❌ ANTHROPIC_API_KEY not found")
            return False
        
        # Load model
        chat_wrapper = load_anthropic_model("claude-3-5-haiku-20241022")
        
        # Test basic functionality
        formatted = chat_wrapper.format_chat(
            system_prompt="You are a helpful assistant.",
            user_message="Hello!"
        )
        
        choice_tokens = get_choice_tokens(chat_wrapper)
        result = chat_wrapper.forward(["test"])
        
        print("✅ Anthropic Claude works")
        return True
        
    except Exception as e:
        print(f"❌ Anthropic error: {e}")
        return False

def test_gemini():
    """Test Google Gemini integration."""
    print("=== Testing Google Gemini ===")
    
    try:
        from model.gemini import load_gemini_model
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from run_experiment import get_choice_tokens
        
        # Check API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("❌ GOOGLE_API_KEY not found")
            return False
        
        # Load model
        chat_wrapper = load_gemini_model("gemini-1.5-flash")
        
        # Test basic functionality
        formatted = chat_wrapper.format_chat(
            system_prompt="You are a helpful assistant.",
            user_message="Hello!"
        )
        
        choice_tokens = get_choice_tokens(chat_wrapper)
        result = chat_wrapper.forward(["test"])
        
        print("✅ Google Gemini works")
        return True
        
    except Exception as e:
        print(f"❌ Gemini error: {e}")
        return False

def test_huggingface():
    """Test HuggingFace integration."""
    print("=== Testing HuggingFace ===")
    
    try:
        from model.load import load_model
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from run_experiment import get_choice_tokens
        
        # Load model
        chat_wrapper = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0", device='auto')
        
        # Test basic functionality
        formatted = chat_wrapper.format_chat(
            system_prompt="You are a helpful assistant.",
            user_message="Hello!"
        )
        
        choice_tokens = get_choice_tokens(chat_wrapper)
        result = chat_wrapper.forward(["test"])
        
        print("✅ HuggingFace works")
        return True
        
    except Exception as e:
        print(f"❌ HuggingFace error: {e}")
        return False

def main():
    """Test all model integrations."""
    print("=== Testing All Model Integrations ===\n")
    
    load_dotenv()
    
    results = {}
    results['anthropic'] = test_anthropic()
    print()
    results['gemini'] = test_gemini()
    print()
    results['huggingface'] = test_huggingface()
    
    print("\n=== Summary ===")
    for model_type, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{model_type.upper()}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All model integrations are working!")
    else:
        print("\n⚠️  Some model integrations have issues.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
