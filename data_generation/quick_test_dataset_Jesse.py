"""
Quick Dataset Generation Test for IDE

Super simple script for quick testing in the IDE.
Just change the settings at the top and run!

QUICK CONFIGURATION:
- Change NUM_ARTICLES to test with different numbers
- Change MODEL_TYPE and MODEL_NAME for different models
- Run the script directly in your IDE
"""

import pandas as pd
import os
from tqdm import tqdm

# Import model wrappers
from model.anthropic import load_anthropic_model, AnthropicWrapper
from model.gemini import load_gemini_model, GeminiWrapper
from model.openai import load_openai_model, OpenAIWrapper
from model.load import load_model

# Import prompts
from prompts.prompts_assist_tag import SYSTEM_PROMPT_TEMPLATE, USER_PROMPT_TEMPLATE

# ===== QUICK CONFIGURATION =====
NUM_ARTICLES = 20  # Change this number to test with different amounts
MODEL_TYPE = "openai"  # "anthropic", "google", "openai", or "huggingface"
MODEL_NAME = "gpt-4.1-2025-04-14" 
# ===== END CONFIGURATION =====

def main():
    print(f"🚀 Quick Test: {NUM_ARTICLES} articles with {MODEL_NAME}")
    print("=" * 50)
    
    # Load articles
    print("📚 Loading articles...")
    articles_df = pd.read_csv("results_and_data/data/WikiSum/wikisum_0_20.csv")
    print(f"✅ Loaded {len(articles_df)} articles")
    
    # Take only the number we want to test
    test_articles = articles_df.head(NUM_ARTICLES)
    print(f"🎯 Testing with {len(test_articles)} articles")
    
    # Load model
    print(f"🤖 Loading {MODEL_TYPE} model: {MODEL_NAME}")
    if MODEL_TYPE == "anthropic":
        model_wrapper = load_anthropic_model(MODEL_NAME)
    elif MODEL_TYPE == "google":
        model_wrapper = load_gemini_model(MODEL_NAME)
    elif MODEL_TYPE == "openai":
        model_wrapper = load_openai_model(MODEL_NAME)
    elif MODEL_TYPE == "huggingface":
        model_wrapper = load_model(MODEL_NAME, device='auto')
    else:
        raise ValueError(f"Unknown model type: {MODEL_TYPE}")
    
    print("✅ Model loaded successfully")
    
    # Generate summaries
    print(f"\n🔄 Generating summaries...")
    results = []
    
    for i, (_, article) in enumerate(tqdm(test_articles.iterrows(), desc="Processing")):
        print(f"\n📝 Article {i+1}: {article['title'][:50]}...")
        
        try:
            # Create prompt
            user_prompt = USER_PROMPT_TEMPLATE.format(passage=article['text'])
            
            # Format conversation
            if isinstance(model_wrapper, (AnthropicWrapper, GeminiWrapper, OpenAIWrapper)):
                conversation = model_wrapper.format_chat(
                    system_prompt=SYSTEM_PROMPT_TEMPLATE,
                    user_message=user_prompt
                )
                response = model_wrapper.generate([conversation], max_new_tokens=500, temperature=0.7)
                summary = response[0] if response else ""
            else:
                conversation = model_wrapper.format_chat(
                    system_prompt=SYSTEM_PROMPT_TEMPLATE,
                    user_message=user_prompt
                )
                response = model_wrapper.generate([conversation], max_new_tokens=500, temperature=0.7)
                summary = response[0] if response else ""
            
            # Store result
            results.append({
                'id': article['id'],
                'title': article['title'],
                'text': article['text'],
                'model_summary': summary.strip()
            })
            
            print(f"✅ Generated: {summary[:100]}...")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                'id': article['id'],
                'title': article['title'],
                'text': article['text'],
                'model_summary': ""
            })
    
    # Save results
    print(f"\n💾 Saving results...")
    model_dir = f"results_and_data/data/WikiSum/{MODEL_NAME}"
    os.makedirs(model_dir, exist_ok=True)
    
    df = pd.DataFrame(results)
    output_file = os.path.join(model_dir, "dataset.csv")
    df.to_csv(output_file, index=False)
    
    print(f"✅ Saved to: {output_file}")
    print(f"📊 Generated {len(results)} summaries")
    
    # Show results
    print(f"\n📋 Results:")
    for i, result in enumerate(results):
        print(f"  {i+1}. {result['id']}: {result['title'][:40]}...")
        print(f"     Summary: {result['model_summary'][:100]}...")
        print()
    
    print("🎉 Quick test completed!")

if __name__ == "__main__":
    main()
