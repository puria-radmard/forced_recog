"""
IDE Dataset Generator - Easy Configuration

This script is designed for easy IDE testing with simple configuration.
Just modify the CONFIG dictionary at the top and run!

FEATURES:
- Easy configuration at the top of the file
- Multiple model support
- Configurable article count
- Detailed progress output
- Error handling
- Results summary
"""

import pandas as pd
import os
from tqdm import tqdm
from typing import Dict, Any, List

# Import model wrappers
from model.anthropic import load_anthropic_model, AnthropicWrapper
from model.gemini import load_gemini_model, GeminiWrapper
from model.load import load_model

# Import prompts
from prompts.prompts_assist_tag import SYSTEM_PROMPT_TEMPLATE, USER_PROMPT_TEMPLATE

# ===== CONFIGURATION =====
# Modify these settings as needed

CONFIG = {
    # Model settings
    "model_type": "google",  # Options: "anthropic", "google", "huggingface"
    "model_name": "gemini-2.5-flash",  # The actual model name
    
    # Data settings
    "num_articles": 3,  # Number of articles to process (None for all)
    "articles_file": "results_and_data/data/WikiSum/wikisum_0_20.csv",
    "output_dir": "results_and_data/data/WikiSum",
    
    # Display settings
    "show_detailed_output": True,  # Show detailed progress
    "show_sample_results": True,  # Show sample results at the end
    
    # Generation settings
    "max_new_tokens": 500,  # Maximum tokens for summary
    "temperature": 0.7,  # Temperature for generation
}

# ===== END CONFIGURATION =====

class IDEDatasetGenerator:
    """IDE-friendly dataset generator."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_wrapper = None
        self.results = []
    
    def load_articles(self) -> pd.DataFrame:
        """Load articles from CSV file."""
        print(f"📚 Loading articles from {self.config['articles_file']}...")
        
        if not os.path.exists(self.config['articles_file']):
            raise FileNotFoundError(f"Articles file not found: {self.config['articles_file']}")
        
        df = pd.read_csv(self.config['articles_file'])
        print(f"✅ Loaded {len(df)} articles")
        
        # Limit articles if specified
        if self.config['num_articles']:
            df = df.head(self.config['num_articles'])
            print(f"🎯 Processing {len(df)} articles (limited by num_articles)")
        
        return df
    
    def load_model(self):
        """Load the specified model."""
        print(f"🤖 Loading {self.config['model_type']} model: {self.config['model_name']}")
        
        if self.config['model_type'] == "anthropic":
            self.model_wrapper = load_anthropic_model(self.config['model_name'])
        elif self.config['model_type'] == "google":
            self.model_wrapper = load_gemini_model(self.config['model_name'])
        elif self.config['model_type'] == "huggingface":
            self.model_wrapper = load_model(self.config['model_name'], device='auto')
        else:
            raise ValueError(f"Unsupported model type: {self.config['model_type']}")
        
        print("✅ Model loaded successfully")
    
    def generate_summaries(self, articles_df: pd.DataFrame) -> List[Dict[str, str]]:
        """Generate summaries for articles."""
        print(f"\n🔄 Generating summaries for {len(articles_df)} articles...")
        
        results = []
        
        for i, (_, article) in enumerate(tqdm(articles_df.iterrows(), total=len(articles_df), desc="Processing")):
            if self.config['show_detailed_output']:
                print(f"\n📝 Processing article {i+1}/{len(articles_df)}: {article['title'][:50]}...")
            
            try:
                # Create user prompt
                user_prompt = USER_PROMPT_TEMPLATE.format(passage=article['text'])
                
                # Format conversation
                if isinstance(self.model_wrapper, (AnthropicWrapper, GeminiWrapper)):
                    conversation = self.model_wrapper.format_chat(
                        system_prompt=SYSTEM_PROMPT_TEMPLATE,
                        user_message=user_prompt
                    )
                    response = self.model_wrapper.generate(
                        [conversation], 
                        max_new_tokens=self.config['max_new_tokens'],
                        temperature=self.config['temperature']
                    )
                    summary = response[0] if response else ""
                else:
                    conversation = self.model_wrapper.format_chat(
                        system_prompt=SYSTEM_PROMPT_TEMPLATE,
                        user_message=user_prompt
                    )
                    response = self.model_wrapper.generate(
                        [conversation], 
                        max_new_tokens=self.config['max_new_tokens'],
                        temperature=self.config['temperature']
                    )
                    summary = response[0] if response else ""
                
                # Store result
                result = {
                    'id': article['id'],
                    'title': article['title'],
                    'text': article['text'],
                    'model_summary': summary.strip()
                }
                results.append(result)
                
                if self.config['show_detailed_output']:
                    print(f"✅ Generated summary: {summary[:100]}...")
                
            except Exception as e:
                print(f"❌ Error processing article {article['id']}: {e}")
                # Add empty summary for failed articles
                results.append({
                    'id': article['id'],
                    'title': article['title'],
                    'text': article['text'],
                    'model_summary': ""
                })
        
        return results
    
    def save_dataset(self, results: List[Dict[str, str]]) -> str:
        """Save the generated dataset."""
        print(f"\n💾 Saving dataset...")
        
        # Create model directory
        model_dir = os.path.join(self.config['output_dir'], self.config['model_name'])
        os.makedirs(model_dir, exist_ok=True)
        
        # Create DataFrame and save
        df = pd.DataFrame(results)
        output_file = os.path.join(model_dir, "dataset.csv")
        df.to_csv(output_file, index=False)
        
        print(f"✅ Dataset saved to: {output_file}")
        print(f"📊 Generated {len(df)} summaries")
        print(f"📊 Non-empty summaries: {len(df[df['model_summary'].str.strip() != ''])}")
        
        return output_file
    
    def show_results(self, results: List[Dict[str, str]]):
        """Show sample results."""
        if not self.config['show_sample_results']:
            return
        
        print(f"\n📋 Sample Results:")
        for i, result in enumerate(results[:3]):  # Show first 3 results
            print(f"  {i+1}. {result['id']}: {result['title'][:50]}...")
            print(f"     Summary: {result['model_summary'][:150]}...")
            print()
    
    def run(self):
        """Run the complete dataset generation process."""
        print("🚀 IDE Dataset Generator")
        print("=" * 50)
        print(f"Model: {self.config['model_type']} - {self.config['model_name']}")
        print(f"Articles: {self.config['num_articles'] or 'All'}")
        print(f"Output: {self.config['output_dir']}")
        print("=" * 50)
        
        try:
            # Load articles
            articles_df = self.load_articles()
            
            # Load model
            self.load_model()
            
            # Generate summaries
            results = self.generate_summaries(articles_df)
            
            # Save dataset
            output_file = self.save_dataset(results)
            
            # Show results
            self.show_results(results)
            
            print(f"\n🎉 Dataset generation completed successfully!")
            print(f"📁 Output file: {output_file}")
            
            return results
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Main function."""
    generator = IDEDatasetGenerator(CONFIG)
    results = generator.run()
    
    if results:
        print(f"\n✅ Success! Generated {len(results)} summaries")
    else:
        print(f"\n❌ Failed to generate dataset")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())


