import pandas as pd
import os
import torch
from tqdm import tqdm
from typing import List, Optional, Set, Tuple, Dict
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from load_data import load_dataset
from model.load import load_model
from model.base import ChatTemplateWrapper

from prompts import (
    DETECTION_SYSTEM_PROMPT, DETECTION_PROMPT_TEMPLATE_VS_MODEL_BASE_PROMPT, 
    DETECTION_PROMPT_TEMPLATE_VS_MODEL_QUESTION, DETECTION_PROMPT_TEMPLATE_VS_MODEL_BODY
)

from util.elicit import get_choice_token_logits_from_token_ids
from util.util import YamlConfig


def get_choice_tokens(chat_wrapper: ChatTemplateWrapper) -> List[List[int]]:
    """Get token IDs for choice responses "1" and "2"."""
    choice_strings = [["1"], ["2"]]
    choice_tokens = []
    
    for option_str_list in choice_strings:
        option_tokens = []
        for option_str in option_str_list:
            token_ids = chat_wrapper.tokenizer.encode(option_str, add_special_tokens=False)
            if len(token_ids) != 1:
                raise ValueError(
                    f"Choice token '{option_str}' produces {len(token_ids)} tokens: {token_ids}. "
                    f"All choice tokens must be exactly one token."
                )
            option_tokens.extend(token_ids)
        choice_tokens.append(option_tokens)
    
    return choice_tokens


def create_mock_summaries(articles: List[str], num_summaries_per_article: int = 3) -> Dict[int, List[str]]:
    """
    Create mock summaries for testing purposes.
    In a real scenario, these would be loaded from your generated summaries.
    """
    mock_summaries = {}
    
    for i, article in enumerate(articles):
        # Create different "styles" of summaries for the same article
        summaries = []
        
        # Style 1: Short and punchy
        summaries.append(f"BREAKING: {article[:50]}... Major developments unfolding.")
        
        # Style 2: Formal and detailed  
        summaries.append(f"Recent analysis indicates significant developments regarding {article[:30]}... with broader implications for stakeholders.")
        
        # Style 3: Casual and conversational
        summaries.append(f"So here's what's happening: {article[:40]}... Pretty wild stuff!")
        
        mock_summaries[i] = summaries[:num_summaries_per_article]
    
    return mock_summaries


def test_pairwise_choices(
    chat_wrapper: ChatTemplateWrapper,
    test_data: pd.DataFrame,
    max_documents: int = 5,
    max_pairs_per_document: int = 3
) -> None:
    """
    Test pairwise choice elicitation on a small sample.
    
    Args:
        chat_wrapper: Loaded model wrapper
        test_data: DataFrame with columns [document_idx, article, summary]
        max_documents: Maximum number of documents to process
        max_pairs_per_document: Maximum number of pairs to test per document
    """
    
    print(f"Testing on {min(max_documents, len(test_data))} documents with max {max_pairs_per_document} pairs each")
    
    # Get choice tokens
    choice_tokens = get_choice_tokens(chat_wrapper)
    print(f"Choice tokens: {choice_tokens}")
    
    # Create mock summaries for testing
    articles = test_data['article'].tolist()[:max_documents]
    mock_summaries = create_mock_summaries(articles, num_summaries_per_article=3)
    
    results = []
    
    for idx, row in tqdm(test_data.head(max_documents).iterrows(), 
                        total=min(max_documents, len(test_data)), 
                        desc="Testing pairwise choices"):
        
        document_idx = row['document_idx']
        article = row['article']
        
        # Get mock summaries for this document
        summaries = mock_summaries[document_idx]
        
        if len(summaries) < 2:
            continue
            
        print(f"\n--- Document {document_idx} ---")
        print(f"Article: {article[:100]}...")
        print(f"Number of summaries: {len(summaries)}")
        
        # Test a few pairs
        pairs_tested = 0
        for i in range(len(summaries)):
            for j in range(i + 1, len(summaries)):
                if pairs_tested >= max_pairs_per_document:
                    break
                    
                summary_1 = summaries[i]
                summary_2 = summaries[j]
                
                print(f"\nPair {pairs_tested + 1}: Summary {i+1} vs Summary {j+1}")
                print(f"Summary 1: {summary_1}")
                print(f"Summary 2: {summary_2}")
                
                # Create forward prompt
                forward_prompt_raw = (
                    DETECTION_PROMPT_TEMPLATE_VS_MODEL_BODY.format(summary_1=summary_1, summary_2=summary_2) 
                    + DETECTION_PROMPT_TEMPLATE_VS_MODEL_QUESTION
                )
                # For models without chat templates, use simple concatenation
                forward_prompt_full = DETECTION_SYSTEM_PROMPT + "\n\n" + DETECTION_PROMPT_TEMPLATE_VS_MODEL_BASE_PROMPT.format(article=article) + forward_prompt_raw
                
                # Create backward prompt
                backward_prompt_raw = (
                    DETECTION_PROMPT_TEMPLATE_VS_MODEL_BODY.format(summary_1=summary_2, summary_2=summary_1) 
                    + DETECTION_PROMPT_TEMPLATE_VS_MODEL_QUESTION
                )
                # For models without chat templates, use simple concatenation
                backward_prompt_full = DETECTION_SYSTEM_PROMPT + "\n\n" + DETECTION_PROMPT_TEMPLATE_VS_MODEL_BASE_PROMPT.format(article=article) + backward_prompt_raw
                
                try:
                    # Get model predictions
                    forward_probs = get_choice_token_logits_from_token_ids(
                        chat_wrapper.forward(chats=[forward_prompt_full]).logits, 
                        choice_tokens
                    )
                    backward_probs = get_choice_token_logits_from_token_ids(
                        chat_wrapper.forward(chats=[backward_prompt_full]).logits, 
                        choice_tokens
                    )
                    
                    # Display results
                    print(f"Forward (1 vs 2): {forward_probs[0,0].item():.3f} vs {forward_probs[0,1].item():.3f}")
                    print(f"Backward (2 vs 1): {backward_probs[0,0].item():.3f} vs {backward_probs[0,1].item():.3f}")
                    
                    # Store results
                    results.append({
                        'document_idx': document_idx,
                        'summary1_idx': i,
                        'summary2_idx': j,
                        'forward_prob_1': forward_probs[0,0].item(),
                        'forward_prob_2': forward_probs[0,1].item(),
                        'backward_prob_1': backward_probs[0,0].item(),
                        'backward_prob_2': backward_probs[0,1].item(),
                    })
                    
                except torch.OutOfMemoryError:
                    print("Out of memory - skipping this pair")
                    continue
                except Exception as e:
                    print(f"Error processing pair: {e}")
                    continue
                
                pairs_tested += 1
                
                # Clear cache to prevent memory issues
                torch.cuda.empty_cache()
    
    # Save results
    if results:
        results_df = pd.DataFrame(results)
        results_file = "test_pairwise_results.csv"
        results_df.to_csv(results_file, index=False)
        print(f"\nResults saved to {results_file}")
        print(f"Processed {len(results)} pairs across {len(set(r['document_idx'] for r in results))} documents")
    else:
        print("No results to save")


if __name__ == "__main__":
    print("=== Pairwise Choice Test Script ===")
    print("This is a lightweight version for testing with small samples")
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python pairwise_hf_test.py <config_path> [max_documents] [max_pairs]")
        print("Example: python pairwise_hf_test.py configs/llama_3_style_test.yaml 3 2")
        sys.exit(1)
    
    config_path = sys.argv[1]
    max_documents = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    max_pairs = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    
    print(f"Config: {config_path}")
    print(f"Max documents: {max_documents}")
    print(f"Max pairs per document: {max_pairs}")
    
    # Load config
    args = YamlConfig(config_path)
    
    # Load model
    print(f"\nLoading model: {args.model_name}")
    chat_wrapper = load_model(args.model_name, device='auto')
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    _, test_data, _ = load_dataset(args.dataset, splits=['test'])
    print(f"Loaded {len(test_data)} test documents")
    
    # Run test
    test_pairwise_choices(
        chat_wrapper=chat_wrapper,
        test_data=test_data,
        max_documents=max_documents,
        max_pairs_per_document=max_pairs
    )
    
    print("\nTest complete!")
