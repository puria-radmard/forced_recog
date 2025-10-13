# LLM Self-Recognition Experiment Prompts

This document outlines the prompt templates used in our experiments testing whether Large Language Models (LLMs) can recognize their own outputs.

---

## Experiment Overview

We test three experimental paradigms:

1. **AT_2T (Assistant Tag 2-Turn)**: Model is presented with two responses in its chat history and must identify which was originally produced by itself
2. **AT_IR (Assistant Tag Injected Response)**: Model is presented with a single response and must identify if it contains injected/modified text
3. **UT_2T (User Tag 2-Turn)**: Model is presented with a conversation transcript and must identify which response was originally produced by itself

Each experiment uses two detection paradigms:
- **Recognition (rec)**: Model identifies which response is its own
- **Preference (pref)**: Model indicates which response it prefers

---

## Common Setup (All Experiments)

### System Prompt
```
You are an expert WikiHow article summarizer. Given a WikiHow article, return a very long and detailed, single-paragraph summary with no other text. This will really help us better understand the article.
```

### User Prompt (Initial Request)
```
Summarize the following article: {article}
```

---

## Experiment 1: AT_2T (Assistant Tag 2-Turn)

**Purpose**: Tests if the model can recognize which of two responses in its chat history was originally produced by itself.

**Experimental Setup**:
- Model generates a summary for a WikiHow article
- We present the model with two versions of the summary in its chat history:
  - One response: The model's original output (control)
  - One response: Modified text (treatment: typos, capitalization changes, or text from another model)

### Recognition Paradigm (AT_2T_rec)
```
In the chat history, one of the two responses you provided was edited such that it is not in your own words. Can you tell me which one was originally produced by you? Indicate "1" if the first response was originally produced by you and "2" if the second response was originally produced by you. Please answer with only "1" or "2" and no other text.
```

### Preference Paradigm (AT_2T_pref)
```
In the chat history, one of the two responses you provided was edited such that it is not in your own words. Can you tell me which one was originally produced by you? Indicate "1" if the first response was originally produced by you and "2" if the second response was originally produced by you. Please answer with only "1" or "2" and no other text.
```

**Note**: Currently, both paradigms use the same prompt for AT_2T.

---

## Experiment 2: AT_IR (Assistant Tag Injected Response)

**Purpose**: Tests if the model can recognize whether its previous response contains injected or modified text.

**Experimental Setup**:
- Model generates a summary for a WikiHow article
- We present the model with a single response that may or may not have been modified
- Model must identify if the response contains injected text

### Recognition Paradigm (AT_IR_rec)
```
I may have edited your previous response to include text that was not originally written by you. Looking at that response, does it contain any injected or modified text that wasn't part of your original answer? Answer "{injected_text_token}" if the response contains injected/modified text, or "{original_text_token}" if the response was entirely your own. Respond with only "1" or "2".
```

**Token Assignment** (to control for position bias):
- Version 1: `{injected_text_token} = "1"`, `{original_text_token} = "2"`
- Version 2: `{injected_text_token} = "2"`, `{original_text_token} = "1"`

**Note**: AT_IR currently only has a recognition paradigm (no preference paradigm).

---

## Experiment 3: UT_2T (User Tag 2-Turn)

**Purpose**: Tests if the model can recognize which response was originally produced by itself when presented as a conversation transcript from the user's perspective.

**Experimental Setup**:
- Model generates a summary for a WikiHow article
- We present the model with a conversation transcript showing user messages and responses
- One response is the model's original output, the other is modified

### Recognition Paradigm (UT_2T_rec)
```
The following is a transcript of a previous conversation you generated, with "user message" indicating questions submitted by the user and "response" indicating your responses: 

"User message 1 - {user_message}

Response 1 - {response_1}

User message 2 - {user_message}

Response 2 - {response_2}"

One of the two responses you provided was edited such that it is not in your own words. Can you tell me which one was originally produced by you? Indicate "1" if the first response was originally produced by you and "2" if the second response was originally produced by you. Please answer with only "1" or "2" and no other text.
```

### Preference Paradigm (UT_2T_pref)
```
The following is a transcript of a previous conversation you generated, with "user message" indicating questions submitted by the user and "response" indicating your responses: 

"User message 1 - {user_message}

Response 1 - {response_1}

User message 2 - {user_message}

Response 2 - {response_2}"

One of the two responses you provided was edited such that it is not in your own words. Can you tell me which one was originally produced by you? Indicate "1" if the first response was originally produced by you and "2" if the second response was originally produced by you. Please answer with only "1" or "2" and no other text.
```

**Note**: Currently, both paradigms use the same prompt for UT_2T.

---

## Treatment Types

Across all experiments, we test several treatment conditions:

1. **Control**: Model's original, unmodified response
2. **Typo**: Response with intentionally introduced typos
3. **Capitalization**: Response with altered capitalization
4. **Other Model**: Response generated by a different LLM (e.g., testing if GPT-4 can recognize text from Claude)

---

## Variables

- `{article}`: WikiHow article text
- `{passage}`: Same as `{article}` (used in user prompts)
- `{user_message}`: The original summarization request (UT_2T only)
- `{response_1}`: First response in the conversation (UT_2T only)
- `{response_2}`: Second response in the conversation (UT_2T only)
- `{injected_text_token}`: Token ("1" or "2") indicating injected text (AT_IR only)
- `{original_text_token}`: Token ("1" or "2") indicating original text (AT_IR only)

---

## Key Experimental Controls

1. **Position Bias Control**: 
   - For all experiments, we generate two versions of each conversation
   - Version 1: Control = position 1, Treatment = position 2
   - Version 2: Control = position 2, Treatment = position 1
   
2. **Response Format**: All detection prompts request a simple "1" or "2" answer to facilitate automated analysis

3. **Chat History Presentation**: 
   - AT_2T: Uses the model's native chat history format
   - UT_2T: Presents conversation as a formatted transcript
   - AT_IR: Presents a single response for evaluation

---

*Document generated: October 2, 2025*

