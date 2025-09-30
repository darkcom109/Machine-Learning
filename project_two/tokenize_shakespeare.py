"""
tokenize_shakespeare.py

This script loads the Shakespeare dataset we built in load_shakespeare.py,
and turns each line of text into token IDs using DistilGPT-2's tokenizer.

Steps:
1. Load the saved dataset (train/validation).
2. Load the DistilGPT-2 tokenizer.
3. Define a tokenization function.
4. Apply tokenization to the whole dataset.
5. Preview results.
"""

# Step 1: Import libraries
from datasets import load_from_disk          # To load the dataset saved in load_shakespeare.py
from transformers import AutoTokenizer       # To handle encoding text into tokens

# Step 2: Load the dataset from disk
# Make sure you've run load_shakespeare.py first so "shakespeare_data" exists
dataset = load_from_disk("shakespeare_data")
print("Original dataset structure:\n", dataset)

# Step 3: Load DistilGPT-2 tokenizer
# - This is the same tokenizer the model was trained with
# - It knows how to split text into tokens and map tokens → numbers
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

# GPT-2 doesn’t have a pad token by default, so we set it = eos_token (end-of-sequence)
tokenizer.pad_token = tokenizer.eos_token

# Step 4: Define a tokenization function
def tokenize_function(example):
    """
    Takes one example (a row with a "text" field) and tokenizes it.
    - truncation=True: cut off text if it’s longer than max_length
    - padding="max_length": pad shorter texts to exactly max_length
    - max_length=128: each sequence is fixed at 128 tokens
    """
    tokens = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=64
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

# Step 5: Apply tokenizer to the entire dataset
# batched=True → processes multiple rows at once (faster)
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Step 6: Preview the tokenized dataset
print("\nTokenized dataset structure:\n", tokenized_dataset)

print("\nExample (first training row):")
print(tokenized_dataset["train"][0])   # Shows text + input_ids + attention_mask

# Step 7: Save the tokenized dataset (for training later)
tokenized_dataset.save_to_disk("shakespeare_tokenized")
print("\nSaved tokenized dataset to 'shakespeare_tokenized'")
