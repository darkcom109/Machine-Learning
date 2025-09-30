"""
load_shakespeare.py

This script downloads the Tiny Shakespeare dataset (a ~1MB text file),
wraps it in a Hugging Face Dataset object, and previews some examples.

Why not use load_dataset("tiny_shakespeare") directly?
- The Hugging Face Datasets library changed and now blocks some dataset scripts.
- To avoid version errors, we just fetch the raw text file ourselves.

Steps:
1. Download Tiny Shakespeare from Karpathy's GitHub (raw .txt file).
2. Wrap it into a Hugging Face Dataset.
3. (Optional) Split into train/validation for later training.
4. Print out previews so we know what we’re working with.
"""

# Import libraries
import requests                     # To download the text file from the web
from datasets import Dataset        # To build a dataset object from raw data
from datasets import DatasetDict    # To organize train/validation splits

# Step 1: Download the raw text
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
response = requests.get(url)         # Fetch the file
raw_text = response.text             # Extract text content as a string

# Step 2: Split the raw text into lines
# Each line is one training example (dialogue line, stage direction, etc.)
lines = raw_text.split("\n")

# Step 3: Create a Hugging Face Dataset from the lines
# Hugging Face wants data in a Dataset object, 1 item dictionary containing a list of all lines
dataset = Dataset.from_dict({"text": lines})

# Step 4: Split into train and validation sets
# Models need training data (to learn) and validation data (to check if it is learning or memorising)
# {test_size=0.1} - 10% of lines go into validation
# {shuffle=True} - mixes lines before splitting so validation isn't just one play
dataset = dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)

# Wrap it into a DatasetDict for consistency with Hugging Face conventions
# dataset["train"] -> ~90k rows | dataset["validation"] -> ~10k rows
dataset = DatasetDict({
    "train": dataset["train"],
    "validation": dataset["test"]
})

# Step 5: Preview the dataset structure
print(dataset)

# Step 6: Print the first few training examples
print("\nFirst few training samples:")
for i in range(3):
    print(dataset["train"][i])

# Save to disk for future usage
dataset.save_to_disk("shakespeare_data")