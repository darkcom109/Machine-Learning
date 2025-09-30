from transformers import AutoTokenizer

# Loads the pre-trained tokenizer for the distilgpt2 model
# This tokenizer knows how to split English text into tokens
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

text = "Once upon a time, there was a dragon"

# Encoding - splits input/text into tokens and maps each token to an integer ID that the model understands
# For example - "Once upon a time, there was a dragon" → [2061, 6037, 257, 640, 11, 612, 373, 257, 12309]
tokens = tokenizer.encode(text)
print("Token IDs:", tokens)

# Decoding - takes the list of token IDs and converts them back into the original text string
decoded = tokenizer.decode(tokens)
print("Decoded text:", decoded)

# encode = assigned an ID according to the string
# decode = converts to a string using the ID
for token_id in tokens:
    print(token_id, "->", tokenizer.decode([token_id]))