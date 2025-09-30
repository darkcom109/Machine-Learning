# Importing the Hugging Face pipeline function.
# Pipelines can be thought of as a pre-built shortcut, bundling together the messy parts
# such as tokenising text, feeding it into the model and decoding the output
# without it, you would have to manually load it yourself

from transformers import pipeline

# This is a pipeline object
# @param "text-generation" is used to just generate text
# @param "distilgpt2" is used to determine which model to use

generator = pipeline("text-generation", model="distilgpt2")

# This object then calls the model
# @param {prompt} is what you input into the object
# @param {max_length} is the maximum number of tokens in the entire sequence
# @param {num_return_sequences} is the quantity of completions to generate

print(generator("Once upon a time,", max_length=30, num_return_sequences=1))