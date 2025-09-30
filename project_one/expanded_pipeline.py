# {AutoTokenizer} loads the correct tokenizer for a model
# {AutoModelForCausalLM} loads a model meant for causal language modelling,
# this predicts the next token given the previous ones
from transformers import AutoTokenizer, AutoModelForCausalLM

# Loads the pre-trained tokenizer for the distilgpt2 model
# This tokenizer knows how to split English text into tokens
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

# Loads the pre-trained DistilGPT-2 model weights
# DistilGPT-2 is a shrunk version of GPT-2 so it runs faster but loses some capacity
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# Prompt
input_text = "Once upon a time,"

# Encodes the raw text into tokens
# @param {return_tensors} returns the result as a PyTorch tensor so the model can process it
# A tensor is a generalisation of numbers, vectors and matrices
# OD tensor = a single number (scalar), 1D tensor = a vector (list of numbers), 2D tensor = a matrix (rows and columns)
# 3D tensor = a cube of numbers, nD = high dimensions (only handled by computers)
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# The heavy lifting
# {model.generate()} = tells the model to generate new tokens given your prompt
# {input_ids} = your prompt tokens
# {max_length} = total length of prompt + generated text, cannot exceed this
# {num_return_sequences} = only create 1 completion
# {do_sample} = sample words according to probabilities
# {temperature} = controls randomness, <1 more deterministic and >1 more random 
# {top_k} = only sample from the top 50 most likely next tokens
# output_ids is a tensor of integers representing your generated sequence
output_ids = model.generate(
    input_ids,
    max_length=30,
    num_return_sequences=3,
    do_sample=True,
    temperature=0.7,
    top_k=50,
)

# Converts the tensor of number back into human-readable text
# output[0] obtains the first index, skip_special_tokens=True strips out weird placeholders
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Outputs response
print(output_text)

# Flow
# Text -> Tokenizer -> Numbers -> Model generates new numbers -> Tokenizer decodes -> Text