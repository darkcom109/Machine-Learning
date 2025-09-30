from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

# Prompt batching required padding, this is where all sequences in the batch must be the same length
# This is because it cannot handle elements of different lengths
# The padding is usually blanks
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained("distilgpt2")

prompts = ["Once upon a time,", "Dear diary,", "In a galaxy far away,"]

batch = tokenizer(prompts, return_tensors="pt", padding=True)

# {batch["input_ids"]} are the tokenized prompts, returns a 2D tensor
# {batch["attention_mask"]} is a mask that tells the model which tokens are real words and which are padding
# Uses 1 (not padding) and 0 (padding)
# {do_sample=True} changes the generation style, 
# False = model picks the most likely next token, more deterministic,
# True = model samples from the probability distribution, more creativity
outputs = model.generate(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],  # 👈 now safe to use
    max_length=40,
    do_sample=True
)

# Pythonic for loop with enumerate
# i = index, out = actual element
for i, out in enumerate(outputs):
    print(f"\nPrompt: {prompts[i]}")
    print(tokenizer.decode(out, skip_special_tokens=True))

# First input - seemed to have hallucinated
# Prompt: Once upon a time,
# Once upon a time," "Taken" , "Fooled" , "Gryst" , } ( $key , value ) { } function ( $k , $value

# Second input - imitating a personal writing style, weird symbols such as !! are Unicode artifacts
# Prompt: Dear diary,
# Dear diary,I‼m not sure what the last sentence was,‑–      

# Third input - lacks strong long-term affection
# Prompt: In a galaxy far away,
# In a galaxy far away, I was in a war cry.
# And it was the first time I ever met a young girl. Even though there was a galaxy far away, I was in a
