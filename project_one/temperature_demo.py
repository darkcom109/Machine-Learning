from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

input_text = "Once upon a time,"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

for temp in [0.3, 0.7, 1.2]:
    print(f"\n--- Temperature {temp} ---")
    output_ids = model.generate(
        input_ids,
        max_length=40,
        do_sample=True,
        temperature=temp,
        top_k=50,
    )
    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))

# --- Temperature 0.3 ---
# The first time the world was in chaos, 
# the world was in a state of chaos.

# --- Temperature 0.7 ---
# Once upon a time, the first time a child has ever been born, 
# they will have their eyes on the young girl.

# --- Temperature 1.2 ---
# Once upon a time, a thousand other people are in mourning. 
# After a thousand deaths, they are saying goodbye, 
# or it will have to be a couple of hundred more years.