from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

# Load the tokenized dataset
tokenized_dataset = load_from_disk("shakespeare_tokenized")

# Load the model for causal language modelling (next-token prediction)
# Contains weights from its pretraining (Reddit, Wikipedia etc)
# We will then fine-tune it using Tiny Shakespeare
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token

# Epoch = going through the entire dataset once
# More epochs = model learns more, risks overfitting (memorising > generalising)

# Batch size = one line at a time is too slow, groups them and calculates an average error
# Higher quantity batch = more CPU/GPU power required

# Learning rate = controls how big each update step is
# Higher learning rate = big leaps (might overshoot)
# Low learning rate = baby steps (slower, but safer)

# Weight decay = prevents model from memorising every detail so weights do not grow too large

# Evaluation strategy = decides when to run the validation dataset
# Set to Epoch = after each full pass through training data, check how it's doing (like an exam)

# Output dir/logging dir = "./logs" is where training metrics go

training_args = TrainingArguments(
    output_dir="./results",          # Where to save model checkpoints
    eval_strategy="epoch",     # Run evaluation after each epoch
    learning_rate=2e-5,              # Step size for weight updates
    per_device_train_batch_size=32,   # How many samples per batch
    num_train_epochs=1,              # Number of passes through dataset
    weight_decay=0.01,               # Regularization to prevent overfitting
    logging_dir="./logs"             # Where to store logs
)

# Build the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"]
)

# Train the model
trainer.train()

# Save the model
trainer.save_model("./shakespeare_model")
tokenizer.save_pretrained("./shakespeare_model")

