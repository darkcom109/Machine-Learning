# from transformers import Trainer

# Glue model + training args + datasets together

# {model} DistilGPT-2
# {args} training arguments we defined
# {train_dataset} 90% data for training
# {eval_dataset} 10% data for validation
"""trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
)

trainer.train()

# 1) Trainer sets up batches, takes tokenized dataset, groups into batches of 4
# 2) Forward pass, feeds each batch into the model, model predicts the next tokens
# 3) Loss calculation (how wrong the model is), compares model predictions vs actual tokens
# 4) Backward pass, use backpropagation to adjust model weights so it is loss reduces
# 5) Repeat until end of epoch
# 6) At the end of the epoch, it runs a validation set and reports validation loss (how well it generalises)
# 7) Logging + checkpoints, logs progress and saves checkpoints (./results and ./logs)


***** Running training *****
  Num examples = 90181
  Num Epochs = 1
  Instantaneous batch size per device = 4
  Total optimization steps = 22546
----------------------------------------------------
Epoch 1:  50%|████████████████████████                | 11273/22546 [00:50<00:50, 220.34it/s, loss=3.24]
Epoch 1: 100%|████████████████████████████████████████| 22546/22546 [01:40<00:00, 224.17it/s, loss=2.95]
***** Running Evaluation *****
  Num examples = 10021
  Perplexity = 20.1
"""

# Loss is the model's 'wrongness' score, for language models we use cross-entropy loss:
# - Model predicts a probability distribution for the next token
# - Loss measures how far off that prediction was from the true token
# - Lower loss = better predictions
# For example:
# True next token = "king" -> Model says {king: 0.7, queen: 0.3} -> loss = low

# Perplexity is a fancy way of turning loss into a more intuitive number:
# - Perplexity = exp(loss)
# - Lower perplexity = better
# - Intuition is the average branching factor of the model, ~1 (almost always right), 
# ~20 (model is choosing between 20 possible tokens on average),
# ~100 (model is confused)