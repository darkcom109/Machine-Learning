from transformers import pipeline # Import pipeline/wrapper

# Pipeline is configured for summarisation, model is pretrained
summariser = pipeline("summarization", model="facebook/bart-large-cnn")

# Raw input
text = """
The quick brown fox jumps over the lazy dog. This sentence is famous for
containing all the letters of the English alphabet. It is often used as a typing exercise
or to display fonts. The phrase was popularized in the late 19th century and has
since become a classic example of pangrams.
"""

# Runs inference (a trained model to make predictions) and returns a list of results
# {do_sample} deterministic (False) or creative (True)
# Tokenization -> Encoder-Decoder forward pass -> Decoding -> Detokenization
summary = summariser(text, max_length=50, min_length=10, do_sample=False)

# Grab the first time
print("Summary:", summary[0]['summary_text'])