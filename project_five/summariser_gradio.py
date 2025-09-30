import gradio as gr # Library for throwing together quick web UIs for ML models
from transformers import pipeline

summariser = pipeline("summarization", model="facebook/bart-large-cnn")

def summarise_text(input_text):
    result = summariser(input_text, max_length=50, min_length=30, do_sample=False)
    return result[0]['summary_text']

# Interface
# {fn=summarise_text} is the function that runs when you hit the button
# {inputs=gr.Textbox(...)} defines the UI input component
# {outputs=gr.Textbox(...)} UI output component
# {title} and {description} metadata for the UI page

demo = gr.Interface(
    fn=summarise_text,
    inputs=gr.Textbox(lines=15, label="Paste your text"),
    outputs=gr.Textbox(label="Summary"),
    title="AI Summariser",
    description="Paste any article or notes and get a concise summary"
)

# Launches the app
demo.launch()