import gradio as gr
from transformers import pipeline

# Pipeline for summariser
summariser = pipeline("summarization", model="facebook/bart-large-cnn")

def summarise_text(input_text, min_len, max_len, creativity):
    result = summariser(
        input_text,
        max_length=max_len,
        min_length=min_len,
        do_sample=(creativity > 0),
        temperature=creativity if creativity > 0 else 1.0
    )
    return result[0]['summary_text']

demo = gr.Interface(
    fn=summarise_text,
    inputs=[
        gr.Textbox(lines=15, label="Paste your text"),
        gr.Slider(10, 200, value=30, step=5, label="Minimum Length"),
        gr.Slider(20, 300, value=100, step=5, label="Maximum Length"),
        gr.Slider(0, 2, value=0, step=0.1, label="Creativity (0 = safe, >0 = random)")
    ],
    outputs=gr.Textbox(lines=15, label="Summary"),
    title="AI Summariser",
    description="Paste text and tweak the knobs to control summary length and style."
)

demo.launch()
