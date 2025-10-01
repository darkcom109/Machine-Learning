import gradio as gr
from transformers import pipeline

# Load a code generation model
generator = pipeline("text-generation", model="bigcode/starcoder2-3B")

def generate_text(input_text, max_tokens, creativity):
    result = generator(
        input_text,
        max_new_tokens=max_tokens,                 # generate only N new tokens
        do_sample=True,                            # always sample
        temperature=creativity if creativity > 0 else 0.7,
        top_p=0.9,                                 # nucleus sampling
        top_k=50,                                  # limit choices
        repetition_penalty=1.2,                    # discourage loops
        pad_token_id=generator.tokenizer.eos_token_id,
    )
    return result[0]["generated_text"]

demo = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(lines=5, label="Prompt (e.g. a docstring or function header)"),
        gr.Slider(10, 200, value=50, step=10, label="Max New Tokens"),
        gr.Slider(0.1, 2, value=0.7, step=0.1, label="Creativity (temperature)"),
    ],
    outputs=gr.Code(language="python", label="Generated Code"),
    title="Code Generator (CodeGen-350M)",
    description="Give it a prompt (like a docstring or function name) and get code back.",
)

if __name__ == "__main__":
    demo.launch()
