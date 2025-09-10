import warnings
from pathlib import Path
from typing import Any

import gradio as gr
from fastai.vision.all import load_learner


def classify_image(img: Any) -> Any:
    return learner.predict(img)[0]


# Get the directory where this script is located
script_dir = Path(__file__).parent

# Suppress the pickle warning since we trust our own model file
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*load_learner.*pickle.*")
    learner = load_learner(script_dir / "export.pkl")

with gr.Blocks() as demo:
    gr.Markdown("# Image Processing Demo")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(width=500, height=500)
            submit_btn = gr.Button("Process Image")

        with gr.Column():
            output_text = gr.Textbox(label="Result")

    # Create examples using gr.Examples with proper file paths
    gr.Examples(
        examples=[
            str(script_dir / "cubic.png"),
            str(script_dir / "quadratic.png"),
            str(script_dir / "linear.png"),
        ],
        inputs=image_input,
        label="Example Images",
    )

    submit_btn.click(fn=classify_image, inputs=image_input, outputs=output_text)

demo.launch()
