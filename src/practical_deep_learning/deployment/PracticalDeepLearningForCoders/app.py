from pathlib import Path

import gradio as gr
from fastai.vision.all import load_learner

script_dir = Path(__file__).parent
learner = load_learner(script_dir / "export.pkl")

demo = gr.Interface(
    fn=lambda img: learner.predict(img)[0],
    inputs=gr.Image(),
    outputs=gr.Textbox(),
    examples=[
        str(script_dir / "cubic.png"),
        str(script_dir / "quadratic.png"),
        str(script_dir / "linear.png"),
    ],
)

demo.launch()
