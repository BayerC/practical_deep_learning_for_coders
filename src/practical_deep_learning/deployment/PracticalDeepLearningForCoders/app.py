import gradio as gr
from fastai.vision.all import load_learner

learner = load_learner("export.pkl")

demo = gr.Interface(
    fn=lambda img: learner.predict(img)[0],
    inputs=gr.Image(),
    outputs=gr.Textbox(),
    examples=["cubic.png", "quadratic.png", "linear.png"],
)

demo.launch()
