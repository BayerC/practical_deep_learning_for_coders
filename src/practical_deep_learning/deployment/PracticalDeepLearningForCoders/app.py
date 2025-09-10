from typing import Any

import gradio as gr


def greet(img: Any) -> str:
    print(img.size())
    return "Hi"


image = gr.Image(shape=(500, 500))
# label = gr.outputs.Label()
examples = ["./cubic.png", "./quadratic.png", "./linear.png"]

demo = gr.Interface(fn=greet, inputs=image, outputs="text", examples=examples)
demo.launch()
