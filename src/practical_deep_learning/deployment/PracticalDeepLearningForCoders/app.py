import gradio as gr


def greet(name: str) -> str:
    return "Hi " + name + "!!"


demo = gr.Interface(fn=greet, inputs="text", outputs="text")
demo.launch()
