import gradio

from modules import scripts, script_callbacks

class Clip_IO(scripts.Script):
    def __init__(self):
        pass

    def title(self):
        return "Clip I/O"
        pass

    def show(self, is_img2img):
        return scripts.AlwaysVisible
        pass

    def ui(self, is_img2img):
        with gradio.Accordion("Clip I/O"):

            pass
        pass
    
    def tab():
        with gradio.Box as tab:
            prompt = gradio.TextArea(max_lines = 256, label = "Prompt")
            with gradio.Row():
                output_embeddings_name = gradio.Textbox(label = "Output embeddings name")
                output_embeddings_button = gradio.Button("Save embeddings")
                pass
            with gradio.Row():
                output_conditioning_name = gradio.Textbox(label = "Output conditioning name")
                output_conditioning_button = gradio.Button("Save conditioning")
                pass
            pass
        return [(tab, "Clip Output", "Clip_Output")]
        pass

    script_callbacks.on_ui_tabs(tab)
    pass
