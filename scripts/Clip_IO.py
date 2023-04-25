import gradio
import torch
import os
import numpy
import pandas

from modules import scripts, script_callbacks, shared, devices
from modules.shared import opts

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
    
    def get_chunks(prompt: str, clip):
        if opts.use_old_emphasis_implementation:
            raise NotImplementedError
            pass
        batch_chunks, _ = clip.process_texts([prompt])
        return batch_chunks
        pass

    def get_flat_embeddings(batch_chunks, clip) -> torch.Tensor:
        input_ids = []
        fixes = []
        for chunk in batch_chunks[0]:
            input_ids += chunk.tokens
            fixes += chunk.fixes
            pass
        clip.hijack.fixes = [fixes]
        input_ids_Tensor = torch.asarray([input_ids]).to(devices.device)
        return clip.wrapped.transformer.text_model.embeddings.token_embedding(input_ids_Tensor)
        pass

    def on_save_embeddings_as_pt(prompt: str, filename: str, transpose: bool):
        clip = shared.sd_model.cond_stage_model
        batch_chunks = Clip_IO.get_chunks(prompt, clip)
        embeddings: torch.Tensor = Clip_IO.get_flat_embeddings(batch_chunks, clip)
        filename = os.path.realpath(filename)
        dir = os.path.dirname(filename)
        if not os.path.exists(dir): os.makedirs(dir)
        if not filename.endswith(".pt"): filename += ".pt"
        torch.save(embeddings.transpose() if transpose else embeddings, filename)
        pass

    def on_save_embeddings_as_csv(prompt: str, filename: str, transpose: bool):
        clip = shared.sd_model.cond_stage_model
        batch_chunks = Clip_IO.get_chunks(prompt, clip)
        embeddings: torch.Tensor = Clip_IO.get_flat_embeddings(batch_chunks, clip)
        filename = os.path.realpath(filename)
        dir = os.path.dirname(filename)
        if not os.path.exists(dir): os.makedirs(dir)
        if not filename.endswith(".csv"): filename += ".csv"
        embeddings_numpy = embeddings[0].transpose().to("cpu").numpy() if transpose else embeddings[0].to("cpu").numpy()
        embeddings_dataframe = pandas.DataFrame(embeddings_numpy)
        embeddings_dataframe.to_csv(filename)
        pass

    def on_save_conditioning(prompt: str, filename: str):
        pass

    def tab():
        with gradio.Blocks() as tab:
            prompt = gradio.TextArea(max_lines = 256, label = "Prompt")
            with gradio.Row():
                output_embeddings_name = gradio.Textbox(value = r"outputs\embeddings\out", label = "Output embeddings name")
                output_embeddings_transpose = gradio.Checkbox(value = False, label = "Transpose matrix")
                output_embeddings_as_pt_button = gradio.Button("Save embeddings as .pt")
                output_embeddings_as_csv_button = gradio.Button("Save embeddings as .csv")
                pass
            with gradio.Row():
                output_conditioning_name = gradio.Textbox(label = "Output conditioning name")
                output_conditioning_button = gradio.Button("Save conditioning")
                pass
            output_embeddings_as_pt_button.click(Clip_IO.on_save_embeddings_as_pt, [prompt, output_embeddings_name, output_embeddings_transpose])
            output_embeddings_as_csv_button.click(Clip_IO.on_save_embeddings_as_csv, [prompt, output_embeddings_name, output_embeddings_transpose])
            output_conditioning_button.click(Clip_IO.on_save_conditioning, [prompt, output_conditioning_name])
            pass
        return [(tab, "Clip Output", "Clip_Output")]
        pass

    script_callbacks.on_ui_tabs(tab)
    pass
