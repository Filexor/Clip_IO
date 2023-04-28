import os

import gradio
import torch
import pandas

from modules import scripts, script_callbacks, shared, devices
from modules.shared import opts
from modules.sd_hijack_clip import PromptChunkFix, PromptChunk, FrozenCLIPEmbedderWithCustomWordsBase

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
    
    def get_chunks(prompt: str, clip: FrozenCLIPEmbedderWithCustomWordsBase) -> PromptChunk:
        if opts.use_old_emphasis_implementation:
            raise NotImplementedError
            pass
        batch_chunks, _ = clip.process_texts([prompt])
        return batch_chunks
        pass

    def get_flat_embeddings(batch_chunks: PromptChunk, clip: FrozenCLIPEmbedderWithCustomWordsBase) -> torch.Tensor:
        input_ids = []
        fixes = []
        offset = 0
        for chunk in batch_chunks[0]:
            input_ids += chunk.tokens
            for i, fix in enumerate(chunk.fixes):
                fix: PromptChunkFix
                fix = PromptChunkFix(fix.offset + offset, fix.embedding)
                chunk.fixes[i] = fix
            fixes += chunk.fixes
            offset += len(chunk.tokens)
            pass
        clip.hijack.fixes = [fixes]
        input_ids_Tensor = torch.asarray([input_ids]).to(devices.device)
        return clip.wrapped.transformer.text_model.embeddings.token_embedding(input_ids_Tensor)
        pass

    def on_save_embeddings_as_pt(prompt: str, filename: str, transpose: bool):
        clip: FrozenCLIPEmbedderWithCustomWordsBase = shared.sd_model.cond_stage_model
        batch_chunks = Clip_IO.get_chunks(prompt, clip)
        embeddings: torch.Tensor = Clip_IO.get_flat_embeddings(batch_chunks, clip)

        filename = os.path.realpath(filename)
        dir = os.path.dirname(filename)
        if not os.path.exists(dir): os.makedirs(dir)
        if not filename.endswith(".pt"): filename += ".pt"
        torch.save(embeddings.t() if transpose else embeddings, filename)
        pass

    def on_save_embeddings_as_csv(prompt: str, filename: str, transpose: bool):
        clip: FrozenCLIPEmbedderWithCustomWordsBase = shared.sd_model.cond_stage_model
        batch_chunks = Clip_IO.get_chunks(prompt, clip)
        embeddings: torch.Tensor = Clip_IO.get_flat_embeddings(batch_chunks, clip)

        filename = os.path.realpath(filename)
        dir = os.path.dirname(filename)
        if not os.path.exists(dir): os.makedirs(dir)
        if not filename.endswith(".csv"): filename += ".csv"
        embeddings_numpy = embeddings[0].t().to("cpu").numpy() if transpose else embeddings[0].to("cpu").numpy()
        embeddings_dataframe = pandas.DataFrame(embeddings_numpy)
        embeddings_dataframe.to_csv(filename, float_format = "%.8e")
        pass

    def on_save_conditioning_as_pt(prompt: str, filename: str, transpose: bool, no_emphasis: bool, no_norm: bool):
        clip: FrozenCLIPEmbedderWithCustomWordsBase = shared.sd_model.cond_stage_model
        batch_chunks = Clip_IO.get_chunks(prompt, clip)
        chunk_count = max([len(x) for x in batch_chunks])
        zs = []
        for i in range(chunk_count):
            batch_chunk = [chunks[i] if i < len(chunks) else clip.empty_chunk() for chunks in batch_chunks]
            remade_batch_tokens = [x.tokens for x in batch_chunk]
            tokens = torch.asarray([x.tokens for x in batch_chunk]).to(devices.device)
            clip.hijack.fixes = [x.fixes for x in batch_chunk]

            if clip.id_end != clip.id_pad:
                for batch_pos in range(len(remade_batch_tokens)):
                    index = remade_batch_tokens[batch_pos].index(clip.id_end)
                    tokens[batch_pos, index+1:tokens.shape[1]] = clip.id_pad
            
            z = clip.encode_with_transformers(tokens)
            if not no_emphasis:
                batch_multipliers = torch.asarray([x.multipliers for x in batch_chunk]).to(devices.device)
                original_mean = z.mean()
                z = z * batch_multipliers.reshape(batch_multipliers.shape + (1,)).expand(z.shape)
                new_mean = z.mean()
                z = z * (original_mean / new_mean) if not no_norm else z
            zs.append(z[0])
        conditioning = torch.hstack(zs)

        filename = os.path.realpath(filename)
        dir = os.path.dirname(filename)
        if not os.path.exists(dir): os.makedirs(dir)
        if not filename.endswith(".pt"): filename += ".pt"
        torch.save(conditioning.t() if transpose else conditioning, filename)
        pass

    def on_save_conditioning_as_csv(prompt: str, filename: str, transpose: bool, no_emphasis: bool, no_norm: bool):
        clip: FrozenCLIPEmbedderWithCustomWordsBase = shared.sd_model.cond_stage_model
        batch_chunks = Clip_IO.get_chunks(prompt, clip)
        chunk_count = max([len(x) for x in batch_chunks])
        zs = []
        for i in range(chunk_count):
            batch_chunk = [chunks[i] if i < len(chunks) else clip.empty_chunk() for chunks in batch_chunks]
            remade_batch_tokens = [x.tokens for x in batch_chunk]
            tokens = torch.asarray([x.tokens for x in batch_chunk]).to(devices.device)
            clip.hijack.fixes = [x.fixes for x in batch_chunk]

            if clip.id_end != clip.id_pad:
                for batch_pos in range(len(remade_batch_tokens)):
                    index = remade_batch_tokens[batch_pos].index(clip.id_end)
                    tokens[batch_pos, index+1:tokens.shape[1]] = clip.id_pad
            
            z = clip.encode_with_transformers(tokens)
            if not no_emphasis:
                batch_multipliers = torch.asarray([x.multipliers for x in batch_chunk]).to(devices.device)
                original_mean = z.mean()
                z = z * batch_multipliers.reshape(batch_multipliers.shape + (1,)).expand(z.shape)
                new_mean = z.mean()
                z = z * (original_mean / new_mean) if not no_norm else z
            zs.append(z[0])
        conditioning = torch.hstack(zs)

        filename = os.path.realpath(filename)
        dir = os.path.dirname(filename)
        if not os.path.exists(dir): os.makedirs(dir)
        if not filename.endswith(".csv"): filename += ".csv"
        conditioning_numpy = conditioning.t().to("cpu").numpy() if transpose else conditioning.to("cpu").numpy()
        conditioning_dataframe = pandas.DataFrame(conditioning_numpy)
        conditioning_dataframe.to_csv(filename, float_format = "%.8e")
        pass

    def tab():
        with gradio.Blocks() as tab:
            prompt = gradio.TextArea(max_lines = 256, label = "Prompt")
            with gradio.Row():
                output_embeddings_name = gradio.Textbox(value = r"outputs\embeddings\out_emb", label = "Output embeddings name")
                output_embeddings_transpose = gradio.Checkbox(value = False, label = "Transpose matrix")
                output_embeddings_as_pt_button = gradio.Button("Save embeddings as .pt")
                output_embeddings_as_csv_button = gradio.Button("Save embeddings as .csv")
                pass
            with gradio.Row():
                output_conditioning_name = gradio.Textbox(value = r"outputs\embeddings\out_cond", label = "Output conditioning name")
                output_conditioning_transpose = gradio.Checkbox(value = False, label = "Transpose matrix")
                output_conditioning_ignore_emphasis = gradio.Checkbox(value = False, label = "Ignore emphasis")
                output_conditioning_bypass_conditioning_normalization = gradio.Checkbox(value = False, label = "Bypass conditioning normalization")
                output_conditioning_button_as_pt = gradio.Button("Save conditioning as .pt")
                output_conditioning_button_as_csv = gradio.Button("Save conditioning as .csv")
                pass
            output_embeddings_as_pt_button.click(Clip_IO.on_save_embeddings_as_pt, [prompt, output_embeddings_name, output_embeddings_transpose])
            output_embeddings_as_csv_button.click(Clip_IO.on_save_embeddings_as_csv, [prompt, output_embeddings_name, output_embeddings_transpose])
            output_conditioning_button_as_pt.click(Clip_IO.on_save_conditioning_as_pt, [prompt, output_conditioning_name, output_conditioning_transpose, output_conditioning_ignore_emphasis, output_conditioning_bypass_conditioning_normalization])
            output_conditioning_button_as_csv.click(Clip_IO.on_save_conditioning_as_csv, [prompt, output_conditioning_name, output_conditioning_transpose, output_conditioning_ignore_emphasis, output_conditioning_bypass_conditioning_normalization])
            pass
        return [(tab, "Clip Output", "Clip_Output")]
        pass

    script_callbacks.on_ui_tabs(tab)
    pass
