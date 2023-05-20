import os, csv, warnings, datetime

import gradio
import torch
import pandas
from tkinter import filedialog

from modules import scripts, script_callbacks, shared, devices, processing, prompt_parser 
from modules.shared import opts
from modules.sd_hijack_clip import PromptChunkFix, PromptChunk, FrozenCLIPEmbedderWithCustomWordsBase

mode_types = ["replace", "concatenate", "command"]

class Clip_IO(scripts.Script):
    class Ui_manager():
        def __init__(self):
            self.mode = "replace"
            self.main_blocks: gradio.Blocks | None = None
            pass
        pass

    ui_txt2img = Ui_manager()
    ui_img2img = Ui_manager()

    enabled = False

    positive_filenames = []
    negative_filenames = []
    conditioning_cache = {}
    positive_exist = False
    negative_exist = False
    
    evacuate_get_learned_conditioning = None
    evacuate_get_multicond_learned_conditioning = None
    evacuate_get_conds_with_caching = None

    def __init__(self):
        pass

    def title(self):
        return "Clip I/O"
        pass

    def show(self, is_img2img):
        return scripts.AlwaysVisible
        pass

    def show_conditioning_open_dialog() -> str:
        results = filedialog.askopenfilenames(filetypes = [("Comma-Separated Values", "*.csv"), ("Pytorch Tensor", "*.pt"), ("Any File", "*")])
        output = ""
        for result in results:
            if ";" in str(result):
                warnings.warn(f'In "{result}",\ninvalid character ";" was found when parsing the file name.\nFile name must not contain ";".\nContinue as if the file is not specified.')
                continue
                pass
            output += str(result) + ";"
            pass
        return output[:-1]
        pass

    def ui(self, is_img2img):
        with gradio.Accordion("Clip input", open = False):
            with gradio.Row():
                enabled = gradio.Checkbox(label = "Enable")
                mode = gradio.Dropdown(choices = mode_types, value = "Replace", label = "Clip input mode")
                pass
            with gradio.Blocks(visible = True) as main_blocks:
                with gradio.Row():
                    replace_positive = gradio.Textbox(label = "Replacement for Positive prompt")
                    replace_positive_button = gradio.Button("📂")
                    replace_negative = gradio.Textbox(label = "Replacement for Negative prompt")
                    replace_negative_button = gradio.Button("📂")
                pass
            pass
        replace_positive_button.click(Clip_IO.show_conditioning_open_dialog, outputs = replace_positive, show_progress = False)
        replace_negative_button.click(Clip_IO.show_conditioning_open_dialog, outputs = replace_negative, show_progress = False)
        if not is_img2img:
            if Clip_IO.ui_txt2img.mode == "replace":
                return [enabled, mode, replace_positive, replace_negative]
                pass
            pass
        else:
            if Clip_IO.ui_img2img.mode == "replace":
                return [enabled, mode, replace_positive, replace_negative]
                pass
            pass
        return []
        pass
    
    def my_get_learned_conditioning(model, prompts, steps, is_negative = True):
        """converts a list of prompts into a list of prompt schedules - each schedule is a list of ScheduledPromptConditioning, specifying the comdition (cond),
        and the sampling step at which this condition is to be replaced by the next one.

        Input:
        (model, ['a red crown', 'a [blue:green:5] jeweled crown'], 20)

        Output:
        [
            [
                ScheduledPromptConditioning(end_at_step=20, cond=tensor([[-0.3886,  0.0229, -0.0523,  ..., -0.4901, -0.3066,  0.0674], ..., [ 0.3317, -0.5102, -0.4066,  ...,  0.4119, -0.7647, -1.0160]], device='cuda:0'))
            ],
            [
                ScheduledPromptConditioning(end_at_step=5, cond=tensor([[-0.3886,  0.0229, -0.0522,  ..., -0.4901, -0.3067,  0.0673], ..., [-0.0192,  0.3867, -0.4644,  ...,  0.1135, -0.3696, -0.4625]], device='cuda:0')),
                ScheduledPromptConditioning(end_at_step=20, cond=tensor([[-0.3886,  0.0229, -0.0522,  ..., -0.4901, -0.3067,  0.0673], ..., [-0.7352, -0.4356, -0.7888,  ...,  0.6994, -0.4312, -1.2593]], device='cuda:0'))
            ]
        ]
        """
        res = []

        prompt_schedules = prompt_parser.get_learned_conditioning_prompt_schedules(prompts, steps)
        cache = {}

        for prompt, prompt_schedule in zip(prompts, prompt_schedules):

            cached = cache.get(prompt, None)
            if cached is not None:
                res.append(cached)
                continue

            texts = [x[1] for x in prompt_schedule]
            conds = model.get_learned_conditioning(texts)

            cond_schedule = []
            for i, (end_at_step, text) in enumerate(prompt_schedule):
                if not Clip_IO.enabled:
                    cond_schedule.append(prompt_parser.ScheduledPromptConditioning(end_at_step, conds[i]))
                    pass
                else:
                    if not is_negative and Clip_IO.positive_exist:
                        if Clip_IO.ui_txt2img.mode == "replace":
                            cond_schedule.append(prompt_parser.ScheduledPromptConditioning(end_at_step, torch.hstack([Clip_IO.conditioning_cache[filename].to(devices.device) for filename in Clip_IO.positive_filenames])))
                            pass
                        pass
                    elif is_negative and Clip_IO.negative_exist:
                        if Clip_IO.ui_txt2img.mode == "replace":
                            cond_schedule.append(prompt_parser.ScheduledPromptConditioning(end_at_step, torch.hstack([Clip_IO.conditioning_cache[filename].to(devices.device) for filename in Clip_IO.negative_filenames])))
                            pass
                        pass
                    else:
                        cond_schedule.append(prompt_parser.ScheduledPromptConditioning(end_at_step, conds[i]))
                        pass
                    pass

            cache[prompt] = cond_schedule
            res.append(cond_schedule)

        return res
        pass

    def my_get_multicond_learned_conditioning(model, prompts, steps) -> prompt_parser.MulticondLearnedConditioning:
        """same as get_learned_conditioning, but returns a list of ScheduledPromptConditioning along with the weight objects for each prompt.
        For each prompt, the list is obtained by splitting the prompt using the AND separator.

        https://energy-based-model.github.io/Compositional-Visual-Generation-with-Composable-Diffusion-Models/
        """

        res_indexes, prompt_flat_list, prompt_indexes = prompt_parser.get_multicond_prompt_list(prompts)

        learned_conditioning = prompt_parser.get_learned_conditioning(model, prompt_flat_list, steps, is_negative = False)

        res = []
        for indexes in res_indexes:
            res.append([prompt_parser.ComposableScheduledPromptConditioning(learned_conditioning[i], weight) for i, weight in indexes])

        return prompt_parser.MulticondLearnedConditioning(shape=(len(prompts),), batch=res)
        pass

    def load_csv_conditioning(filename: str | os.PathLike) -> torch.Tensor | None:
        filename = os.path.realpath(filename)
        with open(filename, newline='') as csv_file:
            csv_reader = csv.reader(csv_file)
            pivot_found = False
            matrix = []
            for i, row in enumerate(csv_reader):
                if not pivot_found:
                    for j, data in enumerate(row):
                        if data == "embeddings" or data == "conditioning":
                            pivot_found = True
                            transposed = False
                            column_begin = i
                            row_begin = j
                            break
                            pass
                        if data == "embeddingsT" or data == "conditioningT":
                            pivot_found = True
                            transposed = True
                            column_begin = i
                            row_begin = j
                            break
                            pass
                        pass
                    pass
                else: 
                    if column_begin == i:
                        continue
                        pass
                    if len(row) < row_begin + 1:
                        if i == column_begin + 1:
                            warnings.warn(f'In "{filename}",\nconditioning data does not exist or misaligned.\nContinue as if the file is not specified.')
                            return None
                            pass
                        else:
                            break
                            pass
                        pass
                    array = []
                    for j, data in enumerate(row[row_begin + 1:]):
                        try:
                            value = float(data)
                            pass
                        except:
                            break
                            pass
                        array.append(value)
                        pass
                    if i == column_begin + 1:
                        length = j
                        pass
                    else:
                        if length != j:
                            warnings.warn(f'In "{filename}",\nshape of conditioning data is non-rectangular or conditioning data does not exist or misaligned.\nContinue as if the file is not specified.')
                            return None
                            pass
                        pass
                    matrix.append(array)
                    pass
                pass
            if not pivot_found:
                warnings.warn(f'In "{filename}",\nnone of following keywords found: "embeddings", "embeddingsT", "conditioning", "conditioningT".\nContinue as if the file is not specified.')
                return None
                pass
            try:
                conditioning = torch.Tensor(matrix)
                pass
            except Exception as e:
                warnings.warn(f'In "{filename}",\nsomething went wrong while converting csv data to pytorch Tensor.\nContinue as if the file is not specified.')
                warnings.warn(repr(e))
                return None
                pass
            return conditioning.t() if transposed else conditioning
            pass
        pass

    def get_my_get_conds_with_caching():
        def get_conds_with_caching(function, required_prompts, steps, cache):
            """
            Returns the result of calling function(shared.sd_model, required_prompts, steps)
            using a cache to store the result if the same arguments have been used before.

            cache is an array containing two elements. The first element is a tuple
            representing the previously used arguments, or None if no arguments
            have been used before. The second element is where the previously
            computed result is stored.
            """

            if cache[0] is not None and (required_prompts, steps) == cache[0]:
                return cache[1]

            with devices.autocast():
                cache[1] = function(shared.sd_model, required_prompts, steps)

            cache[0] = (required_prompts, steps)
            return cache[1]
            pass
        return get_conds_with_caching
        pass

    def get_inner_function(outer, new_inner):
        """Replace a nested function code object used by outer with new_inner

        The replacement new_inner must use the same name and must at most use the
        same closures as the original.

        """
        if hasattr(new_inner, '__code__'):
            # support both functions and code objects
            new_inner = new_inner.__code__

        # find original code object so we can validate the closures match
        ocode = outer.__code__
        function, code = type(outer), type(ocode)
        iname = new_inner.co_name
        orig_inner = next(
            const for const in ocode.co_consts
            if isinstance(const, code) and const.co_name == iname)

        # you can ignore later closures, but since they are matched by position
        # the new sequence must match the start of the old.
        assert (orig_inner.co_freevars[:len(new_inner.co_freevars)] ==
                new_inner.co_freevars), 'New closures must match originals'

        # and a new function object using the updated code object
        return function(
            ocode, outer.__globals__, outer.__name__,
            outer.__defaults__, outer.__closure__
        )

    def replace_inner_function(outer, new_inner):
        """Replace a nested function code object used by outer with new_inner

        The replacement new_inner must use the same name and must at most use the
        same closures as the original.

        """
        if hasattr(new_inner, '__code__'):
            # support both functions and code objects
            new_inner = new_inner.__code__

        # find original code object so we can validate the closures match
        ocode = outer.__code__
        function, code = type(outer), type(ocode)
        iname = new_inner.co_name
        orig_inner = next(
            const for const in ocode.co_consts
            if isinstance(const, code) and const.co_name == iname)

        # you can ignore later closures, but since they are matched by position
        # the new sequence must match the start of the old.
        assert (orig_inner.co_freevars[:len(new_inner.co_freevars)] ==
                new_inner.co_freevars), 'New closures must match originals'

        # replace the code object for the inner function
        new_consts = tuple(
            new_inner if const is orig_inner else const
            for const in outer.__code__.co_consts)

        # create a new code object with the new constants
        try:
            # Python 3.8 added code.replace(), so much more convenient!
            ncode = ocode.replace(co_consts=new_consts)
        except AttributeError:
            # older Python versions, argument counts vary so we need to check
            # for specifics.
            args = [
                ocode.co_argcount, ocode.co_nlocals, ocode.co_stacksize,
                ocode.co_flags, ocode.co_code,
                new_consts,  # replacing the constants
                ocode.co_names, ocode.co_varnames, ocode.co_filename,
                ocode.co_name, ocode.co_firstlineno, ocode.co_lnotab,
                ocode.co_freevars, ocode.co_cellvars,
            ]
            if hasattr(ocode, 'co_kwonlyargcount'):
                # Python 3+, insert after co_argcount
                args.insert(1, ocode.co_kwonlyargcount)
            # Python 3.8 adds co_posonlyargcount, but also has code.replace(), used above
            ncode = code(*args)

        # and a new function object using the updated code object
        return function(
            ncode, outer.__globals__, outer.__name__,
            outer.__defaults__, outer.__closure__
        )

    def process(self, p: processing.StableDiffusionProcessing, *args):
        if args[0]:
            Clip_IO.enabled = True
            Clip_IO.positive_filenames: list[str | os.PathLike] = str.split(args[2], ";")
            Clip_IO.negative_filenames: list[str | os.PathLike] = str.split(args[3], ";")
            Clip_IO.negative_exist = False
            Clip_IO.positive_exist = False

            for i, positive_filename in enumerate(Clip_IO.positive_filenames):
                if positive_filename in Clip_IO.conditioning_cache or not os.path.exists(positive_filename) or os.path.isdir(positive_filename):
                    continue
                    pass
                if positive_filename.endswith(".csv"):
                    conditioning = Clip_IO.load_csv_conditioning(positive_filename)
                    if conditioning is not None:
                        Clip_IO.conditioning_cache[positive_filename] = conditioning
                        Clip_IO.positive_exist = True
                        pass
                    else:
                        del Clip_IO.positive_filenames[i]
                        pass
                    pass
                else:
                    try:
                        Clip_IO.conditioning_cache[positive_filename] = torch.load(positive_filename)
                        Clip_IO.positive_exist = True
                        pass
                    except Exception as e:
                        warnings.warn(f'In "{positive_filename}",\nsomething went wrong while loading pytorch Tensor.\nContinue as if the file is not specified.')
                        warnings.warn(repr(e))
                        return None
                    pass
                pass
            for i, negative_filename in enumerate(Clip_IO.negative_filenames):
                if negative_filename in Clip_IO.conditioning_cache or not os.path.exists(negative_filename) or os.path.isdir(negative_filename):
                    continue
                    pass
                if negative_filename.endswith(".csv"):
                    conditioning = Clip_IO.load_csv_conditioning(negative_filename)
                    if conditioning is not None:
                        Clip_IO.conditioning_cache[negative_filename] = conditioning
                        Clip_IO.negative_exist = True
                        pass
                    else:
                        del Clip_IO.negative_filenames[i]
                        pass
                    pass
                else:
                    try:
                        Clip_IO.conditioning_cache[negative_filename] = torch.load(negative_filename)
                        Clip_IO.negative_exist = True
                        pass
                    except Exception as e:
                        warnings.warn(f'In "{negative_filename}",\nsomething went wrong while loading pytorch Tensor.\nContinue as if the file is not specified.')
                        warnings.warn(repr(e))
                        return None
                    pass
                pass

            if Clip_IO.positive_exist:
                p.prompt = ""
                pass
            if Clip_IO.negative_exist:
                p.negative_prompt = ""
                pass

            Clip_IO.evacuate_get_learned_conditioning = prompt_parser.get_learned_conditioning
            Clip_IO.evacuate_get_multicond_learned_conditioning = prompt_parser.get_multicond_learned_conditioning
            #Clip_IO.evacuate_get_conds_with_caching = Clip_IO.get_inner_function(processing.process_images_inner, Clip_IO.get_my_get_conds_with_caching()) # Flush cache in my_get_learned_conditioning instead. 
            prompt_parser.get_learned_conditioning = Clip_IO.my_get_learned_conditioning
            prompt_parser.get_multicond_learned_conditioning = Clip_IO.my_get_multicond_learned_conditioning
            #Clip_IO.replace_inner_function(processing.process_images_inner, Clip_IO.get_my_get_conds_with_caching())
            pass
        else:
            Clip_IO.enabled = False
            pass
        pass

    def postprocess(self, p: processing.StableDiffusionProcessing, processed, *args):
        if args[0]:
            Clip_IO.enabled = False
            prompt_parser.get_learned_conditioning = Clip_IO.evacuate_get_learned_conditioning
            prompt_parser.get_multicond_learned_conditioning = Clip_IO.evacuate_get_multicond_learned_conditioning
            #Clip_IO.replace_inner_function(processing.process_images_inner, Clip_IO.evacuate_get_conds_with_caching)
            pass
        pass

    def get_chunks(prompt: str, clip: FrozenCLIPEmbedderWithCustomWordsBase) -> PromptChunk:
        if opts.use_old_emphasis_implementation:
            raise NotImplementedError
            pass
        batch_chunks, _ = clip.process_texts([prompt])
        return batch_chunks
        pass

    def get_flat_embeddings(batch_chunks: PromptChunk, clip: FrozenCLIPEmbedderWithCustomWordsBase) -> tuple[torch.Tensor, list[str]]:
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

        tokens = [clip.wrapped.tokenizer.decoder.get(input_id) for input_id in input_ids]
        for fix in fixes:
            tokens[fix.offset + 1] = fix.embedding.name
            for i in range(1, fix.embedding.vec.shape[0]):
                tokens[fix.offset + 1 + i] = ""
                pass
            pass

        return clip.wrapped.transformer.text_model.embeddings.token_embedding(input_ids_Tensor), tokens
        pass

    def on_save_embeddings_as_pt(prompt: str, filename: str, overwrite: bool):
        try:
            clip: FrozenCLIPEmbedderWithCustomWordsBase = shared.sd_model.cond_stage_model
            batch_chunks = Clip_IO.get_chunks(prompt, clip)
            embeddings: torch.Tensor = Clip_IO.get_flat_embeddings(batch_chunks, clip)

            filename = os.path.join(os.path.dirname(__file__), "../conditioning", filename)
            filename = os.path.realpath(filename)
            dir = os.path.dirname(filename)
            if not os.path.exists(dir): os.makedirs(dir)
            if not filename.endswith(".pt"): filename += ".pt"
            if os.path.exists(filename) and not overwrite:
                raise FileExistsError()
                pass
            torch.save(embeddings, filename)
            pass
        except FileExistsError as e:
            print(repr(e))
            return f'<span style="color: red">Saving failed. File "{filename}" already exists. {datetime.datetime.now().isoformat()}</span>'
            pass
        except Exception as e:
            print(repr(e))
            return f'<span style="color: red">Saving failed. {datetime.datetime.now().isoformat()}</span>'
            pass
        return f'File {filename} is successfully saved. {datetime.datetime.now().isoformat()}'
        pass

    def on_save_embeddings_as_csv(prompt: str, filename: str, transpose: bool, add_token: bool, overwrite: bool):
        try:
            clip: FrozenCLIPEmbedderWithCustomWordsBase = shared.sd_model.cond_stage_model
            batch_chunks = Clip_IO.get_chunks(prompt, clip)
            embeddings, tokens = Clip_IO.get_flat_embeddings(batch_chunks, clip)

            embeddings: list[list[str]] = embeddings[0].tolist()
            width = len(embeddings[0])
            for i, row in enumerate(embeddings):
                row.insert(0, str(i))
                if add_token:
                    row.insert(0, tokens[i])
                    pass
                pass
            row_first = list(range(width))
            row_first.insert(0, "embeddingsT" if transpose else "embeddings")
            if add_token:
                    row_first.insert(0, "")
                    pass
            embeddings.insert(0, row_first)
            if transpose:
                embeddings = [list(x) for x in zip(*embeddings)]
                pass

            filename = os.path.join(os.path.dirname(__file__), "../conditioning", filename)
            filename = os.path.realpath(filename)
            dir = os.path.dirname(filename)
            if not os.path.exists(dir): os.makedirs(dir)
            if not filename.endswith(".csv"): filename += ".csv"
            
            with open(filename, "wt" if overwrite else "xt") as file:
                writer = csv.writer(file, lineterminator = "\n")
                writer.writerows(embeddings)
                pass
            pass
        except FileExistsError as e:
            print(repr(e))
            return f'<span style="color: red">Saving failed. File "{filename}" already exists. {datetime.datetime.now().isoformat()}</span>'
            pass
        except Exception as e:
            print(repr(e))
            return f'<span style="color: red">Saving failed. {datetime.datetime.now().isoformat()}</span>'
            pass
        return f'File {filename} is successfully saved. {datetime.datetime.now().isoformat()}'
        pass

    def on_save_conditioning_as_pt(prompt: str, filename: str, no_emphasis: bool, no_norm: bool, overwrite: bool):
        try:
            with devices.autocast():
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

                filename = os.path.join(os.path.dirname(__file__), "../conditioning", filename)
                filename = os.path.realpath(filename)
                dir = os.path.dirname(filename)
                if not os.path.exists(dir): os.makedirs(dir)
                if not filename.endswith(".pt"): filename += ".pt"
                if os.path.exists(filename) and not overwrite:
                    raise FileExistsError()
                    pass
                torch.save(conditioning, filename)
            pass
        except FileExistsError as e:
            print(repr(e))
            return f'<span style="color: red">Saving failed. File "{filename}" already exists. {datetime.datetime.now().isoformat()}</span>'
            pass
        except Exception as e:
            print(repr(e))
            return f'<span style="color: red">Saving failed. {datetime.datetime.now().isoformat()}</span>'
            pass
        return f'File {filename} is successfully saved. {datetime.datetime.now().isoformat()}'
        pass

    def on_save_conditioning_as_csv(prompt: str, filename: str, transpose: bool, no_emphasis: bool, no_norm: bool, add_token: bool, overwrite: bool):
        try:
            with devices.autocast():
                clip: FrozenCLIPEmbedderWithCustomWordsBase = shared.sd_model.cond_stage_model
                batch_chunks = Clip_IO.get_chunks(prompt, clip)
                _, token_list = Clip_IO.get_flat_embeddings(batch_chunks, clip)
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

                conditioning: list[list[str]] = conditioning.tolist()
                width = len(conditioning[0])
                for i, row in enumerate(conditioning):
                    row.insert(0, str(i))
                    if add_token:
                        row.insert(0, token_list[i])
                        pass
                    pass
                row_first = list(range(width))
                row_first.insert(0, "conditioningT" if transpose else "conditioning")
                if add_token:
                        row_first.insert(0, "")
                        pass
                conditioning.insert(0, row_first)
                if transpose:
                    conditioning = [list(x) for x in zip(*conditioning)]
                    pass

                filename = os.path.join(os.path.dirname(__file__), "../conditioning", filename)
                filename = os.path.realpath(filename)
                dir = os.path.dirname(filename)
                if not os.path.exists(dir): os.makedirs(dir)
                if not filename.endswith(".csv"): filename += ".csv"
                
                with open(filename, "wt" if overwrite else "xt") as file:
                    writer = csv.writer(file, lineterminator = "\n")
                    writer.writerows(conditioning)
                    pass
                pass
            pass
        except FileExistsError as e:
            print(repr(e))
            return f'<span style="color: red">Saving failed. File "{filename}" already exists. {datetime.datetime.now().isoformat()}</span>'
            pass
        except Exception as e:
            print(repr(e))
            return f'<span style="color: red">Saving failed. {datetime.datetime.now().isoformat()}</span>'
            pass
        return f'File {filename} is successfully saved. {datetime.datetime.now().isoformat()}'
        pass

    def tab():
        with gradio.Blocks() as tab:
            prompt = gradio.TextArea(max_lines = 256, label = "Prompt")
            with gradio.Row():
                output_transpose = gradio.Checkbox(value = True, label = "Transpose matrix")
                output_ignore_emphasis = gradio.Checkbox(value = False, label = "Ignore emphasis")
                output_bypass_conditioning_normalization = gradio.Checkbox(value = False, label = "Bypass conditioning normalization")
                output_token_string = gradio.Checkbox(value = True, label = "Add token strings")
                output_overwrite = gradio.Checkbox(value = False, label = "Overwrite")
                pass
            with gradio.Row():
                output_name = gradio.Textbox(value = r"output", label = "Output name")
                output_embeddings_as_pt = gradio.Button("Save embeddings as .pt")
                output_embeddings_as_csv = gradio.Button("Save embeddings as .csv")
                output_conditioning_as_pt = gradio.Button("Save conditioning as .pt")
                output_conditioning_as_csv = gradio.Button("Save conditioning as .csv")
                pass
            output_notification = gradio.HTML()
            output_embeddings_as_pt.click(Clip_IO.on_save_embeddings_as_pt, [prompt, output_name, output_overwrite], [output_notification])
            output_embeddings_as_csv.click(Clip_IO.on_save_embeddings_as_csv, [prompt, output_name, output_transpose, output_token_string, output_overwrite], [output_notification])
            output_conditioning_as_pt.click(Clip_IO.on_save_conditioning_as_pt, [prompt, output_name, output_ignore_emphasis, output_bypass_conditioning_normalization, output_overwrite], [output_notification])
            output_conditioning_as_csv.click(Clip_IO.on_save_conditioning_as_csv, [prompt, output_name, output_transpose, output_ignore_emphasis, output_bypass_conditioning_normalization, output_token_string, output_overwrite], [output_notification])
            pass
        return [(tab, "Clip Output", "Clip_Output")]
        pass

    script_callbacks.on_ui_tabs(tab)
    pass
