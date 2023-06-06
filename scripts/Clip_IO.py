import os, csv, warnings, datetime
import math as math
from collections import namedtuple
from enum import IntEnum

import gradio
import torch
import lark
import open_clip

from modules import scripts, script_callbacks, shared, devices, processing, prompt_parser 
from modules.shared import opts
from modules.sd_hijack_clip import PromptChunkFix, PromptChunk, FrozenCLIPEmbedderWithCustomWords
from modules.sd_hijack_open_clip import FrozenOpenCLIPEmbedderWithCustomWords

mode_types = ["replace", "concatenate", "command"]

class Clip_IO(scripts.Script):
    enabled = False
    mode_positive = "Disabled"
    mode_negative = "Disabled"
    conditioning_cache = {}
    
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

    def ui(self, is_img2img):
        with gradio.Accordion("Clip input", open = False):
            with gradio.Row():
                enabled = gradio.Checkbox(label = "Enable")
                mode_positive = gradio.Dropdown(["Disabled", "Simple", "Directive"], value = "Disabled", max_choices = 1, label = "Positive prompt mode")
                mode_negative = gradio.Dropdown(["Disabled", "Simple", "Directive"], value = "Disabled", max_choices = 1, label = "Positive prompt mode")
                pass
            pass
        if not is_img2img:
            return [enabled, mode_positive, mode_negative]
            pass
        else:
            return [enabled, mode_positive, mode_negative]
            pass
        return []
        pass
    
    syntax_simple = r"""
    start: (FILE | PROMPT | SPACE)*
    FILE: /"(?!"").+?"/ | /'(?!"").+?'/ | /[^"'\s]+/
    PROMPT: /"{3}.*?"{3}|'{3}.*?'{3}/
    SPACE: /[\s]+/
    """

    def get_cond_simple(model, input: str, is_negative: bool) -> torch.Tensor | None:
        conds = []
        class Process(lark.Transformer):
            def FILE(self, token: lark.Token):
                cond: torch.Tensor | None = None
                filename_original = token.value
                if filename_original.startswith('"') and filename_original.endswith('"') or filename_original.startswith("'") and filename_original.endswith("'"):
                    filename_original = filename_original[1:-1]
                if filename_original in Clip_IO.conditioning_cache:
                    cond = Clip_IO.conditioning_cache[filename_original]
                    if cond is not None:
                        conds.append(cond)
                        pass
                    return
                    pass
                filename = os.path.join(os.path.dirname(__file__), "../conditioning", filename_original)
                filename = os.path.realpath(filename)
                if filename.endswith(".csv"):
                    cond = Clip_IO.load_csv_conditioning(filename)
                    pass
                elif filename.endswith(".pt"):
                    try:
                        cond = torch.load(filename)
                        pass
                    except Exception:
                        cond = None
                        pass
                    pass
                else:
                    if os.path.exists(filename) and not os.path.isdir(filename):
                        cond =  Clip_IO.load_csv_conditioning(filename)
                        if cond is None:
                            try:
                                cond = torch.load(filename)
                                pass
                            except Exception:
                                cond = None
                                pass
                            pass
                        pass
                    if cond is None and os.path.exists(filename + ".csv"):
                        cond = Clip_IO.load_csv_conditioning(filename + ".csv")
                        pass
                    if cond is None and not os.path.exists(filename + ".csv") and os.path.exists(filename + ".pt"):
                        try:
                            cond = torch.load(filename + ".pt")
                            pass
                        except Exception:
                            cond = None
                            pass
                        pass
                    pass
                if cond is not None:
                    conds.append(cond.to(devices.device))
                    pass
                Clip_IO.conditioning_cache[filename_original] = cond
                pass
            def PROMPT(self, token: lark.Token):
                string = token.value
                if string.startswith('"""') and string.endswith('"""') or string.startswith("'''") and string.endswith("'''"):
                    string = string[3:-3]
                    pass
                conds.append(model.get_learned_conditioning([string])[0].to(devices.device))
                pass
            pass
        Process().transform(lark.Lark(Clip_IO.syntax_simple).parse(input))
        if len(conds) != 0:
            return torch.vstack(conds)
            pass
        else:
            warnings.warn(f"{'Negative prompt' if is_negative else 'Positive prompt'} is empty. Retrieving conditioning for empty string.")
            return model.get_learned_conditioning([""])[0]
            pass
        pass

    syntax_directive = r"""
    start: (FILE | PROMPT | directive | SPACE)*
    FILE: /"(?!"").+?"/ | /'(?!"").+?'/ | /[^?"'\s]+/
    PROMPT: /"{3}.*?"{3}|'{3}.*?'{3}/
    directive: "?" DIRECTIVE ("_" DIRECTIVE_ORDER)? "(" directive_inner ")"
    DIRECTIVE: (/[0-9a-zA-Z]+/ | /_(?![0-9]+\()+/)+
    DIRECTIVE_ORDER: /[0-9]+/
    directive_inner: (DIRECTIVE_PLAIN | directive_parentheses)*
    !directive_parentheses: "(" (DIRECTIVE_PLAIN | directive_parentheses)* ")"
    DIRECTIVE_PLAIN: /[^()]+/
    SPACE: /\s+/
    """

    class Directive:
        class Names(IntEnum):
            eval
            pass

        def __init__(self, name: str, order: int, inner: str):
            self.name = name.lower()
            self.order = order
            self.inner = inner
            pass

        def __lt__(self, other) -> bool:
            if type(self) != type(other):
                raise TypeError()
                pass
            if self.order == other.order:
                return True
                pass
            else:
                return self.order < other.order
                pass
            pass
        pass

    def get_cond_directive(model, input: str, is_negative: bool) -> torch.Tensor | None:
        conds: list[torch.tensor] = []
        dirs: list[Clip_IO.Directive] = []
        class Process(lark.Transformer):
            def FILE(self, token: lark.Token):
                cond: torch.Tensor | None = None
                filename_original = token.value
                if filename_original.startswith('"') and filename_original.endswith('"') or filename_original.startswith("'") and filename_original.endswith("'"):
                    filename_original = filename_original[1:-1]
                if filename_original in Clip_IO.conditioning_cache:
                    cond = Clip_IO.conditioning_cache[filename_original]
                    if cond is not None:
                        conds.append(cond)
                        pass
                    return
                    pass
                filename = os.path.join(os.path.dirname(__file__), "../conditioning", filename_original)
                filename = os.path.realpath(filename)
                if filename.endswith(".csv"):
                    cond = Clip_IO.load_csv_conditioning(filename)
                    pass
                elif filename.endswith(".pt"):
                    try:
                        cond = torch.load(filename)
                        pass
                    except Exception:
                        cond = None
                        pass
                    pass
                else:
                    if os.path.exists(filename) and not os.path.isdir(filename):
                        cond =  Clip_IO.load_csv_conditioning(filename)
                        if cond is None:
                            try:
                                cond = torch.load(filename)
                                pass
                            except Exception:
                                cond = None
                                pass
                            pass
                        pass
                    if cond is None and os.path.exists(filename + ".csv"):
                        cond = Clip_IO.load_csv_conditioning(filename + ".csv")
                        pass
                    if cond is None and not os.path.exists(filename + ".csv") and os.path.exists(filename + ".pt"):
                        try:
                            cond = torch.load(filename + ".pt")
                            pass
                        except Exception:
                            cond = None
                            pass
                        pass
                    pass
                if cond is not None:
                    conds.append(cond.to(devices.device))
                    pass
                Clip_IO.conditioning_cache[filename_original] = cond
                pass
            def PROMPT(self, token: lark.Token):
                string = token.value
                if string.startswith('"""') and string.endswith('"""') or string.startswith("'''") and string.endswith("'''"):
                    string = string[3:-3]
                    pass
                conds.append(model.get_learned_conditioning([string])[0].to(devices.device))
                pass
            def directive(self, args: list[lark.Token | lark.Tree]):
                def flatten(arg: lark.Token | lark.Tree | list[lark.Token | lark.Tree]) -> str | lark.Token:
                    if type(arg) == lark.Token:
                        return arg.value
                        pass
                    elif type(arg) == lark.Tree:
                        array = ""
                        for child in arg.children:
                            array += flatten(child)
                            pass
                        return array
                        pass
                    elif type(arg) == list:
                        array = ""
                        for component in arg:
                            array += flatten(component)
                            pass
                        return array
                        pass
                    else:
                        return arg
                        pass
                    pass

                dirs.append(Clip_IO.Directive(args[0], args[1] if len(args) == 3 else 0, flatten(args[-1])))
                pass
            pass

        Process().transform(lark.Lark(Clip_IO.syntax_directive).parse(input))
        i = torch.vstack(conds)
        o = i.clone()
        dirs.sort()
        for dir in dirs:
            if dir.name == "eval":
                try:
                    for t in range(i.shape[0]):
                        for d in range(i.shape[1]):
                            local = {"i": i, "o": o, "t": t, "d": d, "torch": torch.__dict__} | math.__dict__
                            o[t, d] = eval(dir.inner, None, local)
                            pass
                        pass
                    pass
                except Exception as e:
                    print(repr(e))
                    o = i
                    pass
                finally:
                    i = o.clone()
                pass
            elif dir.name == "exec":
                try:
                    local = {"i": i, "o": o, "torch": torch.__dict__} | math.__dict__
                    exec(dir.inner, None, local)
                except Exception as e:
                    print(repr(e))
                    o = i
                    pass
                finally:
                    i = local["o"].clone()
                pass
            else:
                warnings.warn(f'Directive "{dir.name}" does not exist.')
                pass
            pass
        cond = i

        if cond is not None and cond.shape[0] > 0 and (cond.shape[1] == 768 or cond.shape[1] == 1024):
            return cond.to(devices.device)
            pass
        else:
            warnings.warn(f"{'Negative prompt' if is_negative else 'Positive prompt'} is empty. Retrieving conditioning for empty string.")
            return model.get_learned_conditioning([""])[0]
            pass
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

        if Clip_IO.enabled and (Clip_IO.mode_positive == "Directive" and not is_negative or Clip_IO.mode_negative == "Directive" and is_negative):
            # TODO: Implement own parser
            prompt_schedules = [[[steps, prompt]] for prompt in prompts]
            pass
        else:
            prompt_schedules = prompt_parser.get_learned_conditioning_prompt_schedules(prompts, steps)
            pass

        res = []
        cache = {}
        for prompt, prompt_schedule in zip(prompts, prompt_schedules):

            cached = cache.get(prompt, None)
            if cached is not None:
                res.append(cached)
                continue

            texts: list[str] = [x[1] for x in prompt_schedule]
            if Clip_IO.enabled and (Clip_IO.mode_positive == "Simple" and not is_negative or Clip_IO.mode_negative == "Simple" and is_negative):
                conds = []
                for text in texts:
                    conds.append(Clip_IO.get_cond_simple(model, text, is_negative))
                    pass
                pass
            elif Clip_IO.enabled and (Clip_IO.mode_positive == "Directive" and not is_negative or Clip_IO.mode_negative == "Directive" and is_negative):
                conds = []
                for text in texts:
                    conds.append(Clip_IO.get_cond_directive(model, text, is_negative))
                    pass
                pass
            else:
                conds = model.get_learned_conditioning(texts)
                pass

            cond_schedule = []
            for i, (end_at_step, text) in enumerate(prompt_schedule):
                cond_schedule.append(prompt_parser.ScheduledPromptConditioning(end_at_step, conds[i]))

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
            Clip_IO.conditioning_cache = {}
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
            Clip_IO.conditioning_cache = {}
            prompt_parser.get_learned_conditioning = Clip_IO.evacuate_get_learned_conditioning
            prompt_parser.get_multicond_learned_conditioning = Clip_IO.evacuate_get_multicond_learned_conditioning
            #Clip_IO.replace_inner_function(processing.process_images_inner, Clip_IO.evacuate_get_conds_with_caching)
            pass
        pass

    def process_batch(self, p: processing.StableDiffusionProcessing, *args, **kwargs):
        Clip_IO.mode_positive = args[1]
        Clip_IO.mode_negative = args[2]
        pass

    def postprocess_batch(self, p: processing.StableDiffusionProcessing, *args, **kwargs):
        Clip_IO.mode_positive = "Disabled"
        Clip_IO.mode_negative = "Disabled"
        pass

    def tokenize_line_manual_chunk(prompt: str, clip: FrozenCLIPEmbedderWithCustomWords | FrozenOpenCLIPEmbedderWithCustomWords) -> list[PromptChunk]:
        if opts.enable_emphasis:
            parsed = prompt_parser.parse_prompt_attention(prompt)
            pass
        else:
            parsed = [[prompt, 1.0]]
            pass

        tokenized = clip.tokenize([text for text, _ in parsed])

        chunks: list[PromptChunk] = []
        chunk = PromptChunk()
        last_comma = -1

        def next_chunk(is_last=False):
            nonlocal last_comma
            nonlocal chunk

            # We don't have to fill the chunk.

            last_comma = -1
            chunks.append(chunk)
            chunk = PromptChunk()
            pass

        for tokens, (text, weight) in zip(tokenized, parsed):
            if text == 'BREAK' and weight == -1:
                next_chunk()
                continue
                pass

            position = 0
            while position < len(tokens):
                token = tokens[position]

                if token == clip.comma_token:
                    last_comma = len(chunk.tokens)
                    pass

                # this is when we are at the end of alloted 77 tokens for the current chunk, and the current token is not a comma. opts.comma_padding_backtrack
                # is a setting that specifies that if there is a comma nearby, the text after the comma should be moved out of this chunk and into the next.
                elif opts.comma_padding_backtrack != 0 and len(chunk.tokens) == clip.chunk_length + 2 and last_comma != -1 and len(chunk.tokens) - last_comma <= opts.comma_padding_backtrack:
                    break_location = last_comma + 1

                    reloc_tokens = chunk.tokens[break_location:]
                    reloc_mults = chunk.multipliers[break_location:]

                    chunk.tokens = chunk.tokens[:break_location]
                    chunk.multipliers = chunk.multipliers[:break_location]

                    next_chunk()
                    chunk.tokens = reloc_tokens
                    chunk.multipliers = reloc_mults
                    pass

                if len(chunk.tokens) == clip.chunk_length + 2:
                    next_chunk()
                    pass

                embedding, embedding_length_in_tokens = clip.hijack.embedding_db.find_embedding_at_position(tokens, position)

                # The token is not Textual Inversion
                if embedding is None:
                    chunk.tokens.append(token)
                    chunk.multipliers.append(weight)
                    position += 1
                    continue
                    pass

                emb_len = int(embedding.vec.shape[0])
                if len(chunk.tokens) + emb_len > clip.chunk_length + 2:
                    next_chunk()
                    pass

                chunk.fixes.append(PromptChunkFix(len(chunk.tokens), embedding))

                chunk.tokens += [0] * emb_len
                chunk.multipliers += [weight] * emb_len
                position += embedding_length_in_tokens
                pass

        if len(chunk.tokens) > 0 or len(chunks) == 0:
            next_chunk(is_last=True)
            pass

        return chunks

    def process_texts_manual_chunk(prompts: list[str], clip: FrozenCLIPEmbedderWithCustomWords | FrozenOpenCLIPEmbedderWithCustomWords) -> list[list[PromptChunk]]:
        cache = {}
        batch_chunks: list[list[PromptChunk]] = []
        for prompt in prompts:
            if prompt in cache:
                chunks = cache[prompt]
            else:
                chunks = Clip_IO.tokenize_line_manual_chunk(prompt, clip)
                cache[prompt] = chunks

            batch_chunks.append(chunks)

        # We have to ensure all chunk in batch_chunks have same length.
        # If not, fill with padding token and raise warning.
        max_length = -1
        warned = False
        for chunks in batch_chunks:
            for chunk in chunks:
                if max_length != -1 and max_length != len(chunk.tokens) and not warned:
                    warnings.warn("All chunk doesn't have same length. For processing, we'll fill with padding token to match same length.")
                    warned = True
                    pass
                max_length = max(max_length, len(chunk.tokens))
                pass
            pass

        for chunks in batch_chunks:
            for chunk in chunks:
                chunk.tokens += [clip.id_pad] * max(max_length - len(chunk.tokens), 0)
                chunk.multipliers += [1.0] * max(max_length - len(chunk.multipliers), 0)
                pass
            pass

        return batch_chunks

    def get_chunks(prompt: str, clip: FrozenCLIPEmbedderWithCustomWords | FrozenOpenCLIPEmbedderWithCustomWords, manual_chunk: bool) -> list[list[PromptChunk]]:
        if opts.use_old_emphasis_implementation:
            raise NotImplementedError
            pass
        if manual_chunk:
            return Clip_IO.process_texts_manual_chunk([prompt], clip)
            pass
        else:
            batch_chunks, _ = clip.process_texts([prompt])
            return batch_chunks
            pass
        pass

    def get_flat_embeddings(batch_chunks: list[list[PromptChunk]], clip: FrozenCLIPEmbedderWithCustomWords | FrozenOpenCLIPEmbedderWithCustomWords, manual_chunk: bool) -> tuple[torch.Tensor, list[str]]:
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

        decode: callable[any, str]
        if hasattr(clip.wrapped, "tokenizer"):  # Ver.1.x
            decode = clip.wrapped.tokenizer.decoder.get
            is_open_clip = False
            pass
        else:   # Ver.2.x
            decode = lambda t: open_clip.tokenizer._tokenizer.decoder.get(t)
            is_open_clip = True
            pass
        tokens = [decode(input_id) for input_id in input_ids]

        if  manual_chunk:
            for fix in fixes:
                tokens[fix.offset] = fix.embedding.name
                for i in range(1, fix.embedding.vec.shape[0]):
                    tokens[fix.offset + i] = ""
                    pass
                pass
            pass
        else:
            for fix in fixes:
                tokens[fix.offset + 1] = fix.embedding.name
                for i in range(1, fix.embedding.vec.shape[0]):
                    tokens[fix.offset + 1 + i] = ""
                    pass
                pass
            pass

        return clip.wrapped.model.token_embedding(input_ids_Tensor) if is_open_clip else clip.wrapped.transformer.text_model.embeddings.token_embedding(input_ids_Tensor), tokens
        pass

    def on_save_embeddings_as_pt(prompt: str, filename: str, overwrite: bool, manual_chunk: bool):
        try:
            clip: FrozenCLIPEmbedderWithCustomWords | FrozenOpenCLIPEmbedderWithCustomWords = shared.sd_model.cond_stage_model
            batch_chunks = Clip_IO.get_chunks(prompt, clip, manual_chunk)
            embeddings: torch.Tensor = Clip_IO.get_flat_embeddings(batch_chunks, clip, manual_chunk)

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

    def on_save_embeddings_as_csv(prompt: str, filename: str, transpose: bool, add_token: bool, overwrite: bool, manual_chunk: bool):
        try:
            clip: FrozenCLIPEmbedderWithCustomWords | FrozenOpenCLIPEmbedderWithCustomWords = shared.sd_model.cond_stage_model
            batch_chunks = Clip_IO.get_chunks(prompt, clip, manual_chunk)
            embeddings, tokens = Clip_IO.get_flat_embeddings(batch_chunks, clip, manual_chunk)

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

    def on_save_conditioning_as_pt(prompt: str, filename: str, no_emphasis: bool, no_norm: bool, overwrite: bool, manual_chunk: bool):
        try:
            with devices.autocast():
                clip: FrozenCLIPEmbedderWithCustomWords | FrozenOpenCLIPEmbedderWithCustomWords = shared.sd_model.cond_stage_model
                batch_chunks = Clip_IO.get_chunks(prompt, clip, manual_chunk)
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
                conditioning = torch.vstack(zs)

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

    def on_save_conditioning_as_csv(prompt: str, filename: str, transpose: bool, no_emphasis: bool, no_norm: bool, add_token: bool, overwrite: bool, manual_chunk: bool):
        try:
            with devices.autocast():
                clip: FrozenCLIPEmbedderWithCustomWords | FrozenOpenCLIPEmbedderWithCustomWords = shared.sd_model.cond_stage_model
                batch_chunks = Clip_IO.get_chunks(prompt, clip, manual_chunk)
                _, token_list = Clip_IO.get_flat_embeddings(batch_chunks, clip, manual_chunk)
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
                conditioning = torch.vstack(zs)

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
                output_manual_chunk = gradio.Checkbox(value = False, label = "Don't add bos / eos / pad tokens")
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
            output_embeddings_as_pt.click(Clip_IO.on_save_embeddings_as_pt, [prompt, output_name, output_overwrite, output_manual_chunk], [output_notification])
            output_embeddings_as_csv.click(Clip_IO.on_save_embeddings_as_csv, [prompt, output_name, output_transpose, output_token_string, output_overwrite, output_manual_chunk], [output_notification])
            output_conditioning_as_pt.click(Clip_IO.on_save_conditioning_as_pt, [prompt, output_name, output_ignore_emphasis, output_bypass_conditioning_normalization, output_overwrite, output_manual_chunk], [output_notification])
            output_conditioning_as_csv.click(Clip_IO.on_save_conditioning_as_csv, [prompt, output_name, output_transpose, output_ignore_emphasis, output_bypass_conditioning_normalization, output_token_string, output_overwrite, output_manual_chunk], [output_notification])
            pass
        return [(tab, "Clip Output", "Clip_Output")]
        pass

    script_callbacks.on_ui_tabs(tab)
    pass
