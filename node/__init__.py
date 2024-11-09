import os
import re
import pathlib
from typing import Any, Dict, ClassVar, Literal, Optional
from invokeai.invocation_api import (
    BaseInvocation,
    InvocationContext,
    UIComponent,
    BaseInvocationOutput,
    InputField,
    invocation,
    invocation_output,
    OutputField
)
from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.backend.util.logging import InvokeAILogger
from invokeai.backend.util.devices import TorchDevice
from pydantic import BaseModel, Field
from ..utils import setup_llama_cpp, setup_kgen



invoke_logger = InvokeAILogger.get_logger(name='InvokeAI-TIPO')
invoke_logger.info("Loading TIPO Node")
invoke_logger.info("Check TIPO Requirements")
setup_llama_cpp()
setup_kgen()


try:
    import llama_cpp
except Exception as e:
    llama_cpp = None


import kgen.models as models
import kgen.executor.tipo as tipo
from kgen.logging import logger
from kgen.formatter import seperate_tags, apply_format
from kgen.executor.tipo import (
    parse_tipo_request,
    tipo_single_request,
    tipo_runner,
    apply_tipo_prompt,
    parse_tipo_result,
    OPERATION_LIST,
)



models.model_dir = pathlib.Path(os.path.join(InvokeAIAppConfig.find_root(), "models", "kgen"))
os.makedirs(models.model_dir, exist_ok=True)
logger.info(f"TIPO Model Dir: {models.model_dir}")


MODEL_NAME_LIST = [
    f"{model_name} | {file}".strip("_")
    for model_name, ggufs in models.tipo_model_list
    for file in ggufs
] + [i[0] for i in models.tipo_model_list]


TIPO_MODEL_TYPES = Optional[Literal[tuple(MODEL_NAME_LIST)]]
TIPO_OPERATION_LIST = Optional[Literal[tuple(OPERATION_LIST)]]
PROMPT_LENGTH_TYPES = Optional[Literal[
    "very_short",
    "short",
    "long",
    "very_long"
]]


re_attention = re.compile(
    r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:\s*([+-]?[.\d]+)\s*\)|
\)|
]|
[^\\()\[\]:]+|
:
""",
    re.X,
)

re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)



def parse_prompt_attention(text):
    """
    Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
    Accepted tokens are:
      (abc) - increases attention to abc by a multiplier of 1.1
      (abc:3.12) - increases attention to abc by a multiplier of 3.12
      [abc] - decreases attention to abc by a multiplier of 1.1
      \( - literal character '('
      \[ - literal character '['
      \) - literal character ')'
      \] - literal character ']'
      \\ - literal character '\'
      anything else - just text

    >>> parse_prompt_attention('normal text')
    [['normal text', 1.0]]
    >>> parse_prompt_attention('an (important) word')
    [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
    >>> parse_prompt_attention('(unbalanced')
    [['unbalanced', 1.1]]
    >>> parse_prompt_attention('\(literal\]')
    [['(literal]', 1.0]]
    >>> parse_prompt_attention('(unnecessary)(parens)')
    [['unnecessaryparens', 1.1]]
    >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
    [['a ', 1.0],
     ['house', 1.5730000000000004],
     [' ', 1.1],
     ['on', 1.0],
     [' a ', 1.1],
     ['hill', 0.55],
     [', sun, ', 1.1],
     ['sky', 1.4641000000000006],
     ['.', 1.1]]
    """

    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith("\\"):
            res.append([text[1:], 1.0])
        elif text == "(":
            round_brackets.append(len(res))
        elif text == "[":
            square_brackets.append(len(res))
        elif weight is not None and round_brackets:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ")" and round_brackets:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == "]" and square_brackets:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            parts = re.split(re_break, text)
            for i, part in enumerate(parts):
                if i > 0:
                    res.append(["BREAK", -1])
                res.append([part, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    # merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res


def apply_strength(tag_map, strength_map, strength_map_nl):
    for cate in tag_map.keys():
        new_list = []
        # Skip natural language output at first
        if isinstance(tag_map[cate], str):
            # Ensure all the parts in the strength_map are in the prompt
            if all(part in tag_map[cate] for part, strength in strength_map_nl):
                org_prompt = tag_map[cate]
                new_prompt = ""
                for part, strength in strength_map_nl:
                    before, org_prompt = org_prompt.split(part, 1)
                    new_prompt += before.replace("(", "\(").replace(")", "\)")
                    part = part.replace("(", "\(").replace(")", "\)")
                    new_prompt += f"({part}:{strength})"
                new_prompt += org_prompt
            tag_map[cate] = new_prompt
            continue
        for org_tag in tag_map[cate]:
            tag = org_tag.replace("(", "\(").replace(")", "\)")
            if org_tag in strength_map:
                new_list.append(f"({tag}:{strength_map[org_tag]})")
            else:
                new_list.append(tag)
        tag_map[cate] = new_list
    return tag_map



class TIPOOperationField(BaseModel):
    """TIPO Operation Node Output"""
    data: Dict[str, Any] = Field(default={}, description="TIPO Operation Node Output")


@invocation_output('tipo_prompt_string_output')
class TipoPromptOutput(BaseInvocationOutput):
    """output prompt processed by tipo"""

    prompt: str = OutputField(description="prompt processed by tipo")
    user_prompt: str = OutputField(description="user input prompt")
    unformatted_prompt: str = OutputField(description="unformatted prompt processed by tipo")
    unformatted_user_prompt: str = OutputField(description="unformatted user prompt")


@invocation_output('tipo_operation_output')
class TipoOperationOutput(BaseInvocationOutput):
    """output prompt processed by tipo operation"""

    full_output: TIPOOperationField = OutputField(description="prompt processed by tipo")
    addon_output: TIPOOperationField = OutputField(description="user input prompt")


current_model = None


@invocation(
    "TIPO",
    title="TIPO",
    tags=["tipo", "prompt"],
    category="prompt",
    version="1.0.4"
)
class TIPO(BaseInvocation):
    """Prompt Upscale"""

    current_model: ClassVar[None] = None

    tags: Optional[str] = InputField(
        default="",
        description="enter your tags",
        ui_component=UIComponent.Textarea
    )
    nl_prompt: Optional[str] = InputField(
        default="",
        description="enter your natural language",
        ui_component=UIComponent.Textarea
    )
    ban_tags: Optional[str] = InputField(
        default="",
        description="enter your tags need to ban",
        ui_component=UIComponent.Textarea
    )
    format: Optional[str] = InputField(
        default="""<|special|>, 
<|characters|>, <|copyrights|>, 
<|artist|>, 

<|general|>,

<|extended|>.

<|quality|>, <|meta|>, <|rating|>""",
        description="output tags format",
        ui_component=UIComponent.Textarea
    )
    tipo_model: TIPO_MODEL_TYPES = InputField( # type: ignore
        default="KBlueLeaf/TIPO-200M | TIPO-200M-40Btok-F16.gguf",
        description="model type for prompt generation"
    )
    width: Optional[int] = InputField(
        default=1024,
        description="none"
    )
    height: Optional[int] = InputField(
        default=1024,
        description="none"
    )
    temperature: Optional[float] = InputField(
        default=0.5,
        description="adjust the degree of random"
    )
    top_p: Optional[float] = InputField(
        default=0.95,
        description="adjust the degree of unconfident tokens"
    )
    min_p: Optional[float] = InputField(
        default=0.05,
        description="adjust the degree of unconfident tokens"
    )
    top_k: Optional[int] = InputField(
        default=80,
        description="adjust the degree of unconfident tokens"
    )
    tag_length: PROMPT_LENGTH_TYPES = InputField(
        default="long",
        description="adjust tag length"
    )
    nl_length: PROMPT_LENGTH_TYPES = InputField(
        default="long",
        description="adjust natural language length"
    )
    seed: Optional[int] = InputField(
        default=1,
        description="seed for upsampling tags"
    )
    device: Optional[Literal["Auto", "CPU", "CUDA"]] = InputField(
        default="Auto",
        description="choice device to load tipo model"
    )


    def execute(
        self,
        tipo_model: str,
        tags: str,
        nl_prompt: str,
        width: int,
        height: int,
        seed: int,
        tag_length: str,
        nl_length: str,
        ban_tags: str,
        format: str,
        temperature: float,
        top_p: float,
        min_p: float,
        top_k: int,
        device: str,
    ):
        global current_model
        if (tipo_model, device) != current_model:
            if " | " in tipo_model:
                model_name, gguf_name = tipo_model.split(" | ")
                target_file = f"{model_name.split('/')[-1]}_{gguf_name}"
                if str(models.model_dir / target_file) not in models.list_gguf():
                    models.download_gguf(model_name, gguf_name)
                target = os.path.join(str(models.model_dir), target_file)
                gguf = True
            else:
                target = tipo_model
                gguf = False
            try:
                models.load_model(target, gguf, device=device)
            except:
                models.load_model(target, gguf, device="cpu")
            # self.current_model = tipo_model
            TIPO.current_model = tipo_model
        aspect_ratio = width / height
        prompt_without_extranet = tags
        prompt_parse_strength = parse_prompt_attention(prompt_without_extranet)

        nl_prompt_parse_strength = parse_prompt_attention(nl_prompt)
        nl_prompt = ""
        strength_map_nl = []
        for part, strength in nl_prompt_parse_strength:
            nl_prompt += part
            if strength == 1:
                continue
            strength_map_nl.append((part, strength))

        black_list = [tag.strip() for tag in ban_tags.split(",") if tag.strip()]
        tipo.BAN_TAGS = black_list
        all_tags = []
        strength_map = {}
        for part, strength in prompt_parse_strength:
            part_tags = [tag.strip() for tag in part.strip().split(",") if tag.strip()]
            all_tags.extend(part_tags)
            if strength == 1:
                continue
            for tag in part_tags:
                strength_map[tag] = strength

        tag_length = tag_length.replace(" ", "_")
        org_tag_map = seperate_tags(all_tags)
        meta, operations, general, nl_prompt = parse_tipo_request(
            org_tag_map,
            nl_prompt,
            tag_length_target=tag_length,
            nl_length_target=nl_length,
            generate_extra_nl_prompt=(not nl_prompt and "<|extended|>" in format)
            or "<|generated|>" in format,
        )
        meta["aspect_ratio"] = f"{aspect_ratio:.1f}"

        org_formatted_prompt = parse_tipo_result(
            apply_tipo_prompt(
                meta,
                general,
                nl_prompt,
                "short_to_tag_to_long",
                tag_length,
                True,
                gen_meta=True,
            )
        )
        org_formatted_prompt = apply_strength(
            org_formatted_prompt, strength_map, strength_map_nl
        )
        formatted_prompt_by_user = apply_format(org_formatted_prompt, format)
        unformatted_prompt_by_user = tags + nl_prompt

        tag_map, _ = tipo_runner(
            meta,
            operations,
            general,
            nl_prompt,
            temperature=temperature,
            seed=seed,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
        )

        addon = {
            "tags": [],
            "nl": "",
        }
        for cate in tag_map.keys():
            if cate == "generated" and addon["nl"] == "":
                addon["nl"] = tag_map[cate]
                continue
            if cate == "extended":
                extended = tag_map[cate]
                addon["nl"] = extended
                continue
            if cate not in org_tag_map:
                continue
            for tag in tag_map[cate]:
                if tag in org_tag_map[cate]:
                    continue
                addon["tags"].append(tag)
        addon = apply_strength(addon, strength_map, strength_map_nl)
        unformatted_prompt_by_tipo = (
            tags + ", " + ", ".join(addon["tags"]) + "\n" + addon["nl"]
        )

        tag_map = apply_strength(tag_map, strength_map, strength_map_nl)
        formatted_prompt_by_tipo = apply_format(tag_map, format)
        return (
            formatted_prompt_by_tipo,
            formatted_prompt_by_user,
            unformatted_prompt_by_tipo,
            unformatted_prompt_by_user,
        )


    def invoke(self, context: InvocationContext) -> TipoPromptOutput:
        if self.device == "Auto":
            device = TorchDevice.choose_torch_device()
        elif self.device == "CPU":
            device = "cpu"
        else:
            device = "cuda"

        logger.info(f"Device: {device}")
        logger.info("Upscaling Prompt")
        prompt, user_prompt, unformatted_prompt, unformatted_user_prompt = self.execute(
            tipo_model=self.tipo_model,
            tags=self.tags,
            nl_prompt=self.nl_prompt,
            width=self.width,
            height=self.height,
            seed=self.seed,
            tag_length=self.tag_length,
            nl_length=self.nl_length,
            ban_tags=self.ban_tags,
            format=self.format,
            temperature=self.temperature,
            top_p=self.top_p,
            min_p=self.min_p,
            top_k=self.top_k,
            device=device,
        )

        logger.info("Upscale Done")
        logger.info("TIPO Output Prompt:")
        print("====================================================================================================")
        print(f"- Prompt:\n{prompt}")
        print("====================================================================================================")
        print(f"- User Prompt:\n{user_prompt}")
        print("====================================================================================================")
        print(f"- Unformatted Prompt:\n{unformatted_prompt}")
        print("====================================================================================================")
        print(f"- Unformatted User Prompt:\n{unformatted_user_prompt}")
        print("====================================================================================================\n")

        return TipoPromptOutput(
            prompt=prompt,
            user_prompt=user_prompt,
            unformatted_prompt=unformatted_prompt,
            unformatted_user_prompt=unformatted_user_prompt,
        )



@invocation(
    "TIPOOperation",
    title="TIPO Operation",
    tags=["tipo", "prompt"],
    category="prompt",
    version="1.0.0"
)
class TIPOOperation(BaseInvocation):
    """Prompt Upscale with Full Output"""

    current_model: ClassVar[None] = None

    tags: Optional[str] = InputField(
        default="",
        description="enter your tags",
        ui_component=UIComponent.Textarea
    )
    nl_prompt: Optional[str] = InputField(
        default="",
        description="enter your natural language",
        ui_component=UIComponent.Textarea
    )
    ban_tags: Optional[str] = InputField(
        default="",
        description="enter your tags need to ban",
        ui_component=UIComponent.Textarea
    )
    tipo_model: TIPO_MODEL_TYPES = InputField( # type: ignore
        default="KBlueLeaf/TIPO-200M | TIPO-200M-40Btok-F16.gguf",
        description="model type for prompt generation"
    )
    operation: TIPO_OPERATION_LIST = InputField( # type: ignore
        default=sorted(OPERATION_LIST)[0],
        description="operation"
    )
    width: Optional[int] = InputField(
        default=1024,
        description="none"
    )
    height: Optional[int] = InputField(
        default=1024,
        description="none"
    )
    temperature: Optional[float] = InputField(
        default=0.5,
        description="adjust the degree of random"
    )
    top_p: Optional[float] = InputField(
        default=0.95,
        description="adjust the degree of unconfident tokens"
    )
    min_p: Optional[float] = InputField(
        default=0.05,
        description="adjust the degree of unconfident tokens"
    )
    top_k: Optional[int] = InputField(
        default=80,
        description="adjust the degree of unconfident tokens"
    )
    tag_length: PROMPT_LENGTH_TYPES = InputField(
        default="long",
        description="adjust tag length"
    )
    nl_length: PROMPT_LENGTH_TYPES = InputField(
        default="long",
        description="adjust natural language length"
    )
    seed: Optional[int] = InputField(
        default=1,
        description="seed for upsampling tags"
    )
    device: Optional[Literal["Auto", "CPU", "CUDA"]] = InputField(
        default="Auto",
        description="choice device to load tipo model"
    )


    def execute(
        self,
        tipo_model: str,
        tags: str,
        nl_prompt: str,
        width: int,
        height: int,
        seed: int,
        tag_length: str,
        nl_length: str,
        ban_tags: str,
        operation: str,
        temperature: float,
        top_p: float,
        min_p: float,
        top_k: int,
        device: str,
    ):
        global current_model
        if (tipo_model, device) != current_model:
            if " | " in tipo_model:
                model_name, gguf_name = tipo_model.split(" | ")
                target_file = f"{model_name.split('/')[-1]}_{gguf_name}"
                if str(models.model_dir / target_file) not in models.list_gguf():
                    models.download_gguf(model_name, gguf_name)
                target = os.path.join(str(models.model_dir), target_file)
                gguf = True
            else:
                target = tipo_model
                gguf = False
            try:
                models.load_model(target, gguf, device=device)
            except:
                models.load_model(target, gguf, device="cpu")
            # current_model = (tipo_model, device)
            TIPOOperation.current_model = tipo_model
        aspect_ratio = width / height
        prompt_without_extranet = tags
        prompt_parse_strength = parse_prompt_attention(prompt_without_extranet)

        nl_prompt_wihtout_extranet = nl_prompt
        nl_prompt_parse_strength = parse_prompt_attention(nl_prompt)
        nl_prompt = ""
        strength_map_nl = []
        for part, strength in nl_prompt_parse_strength:
            nl_prompt += part
            if strength == 1:
                continue
            strength_map_nl.append((part, strength))

        black_list = [tag.strip() for tag in ban_tags.split(",") if tag.strip()]
        tipo.BAN_TAGS = black_list
        all_tags = []
        strength_map = {}
        for part, strength in prompt_parse_strength:
            part_tags = [tag.strip() for tag in part.strip().split(",") if tag.strip()]
            all_tags.extend(part_tags)
            if strength == 1:
                continue
            for tag in part_tags:
                strength_map[tag] = strength

        tag_length = tag_length.replace(" ", "_")
        org_tag_map = seperate_tags(all_tags)
        meta, operations, general, nl_prompt = tipo_single_request(
            org_tag_map,
            nl_prompt,
            tag_length_target=tag_length,
            nl_length_target=nl_length,
            operation=operation,
        )
        meta["aspect_ratio"] = f"{aspect_ratio:.1f}"

        tag_map, _ = tipo_runner(
            meta,
            operations,
            general,
            nl_prompt,
            temperature=temperature,
            seed=seed,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
        )

        addon = {
            "tags": [],
            "nl": "",
        }
        for cate in tag_map.keys():
            if cate == "generated" and addon["nl"] == "":
                addon["nl"] = tag_map[cate]
                continue
            if cate == "extended":
                extended = tag_map[cate]
                addon["nl"] = extended
                continue
            if cate not in org_tag_map:
                continue
            for tag in tag_map[cate]:
                if tag in org_tag_map[cate]:
                    continue
                addon["tags"].append(tag)
        addon = apply_strength(addon, strength_map, strength_map_nl)
        addon["user_tags"] = prompt_without_extranet
        addon["user_nl"] = nl_prompt_wihtout_extranet

        tag_map = apply_strength(tag_map, strength_map, strength_map_nl)
        return (
            tag_map,
            addon,
        )


    def invoke(self, context: InvocationContext) -> TipoOperationOutput:
        if self.device == "Auto":
            device = TorchDevice.choose_torch_device()
        elif self.device == "CPU":
            device = "cpu"
        else:
            device = "cuda"

        logger.info(f"Device: {device}")
        logger.info(f"Upscaling Prompt With Operation: {self.operation}")
        full_output, addon_output = self.execute(
            tipo_model=self.tipo_model,
            tags=self.tags,
            nl_prompt=self.nl_prompt,
            width=self.width,
            height=self.height,
            seed=self.seed,
            tag_length=self.tag_length,
            nl_length=self.nl_length,
            ban_tags=self.ban_tags,
            operation=self.operation,
            temperature=self.temperature,
            top_p=self.top_p,
            min_p=self.min_p,
            top_k=self.top_k,
            device=device
        )

        logger.info("Upscale Done")
        logger.info("TIPO Operation Node Output:")
        print("====================================================================================================")
        print(f"- Full Output:")
        print("\n".join(f"{key}: {value}" for key, value in full_output.items()))
        print("====================================================================================================")
        print(f"- Addon Output:")
        print("\n".join(f"{key}: {value}" for key, value in addon_output.items()))
        print("====================================================================================================\n")

        return TipoOperationOutput(
            full_output=TIPOOperationField(data=full_output),
            addon_output=TIPOOperationField(data=addon_output)
        )


@invocation(
    "TIPOFormat",
    title="TIPO Format",
    tags=["tipo", "prompt"],
    category="prompt",
    version="1.0.0"
)
class TIPOFormat(BaseInvocation):
    """Format TIPO Operation Output"""

    full_output: TIPOOperationField = InputField(
        default=TIPOOperationField(data={}),
        description="tipo operation node full output"
    )
    addon_output: TIPOOperationField = InputField(
        default=TIPOOperationField(data={}),
        description="tipo operation node addon output"
    )
    format: Optional[str] = InputField(
        default="""<|special|>, 
<|characters|>, <|copyrights|>, 
<|artist|>, 

<|general|>,

<|extended|>.

<|quality|>, <|meta|>, <|rating|>""",
        description="output tags format",
        ui_component=UIComponent.Textarea
    )


    def execute(
        self,
        full_output: list,
        addon_output: list,
        format: str,
    ):
        tags = addon_output.pop("user_tags", "")
        nl_prompt = addon_output.pop("user_nl", "")
        addon = addon_output
        tag_map = full_output

        prompt_without_extranet = tags
        prompt_parse_strength = parse_prompt_attention(prompt_without_extranet)

        nl_prompt_parse_strength = parse_prompt_attention(nl_prompt)
        nl_prompt = ""
        strength_map_nl = []
        for part, strength in nl_prompt_parse_strength:
            nl_prompt += part
            if strength == 1:
                continue
            strength_map_nl.append((part, strength))

        all_tags = []
        strength_map = {}
        for part, strength in prompt_parse_strength:
            part_tags = [tag.strip() for tag in part.strip().split(",") if tag.strip()]
            all_tags.extend(part_tags)
            if strength == 1:
                continue
            for tag in part_tags:
                strength_map[tag] = strength

        org_tag_map = seperate_tags(all_tags)
        meta, operations, general, nl_prompt = parse_tipo_request(
            org_tag_map,
            nl_prompt,
        )

        org_formatted_prompt = parse_tipo_result(
            apply_tipo_prompt(
                meta,
                general,
                nl_prompt,
                "short_to_tag_to_long",
                "long",
                True,
                gen_meta=True,
            )
        )
        org_formatted_prompt = apply_strength(
            org_formatted_prompt, strength_map, strength_map_nl
        )
        formatted_prompt_by_user = apply_format(org_formatted_prompt, format)
        unformatted_prompt_by_user = tags + nl_prompt
        formatted_prompt_by_tipo = apply_format(tag_map, format)
        unformatted_prompt_by_tipo = (
            tags + ", " + ", ".join(addon["tags"]) + "\n" + addon["nl"]
        )

        return (
            formatted_prompt_by_tipo,
            formatted_prompt_by_user,
            unformatted_prompt_by_tipo,
            unformatted_prompt_by_user,
        )


    def invoke(self, context: InvocationContext) -> TipoPromptOutput:
        logger.info("Fromating TIPO Operation Node Output")

        prompt, user_prompt, unformatted_prompt, unformatted_user_prompt = self.execute(
            full_output=self.full_output.data,
            addon_output=self.addon_output.data,
            format=self.format
        )

        logger.info("Fromat Done")
        logger.info("TIPO Output Prompt:")
        print("====================================================================================================")
        print(f"- Prompt:\n{prompt}")
        print("====================================================================================================")
        print(f"- User Prompt:\n{user_prompt}")
        print("====================================================================================================")
        print(f"- Unformatted Prompt:\n{unformatted_prompt}")
        print("====================================================================================================")
        print(f"- Unformatted User Prompt:\n{unformatted_user_prompt}")
        print("====================================================================================================\n")

        return TipoPromptOutput(
            prompt=prompt,
            user_prompt=user_prompt,
            unformatted_prompt=unformatted_prompt,
            unformatted_user_prompt=unformatted_user_prompt,
        )



invoke_logger.info("Load TIPO Node Done")