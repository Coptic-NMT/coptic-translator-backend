import json
import logging
from pathlib import Path
import anthropic
import openai
import os
from dotenv import load_dotenv
from enum import Enum
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class Providers(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"


class Model(BaseModel):
    type: Providers
    name: str
    input_cost_per_token: float
    output_cost_per_token: float

class Translation(BaseModel):
    text: str

class TranslationResponse(BaseModel):
    translation: Translation
    input_cost: float
    output_cost: float

# OpenAI models first
OPENAI_MODELS = [
    Model(
        type=Providers.OPENAI,
        name='gpt-4o-2024-08-06',
        input_cost_per_token=2.5 * 10**-6,
        output_cost_per_token=10 * 10**-6
    ),
    Model(
        type=Providers.OPENAI,
        name='gpt-4o-mini-2024-07-18',
        input_cost_per_token=0.15 * 10**-6,
        output_cost_per_token=0.6 * 10**-6
    ),
]

ANTHROPIC_MODELS = [
    Model(
        type=Providers.ANTHROPIC,
        name='claude-3-5-sonnet-20241022',
        input_cost_per_token = 3 * 10**-6,
        output_cost_per_token = 15 * 10**-6
    ),
    Model(
        type=Providers.ANTHROPIC,
        name='claude-3-5-haiku-20241022',
        input_cost_per_token = 1 * 10**-6,
        output_cost_per_token = 5 * 10**-6
    ),
    Model(
        type=Providers.ANTHROPIC,
        name='claude-3-haiku-20240307',
        input_cost_per_token=.25 * 10 **-6,
        output_cost_per_token=1.25 * 10 **-6
    ),
]

# OpenAI models first in the list
MODELS = [
    *OPENAI_MODELS,
    *ANTHROPIC_MODELS,
]


load_dotenv()

languages = json.loads(Path('universal_translator/languages.json').read_text())
client_anthropic = anthropic.AsyncAnthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
client_openai = openai.AsyncOpenAI(api_key=os.environ['OPENAI_API_KEY'])

# TODO: make a choice between google_code or flores_code!
def get_language(code: str):
    language = next((language for language in languages if language['google_code'] == code or language['flores_code'] == code), None)
    if not language:
        raise ValueError(f"Invalid code {code}")
    return language


def get_model(model_name: str):
    model = next((model for model in MODELS if model.name == model_name), None)
    if not model:
        raise ValueError(f"Invalid model {model_name}. Valid models are: {', '.join([model.name for model in MODELS])}")
    return model


async def translate_claude(src: dict, tgt: dict, text: str, model: Model) -> TranslationResponse:
    USER_PROMPT = """Translate the following from {src_name} to {tgt_name}.  Wrap your output in <translation></translation> tags. If the input is not correctly in the {src_name} language, do your best attempt at translating to {tgt_name}, or, if it already translated, just copy the output into the <translation> tags.

    {src_name}: {text}"""

    ASSISTANT_PROMPT = "{tgt_name}: <translation>"

    # Prepare the prompt for the translation
    user_prompt = USER_PROMPT.format(src_name=src['name'], tgt_name=tgt['name'], text=text)
    assistant_prompt = ASSISTANT_PROMPT.format(tgt_name=tgt['name'])
    # Perform the translation using the anthropic client
    response = await client_anthropic.messages.create(
        model=model.name,
        max_tokens=400,
        temperature=0,
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": assistant_prompt
                    }
                ]
            }
        ],
        stop_sequences=["</translation>"],
        timeout=5
    )

    # Extract the translation from the response
    translation_text = response.content[0].text.strip()
    translation = Translation(text=translation_text)
    input_cost = response.usage.input_tokens * model.input_cost_per_token
    output_cost = response.usage.output_tokens * model.output_cost_per_token
    return TranslationResponse(translation=translation, input_cost=input_cost, output_cost=output_cost)


async def translate_openai(src: dict, tgt: dict, text: str, model: Model) -> TranslationResponse:
    USER_PROMPT = """Translate the following from {src_name} to {tgt_name}. If the input is not correctly in the {src_name} language, do your best attempt at translating to {tgt_name}, or, if it already translated, just copy the output.

    {src_name} input: {text}"""

    # Prepare the prompt for the translation
    user_prompt = USER_PROMPT.format(src_name=src['name'], tgt_name=tgt['name'], text=text)

    # Perform the translation using the OpenAI client
    response = await client_openai.beta.chat.completions.parse(
        model=model.name,
        temperature=0,
        messages=[
            {"role": "user", "content": user_prompt},
        ],
        response_format=Translation,
        timeout=5
    )

    # Extract the translation from the response
    translation = response.choices[0].message.parsed
    input_cost = response.usage.prompt_tokens * model.input_cost_per_token
    output_cost = response.usage.completion_tokens * model.output_cost_per_token
    return TranslationResponse(translation=translation, input_cost=input_cost, output_cost=output_cost)


# Default fallbacks: OpenAI first, then Anthropic
default_fallbacks = [get_model('gpt-4o-mini-2024-07-18'), get_model('claude-3-haiku-20240307')]

async def translate_universal(src_code, tgt_code, text, model_name = "gpt-4o-mini-2024-07-18", fallbacks=True) -> TranslationResponse:
    src = get_language(src_code)
    tgt = get_language(tgt_code)

    model = get_model(model_name)
    try:
        match model.type:
            case Providers.OPENAI:
                return await translate_openai(src, tgt, text, model)
            case Providers.ANTHROPIC:
                return await translate_claude(src, tgt, text, model)
            case _:
                raise ValueError(f"Invalid model type {model.type}")
    except Exception as e:
        fallbacks = default_fallbacks if fallbacks is True else fallbacks
        if fallbacks:
            # remove the provider from the list
            fallbacks = [fallback for fallback in fallbacks if fallback.type != model.type]
            logger.warning(f"{e}. Falling back to {[f.name for f in fallbacks]}")
            model_name = fallbacks[0].name
            return await translate_universal(src_code, tgt_code, text, model_name, fallbacks=fallbacks[1:])

        raise e
