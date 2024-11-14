import json
from pathlib import Path
import anthropic
import openai
import os
from dotenv import load_dotenv

load_dotenv()

languages = json.loads(Path('universal_translator/languages.json').read_text())
client_anthropic = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
client_openai = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'])

# TODO: make a choice between google_code or flores_code!
def get_language(code: str):
    language = next((language for language in languages if language['google_code'] == code or language['flores_code'] == code), None)
    if not language:
        raise ValueError(f"Invalid code {code}")
    return language


USER_PROMPT = """Translate the following from {src_name} to {tgt_name}. If the input is not correctly in the {src_name} language, just output a message saying "Error: wrong language". Do not provide any output text besides for the translation. Wrap your output in <translation></translation> tags.

{src_name}: {text}"""

ASSISTANT_PROMPT = "{tgt_name}: <translation>"

def translate_universal(src_code, tgt_code, text):
    src = get_language(src_code)
    tgt = get_language(tgt_code)

    # Prepare the prompt for the translation
    user_prompt = USER_PROMPT.format(src_name=src['name'], tgt_name=tgt['name'], text=text)
    assistant_prompt = ASSISTANT_PROMPT.format(tgt_name=tgt['name']) 
    # Perform the translation using the anthropic client
    response = client_anthropic.messages.create(
        model="claude-3-5-sonnet-20241022",
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
        stop_sequences=["</translation>"]
    )

    # Extract the translation from the response
    translation = response.content[0].text.strip()

    return translation
    

