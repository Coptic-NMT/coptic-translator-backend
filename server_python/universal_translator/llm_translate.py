import json
from pathlib import Path
import anthropic
import openai
import os

languages = json.loads(Path('server_python/universal_translator/languages.json').read_text())
client_anthropic = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
client_openai = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'])

def get_language(code: str):
    language = next((language for language in languages if language['flores_code'] == code), None)
    if not language:
        raise ValueError(f"Invalid code {code}")
    return language


USER_PROMPT = """Translate the following from {src_name} to {tgt_name}. Do not provide any output text besides for the translation.

English: {text}"""

ASSISTANT_PROMPT = "{tgt_name}:"

def translate_universal(src_code, tgt_code, text):
    src = get_language(src_code)
    tgt = get_language(tgt_code)

    # Prepare the prompt for the translation
    user_prompt = USER_PROMPT.format(src_name=src['name'], tgt_name=tgt['name'], text=text)
    assistant_prompt = ASSISTANT_PROMPT.format(tgt_name=tgt['name']) 
    # Perform the translation using the anthropic client
    response = client_anthropic.messages.create(
        model="claude-3-5-sonnet-20240620",  # Assuming a model name, replace with the correct one
        max_tokens=100,  # Adjust as needed
        temperature=0,  # Adjust as needed
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
            
        ]
    )

    # Extract the translation from the response
    translation = response.content[0].text.strip()

    return translation
    

