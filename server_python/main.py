import http
import os
import time
from flask import Flask, request, jsonify
from googletrans import Translator, LANGUAGES as googletrans_languages
from dotenv import load_dotenv
import requests
from http import HTTPStatus
from util import degreekify, greekify
import universal_translator.llm_translate
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

load_dotenv()

app = Flask(__name__)
translator = Translator()

PORT = int(os.getenv("PORT", 8080))
API_TOKEN = os.getenv("API_TOKEN")
COPTIC_TO_ENGLISH_ENDPOINT = os.getenv("COPTIC_TO_ENGLISH_ENDPOINT")
ENGLISH_TO_COPTIC_ENDPOINT = os.getenv("ENGLISH_TO_COPTIC_ENDPOINT")
MAX_RETRIES = 5

ENGLISH = "en"
SAHIDIC_COPTIC = "cop_sah"
BOHAIRIC_COPTIC = "cop_boh"

COPTIC_LANGUAGES = [SAHIDIC_COPTIC, BOHAIRIC_COPTIC]

COPTIC_DIALECT_TAG = {
    SAHIDIC_COPTIC: "з",
    BOHAIRIC_COPTIC: "б",
}

GENERATION_CONFIG = {
    "max_length": 20,
    "max_new_tokens": 128,
    "min_new_tokens": 1,
    "min_length": 0,
    "early_stopping": True,
    "do_sample": False,
    "num_beams": 5,
    "num_beam_groups": 1,
    "top_k": 50,
    "top_p": 0.95,
    "temperature": 1.0,
    "diversity_penalty": 0.0,
}

HEADERS = {"Content-Type": "application/json", "Authorization": "Bearer " + API_TOKEN}

limiter = Limiter(
    app=app,
    key_func=get_remote_address,  # Uses IP address to track requests
    default_limits=["200 per day", "50 per hour"]  # Default limits
)


def gtranslate(text, src, tgt):
    return translator.translate(text, src=src, dest=tgt).text


def translate_universal(text: str, src: str, tgt: str, model_name: str) -> tuple[str, HTTPStatus]:
    if src in COPTIC_LANGUAGES:
        pivot_lang = ENGLISH
        translation, status = get_coptic_translation(text, src, pivot_lang)

        if status != 200:
            return translation, status
        
        if tgt == pivot_lang:
            return translation, HTTPStatus.OK
        
        return translate_universal(translation, pivot_lang, tgt, model_name)
        
    if tgt in COPTIC_LANGUAGES:
        pivot_lang = ENGLISH
        if src == pivot_lang:
            return get_coptic_translation(text, src, tgt)
        
        pivot_translation, status = translate_universal(text, src, pivot_lang)
        if status != HTTPStatus.OK:
            return pivot_translation, status
        
        return translate_universal(pivot_translation, pivot_lang, tgt, model_name)

    # Use googletrans, when possible
    if src in googletrans_languages and tgt in googletrans_languages:
        translation = gtranslate(text, src, tgt)
        return translation, HTTPStatus.OK
    
    # Otherwise, just use our universal translator
    translation_response = universal_translator.llm_translate.translate_universal(src, tgt, text, model_name)
    # TODO: record the costs
    print(f"Input cost: {translation_response.input_cost:.6f}")
    print(f"Output cost: {translation_response.output_cost:.6f}")
    print(f"Total cost: {translation_response.input_cost + translation_response.output_cost:.6f}")
    return translation_response.translation.text, HTTPStatus.OK


# Only when one of src or tgt is coptic
def preprocess(src, text, tgt):
    if src != ENGLISH and src not in COPTIC_LANGUAGES:
        raise ValueError(f"Invalid preprocessing of {src} to {tgt}")

    if src in COPTIC_LANGUAGES:
        text = greekify(text.lower())
        text = f"{COPTIC_DIALECT_TAG[src]} {text}"
    else:
        text = f"{COPTIC_DIALECT_TAG[tgt]} {text}"

    return text


def postprocess(tgt, text):
    if tgt in COPTIC_LANGUAGES:
        text = degreekify(text)
    return text


def get_coptic_translation(text, src, tgt) -> tuple[str | None, HTTPStatus]:
    if (src in COPTIC_LANGUAGES and tgt != ENGLISH) or (tgt in COPTIC_LANGUAGES and src != ENGLISH):
        raise ValueError(f"Cannot run coptic translation from {src} to {tgt}")
    if src not in COPTIC_LANGUAGES and tgt not in COPTIC_LANGUAGES:
        raise ValueError(f"Cannot run coptic translation from {src} to {tgt}")

    text = preprocess(src, text, tgt)
    
    api = (
        ENGLISH_TO_COPTIC_ENDPOINT
        if tgt in COPTIC_LANGUAGES
        else COPTIC_TO_ENGLISH_ENDPOINT
    )
    if src in COPTIC_LANGUAGES and tgt in COPTIC_LANGUAGES:
        api = COPTIC_TO_ENGLISH_ENDPOINT

    instance = {
        "inputs": [text],
        "parameters": GENERATION_CONFIG,
        "options": {"wait_for_model": True},
    }

    translation = None
    status = None
    error = None
    for _ in range(MAX_RETRIES):
        try:
            response = requests.post(api, json=instance, headers=HEADERS)
            result = response.json()
            if "error" in result:
                if "Input is too long" in result["error"]:
                    status = HTTPStatus.UNPROCESSABLE_ENTITY
                    break
            response.raise_for_status()
            translation = result[0]["generated_text"]
            status = HTTPStatus.OK
            break
        except Exception as e:
            time.sleep(5)
            error = e
            continue
    
    if error:
        raise error
    translation = postprocess(tgt, translation)
    return translation, status


@app.route("/translate", methods=["POST"])
@limiter.limit("50 per minute")
def translate():
    req = request.get_json()
    src, tgt, text = req["src"], req["tgt"], req["text"]
    model_name = req.get("model", "claude-3-5-sonnet-20241022")
    print(f"Translating from {src} to {tgt} with model {model_name}")
    print(f"Text: {text}")
    # Cut down length of text to 500 characters
    text = text[:500]
    try:
        translation, status = translate_universal(text, src, tgt, model_name)
    except ValueError as e:
        translation, status = f"{e.__class__.__name__}: {str(e)}", HTTPStatus.BAD_REQUEST
    except Exception as e:
        translation, status = f"{e.__class__.__name__}: {str(e)}", HTTPStatus.INTERNAL_SERVER_ERROR

    match status:
        case HTTPStatus.INTERNAL_SERVER_ERROR:
            return jsonify({"code": status, "message": f"InternalServerError: {translation}"}), 500
        case HTTPStatus.UNPROCESSABLE_ENTITY:
            return jsonify({"code": status, "message": f"InputTooLong: {translation}"}), 422
        case HTTPStatus.BAD_REQUEST:
            return jsonify({"code": status, "message": f"BadRequest: {translation}"})
        case HTTPStatus.OK:
            return jsonify({"code": status, "translation": translation})
        case _:
            return jsonify({"code": 500, "message": f"Unknown status code {status}"})

    

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=PORT)
