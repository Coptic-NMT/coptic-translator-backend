import os
import time
from flask import Flask, request, jsonify
from googletrans import Translator, LANGUAGES as googletrans_languages
from dotenv import load_dotenv
import requests
from http import HTTPStatus
import universal_translator.llm_translate
from flask_limiter import Limiter

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

HEADERS = {"Content-Type": "application/json", "Authorization": f"Bearer {API_TOKEN}"}

def key_func():
    return request.headers.get('X-Forwarded-For', request.remote_addr).split(',')[0]

limiter = Limiter(
    app=app,
    key_func=key_func,  # Uses IP address to track requests
    default_limits=["600 per day", "200 per hour", "50 per minute"]
)


def gtranslate(text, src, tgt):
    # HACK: for compliance let's use gpt-4o-mini-2024-07-18
    return universal_translator.llm_translate.translate_universal(src, tgt, text, model_name="gpt-4o-mini-2024-07-18")



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
        
        pivot_translation, status = translate_universal(text, src, pivot_lang, model_name)
        if status != HTTPStatus.OK:
            return pivot_translation, status
        
        return translate_universal(pivot_translation, pivot_lang, tgt, model_name)

    # Use googletrans, when possible
    if src in googletrans_languages and tgt in googletrans_languages:
        translation_response = gtranslate(text, src, tgt)
        print(f"Input cost: {translation_response.input_cost:.6f}")
        print(f"Output cost: {translation_response.output_cost:.6f}")
        print(f"Total cost: {translation_response.input_cost + translation_response.output_cost:.6f}")
        return translation_response.translation.text, HTTPStatus.OK
    
    # Otherwise, just use our universal translator
    translation_response = universal_translator.llm_translate.translate_universal(src, tgt, text, model_name)
    # TODO: record the costs
    print(f"Input cost: {translation_response.input_cost:.6f}")
    print(f"Output cost: {translation_response.output_cost:.6f}")
    print(f"Total cost: {translation_response.input_cost + translation_response.output_cost:.6f}")
    return translation_response.translation.text, HTTPStatus.OK


def get_coptic_translation(text, src, tgt) -> tuple[str | None, HTTPStatus]:
    if (src in COPTIC_LANGUAGES and tgt != ENGLISH) or (tgt in COPTIC_LANGUAGES and src != ENGLISH):
        raise ValueError(f"Cannot run coptic translation from {src} to {tgt}")
    if src not in COPTIC_LANGUAGES and tgt not in COPTIC_LANGUAGES:
        raise ValueError(f"Cannot run coptic translation from {src} to {tgt}")

    if tgt in COPTIC_LANGUAGES:
        api = ENGLISH_TO_COPTIC_ENDPOINT
        instance = {
            "inputs": text,
            "to_bohairic": tgt == BOHAIRIC_COPTIC,
        }
    else:
        api = COPTIC_TO_ENGLISH_ENDPOINT
        instance = {
            "inputs": text,
            "from_bohairic": src == BOHAIRIC_COPTIC,
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
            translation = result[0]["translation"]
            status = HTTPStatus.OK
            break
        except Exception as e:
            time.sleep(5)
            error = e
            continue

    if error and status is None:
        raise error
    return translation, status


@app.route("/translate", methods=["POST"])
def translate():
    req = request.get_json()
    src, tgt, text = req["src"], req["tgt"], req["text"]
    model_name = req.get("model", "claude-3-5-sonnet-20241022")
    print(f"Translating from {src} to {tgt}")
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
            return jsonify({"code": status, "message": f"InternalServerError: {translation}"}), HTTPStatus.INTERNAL_SERVER_ERROR
        case HTTPStatus.UNPROCESSABLE_ENTITY:
            return jsonify({"code": status, "message": f"InputTooLong: {translation}"}), HTTPStatus.UNPROCESSABLE_ENTITY
        case HTTPStatus.BAD_REQUEST:
            return jsonify({"code": status, "message": f"BadRequest: {translation}"}), HTTPStatus.BAD_REQUEST
        case HTTPStatus.OK:
            return jsonify({"code": status, "translation": translation}), HTTPStatus.OK
        case _:
            return jsonify({"code": 500, "message": f"Unknown status code {status}"}), HTTPStatus.INTERNAL_SERVER_ERROR

    

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=PORT)
