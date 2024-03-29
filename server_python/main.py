import http
import os
import time
from flask import Flask, request, jsonify
from googletrans import Translator
from dotenv import load_dotenv
import requests
from http import HTTPStatus
from util import degreekify, greekify

load_dotenv()

app = Flask(__name__)
translator = Translator()

PORT = int(os.getenv("PORT", 8080))
API_TOKEN = os.getenv("API_TOKEN")
COPTIC_TO_ENGLISH_ENDPOINT = os.getenv("COPTIC_TO_ENGLISH_ENDPOINT")
ENGLISH_TO_COPTIC_ENDPOINT = os.getenv("ENGLISH_TO_COPTIC_ENDPOINT")
MAX_RETRIES = 5

ENGLISH = "en"
COPTIC = "cop"
ARABIC = "ar"

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


def gtranslate(text, src, tgt):
    return translator.translate(text, src=src, dest=tgt).text


def preprocess(src, text):
    if src == ARABIC:
        text = gtranslate(text, ARABIC, ENGLISH)
    elif src == COPTIC:
        text = greekify(text.lower())

    return text


def postprocess(tgt, text):
    if tgt == COPTIC:
        text = degreekify(text)
    elif tgt == ARABIC:
        text = gtranslate(text, ENGLISH, ARABIC)
    return jsonify({"code": 200, "translation": text})


@app.route("/translate", methods=["POST"])
def translate():
    req = request.get_json()
    src, tgt, text = req["src"], req["tgt"], req["text"]
    print(f"src: {src}, tgt: {tgt}, text: {text}")
    if (src, tgt) == (ARABIC, ENGLISH) or (src, tgt) == (ENGLISH, ARABIC):
        try:
            return postprocess(tgt=tgt, text=gtranslate(text, src, tgt)), 200
        except Exception as e:
            print(f"Error: {e}")
            return jsonify({"code": 500, "message": "InternalServerError"}), 500

    text = preprocess(src=src, text=text)
    api = ENGLISH_TO_COPTIC_ENDPOINT if tgt == COPTIC else COPTIC_TO_ENGLISH_ENDPOINT

    instance = {
        "inputs": [text],
        "parameters": GENERATION_CONFIG,
        "options": {"wait_for_model": True},
    }

    translation = None
    status = None

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
            status = 200
            break
        except Exception as e:
            time.sleep(5)
            status = HTTPStatus.INTERNAL_SERVER_ERROR
            print(f"Error: {e}")
            continue

    match status:
        case HTTPStatus.INTERNAL_SERVER_ERROR:
            return jsonify({"code": status, "message": "InternalServerError"}), 500
        case HTTPStatus.UNPROCESSABLE_ENTITY:
            return jsonify({"code": status, "message": "InputTooLong"}), 422
        case HTTPStatus.OK:
            return postprocess(tgt=tgt, text=translation), 200


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=PORT)
