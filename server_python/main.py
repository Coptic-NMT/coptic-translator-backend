import os
import asyncio
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from googletrans import LANGUAGES as googletrans_languages
from dotenv import load_dotenv
import httpx
from http import HTTPStatus
import universal_translator.llm_translate
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

load_dotenv()

app = FastAPI()

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


def key_func(request: Request) -> str:
    forwarded = request.headers.get('X-Forwarded-For')
    if forwarded:
        return forwarded.split(',')[0]
    return get_remote_address(request)


limiter = Limiter(key_func=key_func)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"code": 429, "message": "Rate limit exceeded"}
    )


async def gtranslate(text: str, src: str, tgt: str):
    # HACK: for compliance let's use gpt-4o-mini-2024-07-18
    return await universal_translator.llm_translate.translate_universal(src, tgt, text, model_name="gpt-4o-mini-2024-07-18")


async def translate_universal(text: str, src: str, tgt: str, model_name: str) -> tuple[str, HTTPStatus]:
    if src in COPTIC_LANGUAGES:
        pivot_lang = ENGLISH
        translation, status = await get_coptic_translation(text, src, pivot_lang)

        if status != 200:
            return translation, status

        if tgt == pivot_lang:
            return translation, HTTPStatus.OK

        return await translate_universal(translation, pivot_lang, tgt, model_name)

    if tgt in COPTIC_LANGUAGES:
        pivot_lang = ENGLISH
        if src == pivot_lang:
            return await get_coptic_translation(text, src, tgt)

        pivot_translation, status = await translate_universal(text, src, pivot_lang, model_name)
        if status != HTTPStatus.OK:
            return pivot_translation, status

        return await translate_universal(pivot_translation, pivot_lang, tgt, model_name)

    # Use googletrans, when possible
    if src in googletrans_languages and tgt in googletrans_languages:
        translation_response = await gtranslate(text, src, tgt)
        return translation_response.translation.text, HTTPStatus.OK

    # Otherwise, just use our universal translator
    translation_response = await universal_translator.llm_translate.translate_universal(src, tgt, text, model_name)
    return translation_response.translation.text, HTTPStatus.OK


async def get_coptic_translation(text: str, src: str, tgt: str) -> tuple[str | None, HTTPStatus]:
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

    async with httpx.AsyncClient() as client:
        for _ in range(MAX_RETRIES):
            try:
                response = await client.post(api, json=instance, headers=HEADERS)
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
                await asyncio.sleep(5)
                error = e
                continue

    if error and status is None:
        raise error
    return translation, status


class TranslateRequest(BaseModel):
    src: str
    tgt: str
    text: str
    model: str = "gpt-4o-2024-08-06"


@app.post("/translate")
@limiter.limit("600 per day, 200 per hour, 50 per minute")
async def translate(request: Request, req: TranslateRequest):
    src, tgt, text = req.src, req.tgt, req.text
    model_name = req.model
    text = text[:500]
    try:
        translation, status = await translate_universal(text, src, tgt, model_name)
    except ValueError as e:
        translation, status = f"{e.__class__.__name__}: {str(e)}", HTTPStatus.BAD_REQUEST
    except Exception as e:
        translation, status = f"{e.__class__.__name__}: {str(e)}", HTTPStatus.INTERNAL_SERVER_ERROR

    match status:
        case HTTPStatus.INTERNAL_SERVER_ERROR:
            return JSONResponse(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                content={"code": status.value, "message": f"InternalServerError: {translation}"}
            )
        case HTTPStatus.UNPROCESSABLE_ENTITY:
            return JSONResponse(
                status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
                content={"code": status.value, "message": f"InputTooLong: {translation}"}
            )
        case HTTPStatus.BAD_REQUEST:
            return JSONResponse(
                status_code=HTTPStatus.BAD_REQUEST,
                content={"code": status.value, "message": f"BadRequest: {translation}"}
            )
        case HTTPStatus.OK:
            return {"code": status.value, "translation": translation}
        case _:
            return JSONResponse(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                content={"code": 500, "message": f"Unknown status code {status}"}
            )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
