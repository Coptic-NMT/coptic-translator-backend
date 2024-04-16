from typing import Dict
import numpy as np
import torch
from transformers import Pipeline
from transformers.utils import ModelOutput
from transformers import pipeline
from transformers.pipelines import PIPELINE_REGISTRY
from transformers import AutoModelForSeq2SeqLM
from huggingface_hub import Repository

SAHIDIC_TAG = "з"
BOHAIRIC_TAG = "б"

from transformers import GenerationConfig

GENERATION_CONFIG = GenerationConfig(
    max_length=20,
    max_new_tokens=128,
    min_new_tokens=1,
    min_length=0,
    early_stopping=True,
    do_sample=True,
    num_beams=5,
    num_beam_groups=1,
    top_k=50,
    top_p=0.95,
    temperature=1.0,
    diversity_penalty=0.0,
    output_scores=True,
    return_dict_in_generate=True,
)


class CopticEnglishPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "from_bohairic" in kwargs and kwargs["from_bohairic"]:
            preprocess_kwargs["from_bohairic"] = True
        forward_kwargs = {}
        if "output_confidence" in kwargs and kwargs["output_confidence"]:
            forward_kwargs["output_confidence"] = True

        return preprocess_kwargs, forward_kwargs, {}

    def preprocess(self, text, from_bohairic=False):
        text = greekify(text.lower())

        if from_bohairic:
            text = f"{BOHAIRIC_TAG} {text}"
        else:
            text = f"{SAHIDIC_TAG} {text}"

        return self.tokenizer.encode(text, return_tensors="pt")

    def _forward(self, input_tensors, output_confidence=False) -> ModelOutput:
        outputs = self.model.generate(
            input_tensors[:, : self.tokenizer.model_max_length],
            generation_config=GENERATION_CONFIG,
        )

        translated_text = self.tokenizer.decode(
            outputs.sequences[0], skip_special_tokens=True
        )

        if output_confidence:
            scores = outputs.scores
            confidences = [
                torch.softmax(score, dim=-1).max().item() for score in scores
            ]
            num_words = len(translated_text.split())
            # scale the predicition probability by the number of words in the sentence
            scaled_probability = np.exp(sum(np.log(confidences)) / num_words)
            return translated_text, scaled_probability

        return translated_text, None

    def postprocess(self, outputs):
        text, confidence = outputs
        if confidence is None:
            return {
                "translation": text,
            }
        return {
            "translation": text,
            "confidence": confidence,
        }


COPTIC_TO_GREEK = {
    "ⲁ": "α",
    "ⲃ": "β",
    "ⲅ": "γ",
    "ⲇ": "δ",
    "ⲉ": "ε",
    "ⲋ": "ϛ",
    "ⲍ": "ζ",
    "ⲏ": "η",
    "ⲑ": "θ",
    "ⲓ": "ι",
    "ⲕ": "κ",
    "ⲗ": "λ",
    "ⲙ": "μ",
    "ⲛ": "ν",
    "ⲝ": "ξ",
    "ⲟ": "ο",
    "ⲡ": "π",
    "ⲣ": "ρ",
    "ⲥ": "σ",
    "ⲧ": "τ",
    "ⲩ": "υ",
    "ⲫ": "φ",
    "ⲭ": "χ",
    "ⲯ": "ψ",
    "ⲱ": "ω",
    "ϣ": "s",
    "ϥ": "f",
    "ϧ": "k",
    "ϩ": "h",
    "ϫ": "j",
    "ϭ": "c",
    "ϯ": "t",
}


def greekify(coptic_text):
    chars = []
    for c in coptic_text:
        l_c = c.lower()
        chars.append(COPTIC_TO_GREEK.get(l_c, l_c))
    return "".join(chars)