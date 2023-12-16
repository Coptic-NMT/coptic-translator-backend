from huggingface_model import HuggingFaceTranslationModelTrainingConfig
from base_model import GenerationConfig

"""
Defines configuration constants for the translation app.
"""


BEAM_GENERATION_CONFIG = GenerationConfig(
    max_new_tokens=128,
    min_new_tokens=1,
    num_beams=5,
)

GREEDY_GENERATION_CONFIG = GenerationConfig(
    max_new_tokens=128,
    min_new_tokens=1,
)

SAMPLE_GENERATION_CONFIG = GenerationConfig(
    max_new_tokens=128,
    min_new_tokens=1,
    do_sample=True,
    top_k=50,
    top_p=0.5,
    temperature=0.7,
)

