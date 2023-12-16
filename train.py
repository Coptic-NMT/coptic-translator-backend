#!/home/bizon/anaconda3/envs/coptic/bin/python


import os
import datasets
import sys
import parse_data
from base_model import GenerationConfig
from huggingface_model import (
    HuggingFaceTranslationModel,
    HuggingFaceTranslationModelTrainingConfig,
)
from model import TranslationModel, TranslationTrainingConfig
from transformers import (
    AutoModel,
    AutoTokenizer,
    MarianMTModel,
    MarianConfig,
    BartTokenizer,
    BartForConditionalGeneration,
    BartConfig,
    BartTokenizerFast,
    Seq2SeqTrainingArguments,
    AutoModelForSeq2SeqLM,
)
from datasets import Dataset
import data_utils
from data_utils import load_dataset
from config_consts import *

if __name__ == "__main__":
    try:
        [job_hash] = sys.argv[1:]

    except ValueError:
        print("Usage: python3 train.py <job_hash>")
        sys.exit(1)
    except IndexError:
        print("Usage: python3 train.py <job_hash>")
        sys.exit(1)

    # Load the dataset
    COPTIC_COL = "norm_romanized"
    # data_dict = parse_data.load_dataset_for_translation(
    #     "datasets/third-dataset", "cop", "eng", COPTIC_COL, "eng"
    # )

    # train_config = HuggingFaceTranslationModelTrainingConfig(
    #     job_hash,
    #     evaluation_strategy="steps",
    #     eval_steps=80,
    #     logging_steps=25,
    #     learning_rate=1e-5,
    #     weight_decay=0,
    #     per_device_train_batch_size=128,
    #     per_device_eval_batch_size=64,
    #     num_train_epochs=100,
    #     save_total_limit=100,
    #     label_smoothing_factor=0.5,
    # )

    # model = HuggingFaceTranslationModel(
    #     f"hf/tuned-final_attempt-{COPTIC_COL}-finetuned",
    #     "cop",
    #     "eng",
    #     tokenizer=AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-mul-en"),
    #     model=MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-mul-en"),
    # )
    # model.train(data_dict, train_config)
    import time
    model = HuggingFaceTranslationModel.from_pretrained(f"models/hf/en-mul-norm_group_greekified-finetuned-eng-cop")
    test_dir = "datasets/test_data"
    for test in os.listdir(test_dir):
        print("Loading", test)
        test_data = parse_data.load_test_set(os.path.join(test_dir, test), "cop", "eng", "norm_group_greekified", "eng")
        for num_beams in [4,8,16]:
            for num_beam_groups in [1,2,4]:
                print(f"Using {num_beams} beams, {num_beam_groups} beam groups")
                config = GenerationConfig(
                    max_new_tokens=128,
                    min_new_tokens=1,
                    num_beams=num_beams,
                    num_beam_groups=num_beam_groups,
                    diversity_penalty=(num_beam_groups // 2) * 0.25
                )
                start = time.time()
                model.translate_test_data(test_data, config)
                end = time.time()
                print(f"Time elapsed: {end-start}")
        print()

