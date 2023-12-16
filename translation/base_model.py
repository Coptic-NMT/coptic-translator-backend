import json
import os
from pathlib import Path
from typing import Optional
from attr import dataclass
import sacrebleu
from datasets import Dataset
from tqdm import tqdm
import torch.nn as nn
from hashlib import sha256


@dataclass
class GenerationConfig:
    """
    Config for generation. Can be used to configure the generation process.
    max_length represents the maximum length of the generated sequence.
    max_new_tokens represents the maximum number of new tokens that can be generated.
    min_length represents the minimum length of the generated sequence.
    min_new_tokens represents the minimum number of new tokens that can be generated.
    early_stopping represents whether generation should stop when the model is confident in its prediction.
    do_sample represents whether sampling should be used when generating.
    num_beams represents the number of beams to use when generating. Default is 1, which means greedy search.
    num_beam_groups represents the number of groups to use when generating. Default is 1, which means no group.
    """

    max_length: int = 20
    max_new_tokens: Optional[int] = None
    min_length: int = 0
    min_new_tokens: Optional[int] = None
    early_stopping: bool = True
    do_sample: bool = False
    num_beams: int = 1
    num_beam_groups: int = 1
    top_k: int = 50
    top_p: float = 0.95
    temperature: float = 1.0
    diversity_penalty: float = 0.0

    def save(
        self,
        path: str,
        dataset_dir: Optional[str] = None,
        dataset_task: Optional[str] = None,
    ):
        params = self.__dict__.copy()
        if dataset_dir:
            params["dataset_dir"] = dataset_dir
        if dataset_task:
            params["dataset_task"] = dataset_task
        with open(path, "w") as f:
            json.dump(params, f, indent=4)

    def hash_fields(self):
        sha = sha256()
        for value in self.__dict__.values():
            sha.update(str(hash(value)).encode("utf-8"))
        return sha.hexdigest()[:8]


class BaseTranslationModel:
    def __init__(
        self,
        model_name: str,
        src_language: str,
        tgt_language: str,
        model: Optional[nn.Module],
        save_to_disk=True,
    ):
        self.model_name = model_name
        self.src_language = src_language
        self.tgt_language = tgt_language
        self.model = model
        self.dir_path = os.path.join(
            "models", f"{model_name}-{src_language}-{tgt_language}/"
        )
        if save_to_disk:
            # Prevent overwriting the model if it already exists.
            if os.path.exists(self.dir_path):
                raise FileExistsError(
                    f"Model {model_name} at path {self.dir_path} already exists. Did you mean to load using from_pretrained?"
                )
            else:
                os.makedirs(self.dir_path)

        self.save_to_disk = save_to_disk

    def translate(self, src_sentence: str, config: GenerationConfig):
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()

    @classmethod
    def from_pretrained(cls, path: str):
        raise NotImplementedError()

    def _hash_data_with_config(self, test_dataset, config):
        sha = sha256()
        sha.update(test_dataset._fingerprint.encode("utf-8"))
        sha.update(config.hash_fields().encode("utf-8"))
        return sha.hexdigest()[:8]

    def _old_hash_data_with_config(self, test_dataset, config, use_old_hash=False):
        data_cache_files = test_dataset.cache_files
        sha = sha256()
        for data_cache_file in data_cache_files:
            sha.update(data_cache_file["filename"].encode("utf-8"))
        sha.update(config.hash_fields().encode("utf-8"))
        return sha.hexdigest()[:8]

    def _apply_kwargs(self, config, **kwargs):
        for kwarg in kwargs:
            if hasattr(config, kwarg):
                setattr(config, kwarg, kwargs[kwarg])
            else:
                raise ValueError(f"Invalid kwarg {kwarg}.")

    def translate_test_data(self, test_dataset, config=GenerationConfig(), **kwargs):
        """
        Translates test_dataset using the model and saves the translations to a file.
        Takes a GenerationConfig as input, which can be used to configure the generation process.
        If a kwarg is passed, it will override the corresponding attribute in the config.
        """
        if not isinstance(test_dataset, Dataset):
            raise ValueError(
                f"test_dataset must be of type Dataset, but got {type(test_dataset)}."
            )
        print(f"Computing translations from {self.src_language}-{self.tgt_language}.")
        self._apply_kwargs(config, **kwargs)

        # backwards compatibility: look for old hash as well
        old_hash = self._old_hash_data_with_config(test_dataset, config)
        config_hash = self._hash_data_with_config(test_dataset, config)

        data_dir = os.path.join(self.dir_path, "data", config_hash)
        if not os.path.exists(data_dir):
            # if old hash path exists, rename to new hash path
            old_dir = os.path.join(self.dir_path, "data", old_hash)
            if not os.path.exists(old_dir):
                os.makedirs(data_dir)
            else:
                # rename
                print(f"Renaming f{old_dir} to {data_dir}")
                os.rename(old_dir, data_dir)

        translation_file = os.path.join(data_dir, f"translations.{self.tgt_language}")

        if os.path.exists(translation_file):
            print(f"Translation file already exists. Skipping computation.")
            return

        config.save(
            os.path.join(data_dir, "generation_config.json"),
            test_dataset.data_dir,
            f"{self.src_language}-{self.tgt_language}",
        )

        with open(translation_file, "w") as translations:
            for language_pair in tqdm(
                test_dataset["translation"], total=len(test_dataset)
            ):
                test_sentence = language_pair[self.src_language]
                translations.write(self.translate(test_sentence, config) + "\n")

        print("Translations completed. Computing metrics...")
        bleu = self.compute_bleu(test_dataset, config)
        chrf = self.compute_chrf(test_dataset, config)
        print(bleu)
        print(chrf)
        with open(os.path.join(data_dir, "metrics.txt"), "w") as metrics:
            metrics.write(str(bleu) + "\n")
            metrics.write(str(chrf) + "\n")

    def compute_bleu(self, test_dataset: Dataset, config=GenerationConfig(), max_num_translations = None, **kwargs):
        # TODO: improve caching so that multiple test sets can be used
        self._apply_kwargs(config, **kwargs)
        data_hash = self._hash_data_with_config(test_dataset, config)
        translation_file = os.path.join(
            self.dir_path, "data", data_hash, f"translations.{self.tgt_language}"
        )
        if not os.path.exists(translation_file):
            print(
                f"Could not find cached dataset at {translation_file}. Computing translations..."
            )
            self.translate_test_data(test_dataset, config)

        max_num_translations = len(test_dataset) if max_num_translations is None else max_num_translations
        translations = Path(translation_file).read_text().strip().split("\n")
        refs = [
            language_pair[self.tgt_language]
            for language_pair in test_dataset["translation"]
        ]
        bleu = sacrebleu.metrics.BLEU()
        return bleu.corpus_score(translations[:max_num_translations], [refs[:max_num_translations]])

    def compute_chrf(self, test_dataset: Dataset, config=GenerationConfig(), **kwargs):
        self._apply_kwargs(config, **kwargs)
        data_hash = self._hash_data_with_config(test_dataset, config)
        translation_file = os.path.join(
            self.dir_path, "data", data_hash, f"translations.{self.tgt_language}"
        )
        if not os.path.exists(translation_file):
            print(
                f"Could not find cached dataset at {translation_file}. Computing translations..."
            )
            self.translate_test_data(test_dataset, config)

        translations = Path(translation_file).read_text().strip().split("\n")
        refs = [
            language_pair[self.tgt_language]
            for language_pair in test_dataset["translation"]
        ]
        return sacrebleu.corpus_chrf(translations, [refs])
    

    def add_translations(self, test_dataset, config, **kwargs):
        self._apply_kwargs(config, **kwargs)
        # add translation column to dataset, including translation and sentence bleu
        data_hash = self._hash_data_with_config(test_dataset, config)
        translation_file = os.path.join(
            self.dir_path, "data", data_hash, f"translations.{self.tgt_language}"
        )
        if not os.path.exists(translation_file):
            print(
                f"Could not find cached dataset at {translation_file}. Computing translations..."
            )
            self.translate_test_data(test_dataset, config)
        # add translation column to dataset
        translations = Path(translation_file).read_text().strip().split("\n")
        test_dataset = test_dataset.map(lambda x: {"machine_translation": translations.pop(0)}, batched=False)
        # add sentence bleu score to dataset
        test_dataset = test_dataset.map(lambda x: {"sentence_bleu": sacrebleu.sentence_bleu(x["machine_translation"], [x["translation"][self.tgt_language]]).score})
        return test_dataset



    def print_stats(self, test_dataset: Dataset):
        print(
            f"Translation model {self.model_name} trained on {self.src_language}-{self.tgt_language}:"
        )
        print(self.compute_bleu(test_dataset))
        print(f"{self.compute_chrf(test_dataset)}\n")
