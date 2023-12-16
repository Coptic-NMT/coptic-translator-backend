import matplotlib.pyplot as plt
import numpy as np
import torch
from collections import defaultdict
import json
import os
import random
from typing import DefaultDict, Dict, List
import torch
from transformers import BertTokenizer, AutoTokenizer
from huggingface_model import HuggingFaceTranslationModel
from base_model import BaseTranslationModel
import datasets
from datasets import Dataset, DatasetDict
from tqdm import tqdm
import pandas as pd
import sacrebleu
from trie import Trie


def generate_shuffled_data(
    language: str,
    dataset_dict,
    num_examples: int,
    tokenizer: BertTokenizer,
    p_split: float,
):
    """Generate shuffled data from a dataset, by rearranging tokens in each example."""
    seen_tokens = set()
    for split in dataset_dict:
        dataset = dataset_dict[split]
        dataset = datasets.Dataset.from_dict(dataset[:num_examples])
        dataset_dict[split] = dataset
        for example in dataset:
            sentence = example["translation"][language]
            seen_tokens.update(tokenizer.tokenize(sentence))

    special_tokens = set(tokenizer.all_special_tokens)
    seen_tokens.difference_update(special_tokens)

    vocab = list(seen_tokens)
    shuffled_vocab = list(vocab)
    random.shuffle(shuffled_vocab)
    vocab_map = {
        k: v
        for k, v in zip(vocab, shuffled_vocab)
        if k and v and k.isalpha() and v.isalpha()
    }
    vocab_map.update(
        {
            k: v
            for k, v in zip(vocab, shuffled_vocab)
            if k.startswith("##") and v.startswith("##")
        }
    )
    print(len(vocab_map))

    for k, v in vocab_map.items():
        prefix, v = (v[:2], v[:2]) if v.startswith("##") else ("", v)
        if len(v) <= 1:
            continue
        if random.random() < p_split:
            # split the token at a random position
            split = random.randint(1, len(v) - 1)
            vocab_map[k] = prefix + v[:split] + " " + v[split:]

    return dataset_dict.map(
        lambda example: {
            "translation": {
                "fr": tokenizer.convert_tokens_to_string(
                    [
                        vocab_map.get(token, token)
                        for token in tokenizer.tokenize(example["translation"]["fr"])
                    ]
                ),
                "en": example["translation"]["en"],
            }
        }
    )


def generate_shuffled_data_files(p_vals: List[float]):
    for p in p_vals:
        filename = f"datasets/copy_p_{p}_shuffled_fr-en"
        fr_dataset = load_dataset("wmt15", "fr-en")
        shuffled = generate_shuffled_data(
            "fr",
            fr_dataset,
            100_000,
            AutoTokenizer.from_pretrained("dbmdz/bert-base-french-europeana-cased"),
            p,
        )
        for split in shuffled:
            shuffled[split].to_json(os.path.join(filename, split, "translations.json"))
            print(
                f"Saved {split} to {os.path.join(filename, split, 'translations.json')}"
            )


def generate_synthetic_data(
    language: str, dataset_dict, max_dataset_size: int, data_map: Dict[str, str]
):
    """
    Perform substitutions on a dataset, using a data_map to map strings to their substitutes.
    """
    # create trie from data_map
    reduced_dataset_dict = DatasetDict()
    trie = Trie()
    for key, value in data_map.items():
        trie.add_word(key, value)

    for split in dataset_dict:
        dataset_split = dataset_dict[split]
        reduced_dataset_dict[split] = datasets.Dataset.from_dict(
            dataset_split[:max_dataset_size]
        )

    def apply_substitutions(sentence):
        curr_index = 0
        while curr_index < len(sentence):
            word, largest_substitution = trie.find_largest_substitution(
                sentence, curr_index
            )
            sentence = (
                sentence[:curr_index]
                + largest_substitution
                + sentence[curr_index + len(word) :]
            )
            curr_index += len(largest_substitution) + 1
        return sentence

    return reduced_dataset_dict.map(
        lambda example: {
            "translation": {
                language: apply_substitutions(example["translation"][language]),
                "en": example["translation"]["en"],
            }
        }
    )


def generate_caesered_data(
    language: str, dataset_dict, max_dataset_size: int, num_shifts: int
):
    """
    Perform caeser cipher on a dataset, and save to json.
    """
    from string import ascii_lowercase

    caeser_map = {c: chr(ord(c) + num_shifts) for c in ascii_lowercase}
    caesered_data = generate_synthetic_data(
        language, dataset_dict, max_dataset_size, caeser_map
    )
    for split in caesered_data:
        caesered_data[split].to_json(
            f"datasets/{num_shifts}_caesered_{language}-en/{split}/translations.json"
        )


def generate_vowel_shifted_data(language: str, dataset_dict, max_dataset_size: int):
    """
    Perform caeser cipher on a dataset, and save to json.
    """
    lower_vowels = ["a", "e", "i", "o", "u", "é"]
    upper_vowels = [vowel.upper() for vowel in lower_vowels]
    vowel_map = {
        vowel: lower_vowels[(i + 1) % len(lower_vowels)]
        for i, vowel in enumerate(lower_vowels)
    }
    vowel_map.update(
        {
            vowel: upper_vowels[(i + 1) % len(upper_vowels)]
            for i, vowel in enumerate(upper_vowels)
        }
    )
    shifted_data = generate_synthetic_data(
        language, dataset_dict, max_dataset_size, vowel_map
    )
    for split in shifted_data:
        shifted_data[split].to_json(
            f"datasets/vowel_shifted_{language}-en/{split}/translations.json"
        )


def get_most_frequent(
    language: str, dataset_dict: DatasetDict, max_dataset_size: int, tokenizer
):
    """Return the tokens in a dataset, sorted by greatest to least frequency."""
    token_freq_map = defaultdict(int)
    for split in dataset_dict:
        dataset = datasets.Dataset.from_dict(dataset_dict[split][:max_dataset_size])
        for example in dataset:
            sentence = example["translation"][language]
            for token in tokenizer.tokenize(sentence):
                text_token = "".join(c for c in token if c.isalpha())
                if token in tokenizer.all_special_tokens:
                    continue
                if not text_token:
                    continue
                token_freq_map[text_token] += 1
    return sorted(token_freq_map, key=token_freq_map.get, reverse=True)


def get_prefix_tokens(
    language: str, dataset, max_dataset_size: int, tokenizer, max_num=1000
):
    seen_tokens = []
    prefixes = []
    most_frequent = get_most_frequent(language, dataset, max_dataset_size, tokenizer)
    for index, token in enumerate(most_frequent):
        if len(token) < 3:
            continue
        if index > max_num:
            break
        in_tokens = [seen_token for seen_token in seen_tokens if token in seen_token]
        if len(in_tokens) > 0:
            prefixes.append((token, in_tokens))
        seen_tokens.append(token)
    # json.dump(prefixes, open(f"datasets/{language}_prefixes.json", "w"))
    import json

    json.dump(prefixes, open(f"datasets/{language}_prefixes.json", "w"))
    return prefixes


def generate_common_token_split_data_files(
    language: str, dataset_dict, max_dataset_size: int
):
    data_map = {"sont": "s ont", "peut": "peu t"}
    common_token_split_data = generate_synthetic_data(
        language, dataset_dict, max_dataset_size, data_map
    )
    for split in common_token_split_data:
        common_token_split_data[split].to_json(
            f"datasets/common_token_split_{language}-en/{split}/translations.json"
        )


def generate_common_token_shuffled_data_files(
    language: str, dataset_dict, max_dataset_size: int
):
    data_map = {"sont": "peut", "peut": "sont"}
    common_token_split_data = generate_synthetic_data(
        language, dataset_dict, max_dataset_size, data_map
    )
    for split in common_token_split_data:
        common_token_split_data[split].to_json(
            f"datasets/common_token_shuffle_{language}-en/{split}/translations.json"
        )


def generate_shuffled_most_common(
    language: str,
    dataset_dict,
    max_dataset_size: int,
    tokenizer,
    num_shuffled=1000,
    p_split=0,
):
    most_frequent = get_most_frequent(
        language, dataset_dict, max_dataset_size, tokenizer
    )
    most_frequent = most_frequent[:num_shuffled]
    to_shuffle = []
    for token in most_frequent:
        alpha_token = "".join(c for c in token if c.isalpha())
        if len(alpha_token) < 3:
            continue
        to_shuffle.append(alpha_token)
    shuffled = to_shuffle.copy()
    random.seed(42)
    random.shuffle(shuffled)
    data_map = dict(zip(to_shuffle, shuffled))
    for k, v in data_map.items():
        if random.random() < p_split:
            # split the token at a random position
            split = random.randint(1, len(v) - 1)
            data_map[k] = v[:split] + " " + v[split:]
    shuffled_most_common = generate_synthetic_data(
        language, dataset_dict, max_dataset_size, data_map
    )
    for split in shuffled_most_common:
        shuffled_most_common[split].to_json(
            f"datasets/shuffle_{num_shuffled}_most_common_p={p_split}_{language}-en/{split}/translations.json"
        )


def generate_split_most_common(
    language: str,
    dataset_dict,
    max_dataset_size: int,
    tokenizer,
    num_shuffled=1000,
    num_split=100,
):
    most_frequent = get_most_frequent(
        language, dataset_dict, max_dataset_size, tokenizer
    )
    most_frequent = most_frequent[:num_shuffled]
    to_split = set(most_frequent[:num_split])
    to_shuffle = []
    for token in most_frequent:
        alpha_token = "".join(c for c in token if c.isalpha())
        if len(alpha_token) < 3:
            continue
        to_shuffle.append(alpha_token)
    shuffled = to_shuffle.copy()
    random.seed(42)
    random.shuffle(shuffled)
    data_map = dict(zip(to_shuffle, shuffled))
    for k, v in data_map.items():
        if k in to_split:
            # split the token at a random position
            split = random.randint(1, len(v) - 1)
            data_map[k] = v[:split] + " " + v[split:]
    shuffled_most_common = generate_synthetic_data(
        language, dataset_dict, max_dataset_size, data_map
    )
    for split in shuffled_most_common:
        shuffled_most_common[split].to_json(
            f"datasets/shuffle_{num_shuffled}_most_common_split={num_split}_{language}-en/{split}/translations.json"
        )


def make_splits(
    dir_path: str,
    translation_task: str,
    train_split: float = 0.8,
    validation_split: float = 0.1,
    test_split: float = 0.1,
):
    """
    Split two parallel datasets into train, validation, and test sets. Datasets must be named as {src_language}.txt and {tgt_language}.txt.
    dir_path: path to directory containing source and target files.
    translation_task: string of the form "src_language-tgt_language".
    train_split: (Optional) fraction of dataset to use for training. Defaults to 0.8.
    valid_split: (Optional) fraction of dataset to use for validation. Defaults to 0.1.
    test_split: (Optional) fraction of dataset to use for testing. Defaults to 0.1.
    """
    # TODO: implement with streaming for very big datasets
    src_language, tgt_language = translation_task.split("-")
    src_path = os.path.join(dir_path, f"{src_language}.txt")
    tgt_path = os.path.join(dir_path, f"{tgt_language}.txt")
    if not os.path.exists(os.path.join(dir_path, src_path)):
        raise FileNotFoundError(f"{src_path} not found.")
    if not os.path.exists(os.path.join(dir_path, tgt_path)):
        raise FileNotFoundError(f"{tgt_path} not found.")
    src_lines = open(src_path, "r", encoding="utf-8").readlines()
    tgt_lines = open(tgt_path, "r", encoding="utf-8").readlines()
    if not len(src_lines) == len(tgt_lines):
        raise ValueError(
            f"Line count for src {src_path} ({len(src_lines)}) does not match line count for tgt {tgt_path} ({len(tgt_lines)})."
        )
    num_lines = len(src_lines)

    num_train, num_validation, num_test = (
        int(train_split * num_lines),
        int(validation_split * num_lines),
        int(test_split * num_lines),
    )
    train_dir, validation_dir, test_dir = (
        os.path.join(dir_path, "train"),
        os.path.join(dir_path, "validation"),
        os.path.join(dir_path, "test"),
    )
    for directory in [train_dir, validation_dir, test_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
        else:
            raise FileExistsError(
                f"{directory} already exists. Cannot generate splits."
            )

    for lines, language in zip([src_lines, tgt_lines], [src_language, tgt_language]):
        train_path = os.path.join(train_dir, f"{language}.txt")
        validation_path = os.path.join(validation_dir, f"{language}.txt")
        test_path = os.path.join(test_dir, f"{language}.txt")
        for filename in [train_path, validation_path, test_path]:
            if os.path.exists(train_path):
                raise FileExistsError(f"{filename} already exists.")

        with open(train_path, "w") as train:
            train.writelines(lines[:num_train])

        with open(validation_path, "w") as train:
            train.writelines(lines[num_train : num_train + num_validation])

        with open(test_path, "w") as train:
            train.writelines(lines[num_train + num_validation :])


def translations(src_filepath: str, tgt_filepath, src_language: str, tgt_language: str):
    """Load parallel data from files in data_dir."""
    if not os.path.exists(src_filepath):
        raise FileNotFoundError(f"{src_filepath} not found.")
    if not os.path.exists(tgt_filepath):
        raise FileNotFoundError(f"{tgt_filepath} not found.")

    with open(src_filepath, "r", encoding="utf-8") as src_file, open(
        tgt_filepath, "r", encoding="utf-8"
    ) as tgt_file:
        return [
            {src_language: src_sentence.strip(), tgt_language: tgt_sentence.strip()}
            for src_sentence, tgt_sentence in zip(src_file, tgt_file)
        ]


def set_translation_json(data_dir, src_language: str, tgt_language: str):
    if os.path.exists(os.path.join(data_dir, "translations.json")):
        raise FileExistsError(
            f"{os.path.join(data_dir, 'translations.json')} already exists."
        )

    src_filepath = os.path.join(data_dir, f"{src_language}.txt")
    tgt_filepath = os.path.join(data_dir, f"{tgt_language}.txt")
    """Load parallel data from files in data_dir into a json file."""
    if not os.path.exists(src_filepath):
        raise FileNotFoundError(f"{src_filepath} not found.")
    if not os.path.exists(tgt_filepath):
        raise FileNotFoundError(f"{tgt_filepath} not found.")

    with open(src_filepath, "r", encoding="utf-8") as src_file, open(
        tgt_filepath, "r", encoding="utf-8"
    ) as tgt_file:
        translations = [
            {
                "translation": {
                    src_language: src_sentence.strip(),
                    tgt_language: tgt_sentence.strip(),
                }
            }
            for src_sentence, tgt_sentence in zip(src_file, tgt_file)
            if src_sentence and tgt_sentence
        ]

        with open(os.path.join(data_dir, "translations.json"), "w") as f:
            json.dump(translations, f)


def load_dataset(
    data_dir: str,
    translation_task: str,
    split=None,
    max_train_size: int = None,
    max_validation_size: int = None,
    max_test_size: int = None,
    **kwargs,
):
    """
    Custom load dataset function replacing the HuggingFace load_dataset.
    data_dir: path to directory containing train, validation, and test files.
    translation_task: string of the form "src_language-tgt_language".
    split: optional string specifying which split to load.
    """

    splits = ["train", "validation", "test"] if not split else [split]
    data_files = {
        split: os.path.join(data_dir, split, f"translations.json") for split in splits
    }

    try:
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"{data_dir} not found.")
        dataset = datasets.load_dataset("json", data_files=data_files, split=split)

    except FileNotFoundError:
        try:
            dataset = datasets.load_dataset(
                data_dir, translation_task, split=split, **kwargs
            )
        except:
            dataset = datasets.load_from_disk(data_dir)

    if split is None:
        for split in splits:
            if split == "train" and max_train_size:
                dataset[split] = dataset[split].select(range(max_train_size))
            if split == "validation" and max_validation_size:
                dataset[split] = dataset[split].select(range(max_validation_size))
            if split == "test" and max_test_size:
                dataset[split] = dataset[split].select(range(max_test_size))
            dataset[split].data_dir = data_dir
    dataset.data_dir = data_dir

    return dataset


def compute_confidence(
    translation_model: HuggingFaceTranslationModel, src_sentence: str, tgt_sentence: str
):
    model = translation_model.model
    tokenizer = translation_model.tokenizer
    device = next(model.parameters()).device

    # Tokenize the source and target sentences
    inputs = tokenizer(src_sentence, return_tensors="pt", add_special_tokens=True).to(
        device
    )
    target_tokens = tokenizer(
        text_target=tgt_sentence, return_tensors="pt", add_special_tokens=True
    ).input_ids.to(device)

    # We want to calculate the probability of each target token given the previous ones, so we create a sequence of decoder_input_ids
    # that starts with the start token and then includes all but the last token of the target sentence
    decoder_input_ids = torch.full(
        (1, 1), tokenizer.pad_token_id, dtype=torch.long, device=device
    )

    model.eval()  # Put the model in evaluation mode
    with torch.no_grad():  # Turn off gradients to save memory and computations
        total_prob = 0.0
        for i in range(target_tokens.size(1)):
            # Pass the current sequence to the model
            outputs = model(
                input_ids=inputs["input_ids"], decoder_input_ids=decoder_input_ids
            )
            logits = outputs.logits
            probs = logits.softmax(dim=-1)
            # Select the probability of the true next token
            next_token_id = target_tokens[0, i].unsqueeze(0)  # [1, 1]
            next_token_prob = probs[0, -1, :][next_token_id]
            total_prob += next_token_prob
            # Update the decoder_input_ids to include the true next token
            decoder_input_ids = torch.cat(
                [decoder_input_ids, next_token_id.unsqueeze(0)], dim=1
            )

        # Calculate the average probability
        avg_prob = total_prob / (target_tokens.size(1))

    return float(avg_prob)

def compute_probability(
    translation_model: HuggingFaceTranslationModel, src_sentence: str, tgt_sentence: str
):
    """
    Computes the probability of the model predicting the source sequence given the target sequence.
    """
    model = translation_model.model
    tokenizer = translation_model.tokenizer
    device = next(model.parameters()).device

    # Tokenize the source and target sentences
    inputs = tokenizer(src_sentence, return_tensors="pt", add_special_tokens=True).to(
        device
    )
    target_tokens = tokenizer(
        text_target=tgt_sentence, return_tensors="pt", add_special_tokens=True
    ).input_ids.to(device)

    # We want to calculate the probability of each target token given the previous ones, so we create a sequence of decoder_input_ids
    # that starts with the start token and then includes all but the last token of the target sentence
    decoder_input_ids = torch.full(
        (1, 1), tokenizer.pad_token_id, dtype=torch.long, device=device
    )

    model.eval()  # Put the model in evaluation mode
    with torch.no_grad():  # Turn off gradients to save memory and computations
        neg_log_likelihood = 0.0
        for i in range(target_tokens.size(1)):
            # Pass the current sequence to the model
            outputs = model(
                input_ids=inputs["input_ids"], decoder_input_ids=decoder_input_ids
            )
            logits = outputs.logits
            probs = logits.softmax(dim=-1)
            # Select the probability of the true next token
            next_token_id = target_tokens[0, i].unsqueeze(0)  # [1, 1]
            next_token_prob = probs[0, -1, :][next_token_id]
            neg_log_likelihood += -np.log2(next_token_prob)
            # Update the decoder_input_ids to include the true next token
            decoder_input_ids = torch.cat(
                [decoder_input_ids, next_token_id.unsqueeze(0)], dim=1
            )

    return np.exp(-neg_log_likelihood)


def compute_scaled_probability(
    translation_model: HuggingFaceTranslationModel, src_sentence: str, tgt_sentence: str
):
    """
    Computes the probability of the model predicting the source sequence given the target sequence, scaled
    by the number of tokens in the target sequence.
    """
    model = translation_model.model
    tokenizer = translation_model.tokenizer
    device = next(model.parameters()).device

    # Tokenize the source and target sentences
    inputs = tokenizer(src_sentence, return_tensors="pt", add_special_tokens=True).to(
        device
    )
    target_tokens = tokenizer(
        text_target=tgt_sentence, return_tensors="pt", add_special_tokens=True
    ).input_ids.to(device)

    # We want to calculate the probability of each target token given the previous ones, so we create a sequence of decoder_input_ids
    # that starts with the start token and then includes all but the last token of the target sentence
    decoder_input_ids = torch.full(
        (1, 1), tokenizer.pad_token_id, dtype=torch.long, device=device
    )

    model.eval()  # Put the model in evaluation mode
    with torch.no_grad():  # Turn off gradients to save memory and computations
        neg_log_likelihood = 0.0
        for i in range(target_tokens.size(1)):
            # Pass the current sequence to the model
            outputs = model(
                input_ids=inputs["input_ids"], decoder_input_ids=decoder_input_ids
            )
            logits = outputs.logits
            probs = logits.softmax(dim=-1)
            # Select the probability of the true next token
            next_token_id = target_tokens[0, i].unsqueeze(0)  # [1, 1]
            next_token_prob = probs[0, -1, :][next_token_id]
            neg_log_likelihood += -np.log2(next_token_prob)
            # Update the decoder_input_ids to include the true next token
            decoder_input_ids = torch.cat(
                [decoder_input_ids, next_token_id.unsqueeze(0)], dim=1
            )

        # Calculate the average probability
        mean_log_likelihood = neg_log_likelihood / (target_tokens.size(1))

    return np.exp(-mean_log_likelihood)


def compute_all_confidence(
    model_path: str,
    dataset: Dataset,
    src_language: str,
    tgt_language: str,
    max_num=500,
    max_num_epochs=16,
    include_base=False,
    confidence_func=compute_scaled_probability,
    shuffle=True,
) -> DefaultDict[int, float]:
    """Computes a list of confidence scores for each checkpoint in a model directory."""
    all_confidences = defaultdict(list)

    checkpoints = list(
        sorted(
            int(checkpoint.split("checkpoint-")[1])
            for checkpoint in os.listdir(model_path)
            if checkpoint.startswith("checkpoint-")
        )
    )
    if shuffle:
        dataset = dataset.shuffle()
        print("Dataset shuffled...")
    print(f"Found checkpoints: {checkpoints}")
    if max_num_epochs:
        checkpoints = checkpoints[:max_num_epochs]
    if include_base:
        checkpoints.append(None)
    for checkpoint in checkpoints:
        model = HuggingFaceTranslationModel.from_pretrained(
            model_path, checkpoint=checkpoint
        )
        # sample get max_num from dataset
        translations = dataset.select(range(max_num))["translation"]
        print(f"Checkpoint {checkpoint} loaded...", flush=True)
        for index, row in enumerate(tqdm(translations)):
            all_confidences[index].append(
                confidence_func(model, row[src_language], row[tgt_language])
            )

    return all_confidences


def get_detailed_df(
    model_path: str,
    dataset: Dataset,
    src_language: str,
    tgt_language: str,
    max_num=500,
    max_num_epochs=16,
    include_base=False,
    shuffle=True,
    confidence_func=compute_scaled_probability,
):
    """
    Returns a dataframe of all confidence scores for each checkpoint in a model directory, 
    for each sentence in a dataset. Clips the number of sentences to max_num.
    """
    # sample random selection from dataset
    result_dict = {"epoch": [], "src": [], "tgt": [], "confidence": []}

    checkpoints = list(
        sorted(
            int(checkpoint.split("checkpoint-")[1])
            for checkpoint in os.listdir(model_path)
            if checkpoint.startswith("checkpoint-")
        )
    )
    if shuffle:
        dataset = dataset.shuffle()
        print("Dataset shuffled...")
    print(f"Found checkpoints: {checkpoints}")
    if max_num_epochs:
        checkpoints = checkpoints[:max_num_epochs]
    if include_base:
        checkpoints.append(None)
    for checkpoint in checkpoints:
        model = HuggingFaceTranslationModel.from_pretrained(
            model_path, checkpoint=checkpoint
        )
        translations = dataset.select(range(max_num))["translation"]
        print(f"Checkpoint {checkpoint} loaded...", flush=True)
        for translation in tqdm(translations):
            # add row corresponding to confidence of translation
            src_sentence = translation[src_language]
            tgt_sentence = translation[tgt_language]
            confidence = confidence_func(model, src_sentence, tgt_sentence)
            result_dict["epoch"].append(checkpoint)
            result_dict["src"].append(src_sentence)
            result_dict["tgt"].append(tgt_sentence)
            result_dict["confidence"].append(confidence)
    return pd.DataFrame(result_dict)


def get_confidence_dataframe(
    model_path: str,
    dataset: Dataset, 
    src_language: str,
    tgt_language: str,
    save_to_disk = True,
    **kwargs,
):
    """Computes a list of confidence scores for each checkpoint in a model directory."""
    # sample random selection from dataset
    confidence_dict = compute_all_confidence(
        model_path, dataset, src_language, tgt_language, **kwargs
    )
    df_dict = {"src": [], "tgt": [], "confidence": [], "variability": []}
    for index, confidences in confidence_dict.items():
        src_sentence = dataset[index]["translation"][src_language]
        tgt_sentence = dataset[index]["translation"][tgt_language]
        confidences = confidence_dict[index]
        confidence = np.mean(confidences)
        stdev = np.std(confidences)
        df_dict["src"].append(src_sentence)
        df_dict["tgt"].append(tgt_sentence)
        df_dict["confidence"].append(confidence)
        df_dict["variability"].append(stdev)
    
    df = pd.DataFrame(df_dict)
    if save_to_disk:
        confidence_func = kwargs["confidence_func"].__name__ if "confidence_func" in kwargs else "compute_confidence"
        df_dir = os.path.join(model_path, f"dataframes_{confidence_func}")
        while os.path.exists(df_dir):
            df_dir = df_dir + "(copy)"
        if not os.path.isdir(df_dir):
            os.makedirs(df_dir)
        df.to_csv(os.path.join(df_dir, "confidence_df.csv"))
    return df


def plot_dataset(df: pd.DataFrame):
    # Plotting the points
    plt.figure(figsize=(10, 6))
    plt.scatter(df["variability"], df["confidence"], color="blue", label="data")

    # Adding title and labels
    plt.title("Confidence and Variability by Index")
    plt.xlabel("Variability")
    plt.ylabel("Confidence")
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)

    # Display the plot
    plt.show()
    plt.savefig("confidence.png")


def get_rttl(model1, model2, sentence, translation_config):
    # Gets the round trip translation of a sentence
    return model2.translate(model1.translate(sentence, translation_config), translation_config)

def rttl_chrf(model1, model2, sentence, translation_config):
    # Gets the round trip translation of a sentence
    rttl = model2.translate(model1.translate(sentence, translation_config), translation_config)
    return rttl, sacrebleu.sentence_chrf(rttl, [sentence])