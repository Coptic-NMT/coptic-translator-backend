import pandas as pd
from collections import defaultdict
import random
import re
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModel,
)
import torch


class DictionaryTranslator:
    def __init__(self):
        dictionary_data = pd.read_csv("datasets/raw_dictionary.csv")

        self.cop_en_dict = defaultdict(list)
        self.en_cop_dict = defaultdict(list)

        for _, row in dictionary_data.iterrows():
            eng = row["eng"]
            translation = re.sub(r"\([^)]*\)", "", eng)
            translation = re.sub(r"\([^)]$", "", translation)

            eng = translation.split(",")
            for e in eng:
                stripped_e = e.strip()
                self.cop_en_dict[row["coptic"]].append(stripped_e)
                self.en_cop_dict[stripped_e].append(row["coptic"])
        self.init_eng_cop()

    def _get_dictionary(self, src, tgt):
        if src == "cop" and tgt == "en":
            return self.cop_en_dict
        elif src == "en" and tgt == "cop":
            return self.en_cop_dict
        else:
            raise ValueError("Invalid source or target language")

    def translate(self, word, src="cop", tgt="en"):
        if src == "en":
            return self.find_closest_coptic_word(word)

        dictionary = self._get_dictionary(src, tgt)
        entry = dictionary[word]
        if len(entry) == 0:
            return "?"
        rand_idx = random.randint(0, len(entry) - 1)
        translation = entry[rand_idx]
        # remove parenthesised text
        return translation

    def translate_sentence(self, sentence, src="cop", tgt="en"):
        # remove punctuations
        sentence = (
            sentence.replace(".", "").replace(",", "").replace("?", "").replace("!", "")
        )
        words = sentence.split(" ")
        translation = [self.translate(word, src, tgt) for word in words]
        return " ".join(translation)
    
    def vectorize(self, phrase):
        corpus_tokens = self.tokenizer(
            phrase, padding=True, truncation=True, return_tensors="pt"
        )

        with torch.no_grad():
            corpus_outputs = self.model(**corpus_tokens)
            # print("Got model outputs...")
            return (
                torch.mean(corpus_outputs.last_hidden_state, dim=1)
                .softmax(dim=-1)
                .cpu()
                .detach()
                .numpy()
            )[0]

    def init_eng_cop(self):
        english_coptic_dict = {k: v[0] for k, v in self.en_cop_dict.items()}
        # Load pre-trained word embeddings
        self.model = AutoModel.from_pretrained("bert-base-cased")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        self.english_coptic_dict = english_coptic_dict
        self.dict_words = list(english_coptic_dict.keys())
        self.eng_embeddings = []
        for word in tqdm(self.dict_words):
            self.eng_embeddings.append(self.vectorize(word))
        self.eng_embeddings = torch.tensor(self.eng_embeddings)
    def find_closest_coptic_word(self, english_word):
        embedding = self.vectorize(english_word)
        similarity_scores = cosine_similarity(
            embedding.reshape(1, -1),
            self.eng_embeddings
        )

        most_similar_index = similarity_scores.argmax()
        most_similar_word = self.dict_words[most_similar_index]
        return self.english_coptic_dict[most_similar_word]
