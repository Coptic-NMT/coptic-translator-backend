import json
import os
from attr import dataclass
import numpy as np
import torch
from torch.nn.functional import softmax
import evaluate
import transformers
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    MarianConfig,
    AutoTokenizer,
    EarlyStoppingCallback,
)
from datasets import Dataset
from base_model import BaseTranslationModel, GenerationConfig
# from utils import get_git_revision_short_hash


@dataclass
class HuggingFaceTranslationModelTrainingConfig:
    job_hash: str
    evaluation_strategy: str = "epoch"
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 32
    weight_decay: float = 0.01
    save_total_limit: int = 8
    num_train_epochs: int = 1
    label_smoothing_factor: float = 0
    predict_with_generate: bool = True
    eval_steps: int = 16000
    logging_steps: int = 500
    # commit_hash: str = get_git_revision_short_hash()

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.__dict__, f)


class HuggingFaceTranslationModel(BaseTranslationModel):
    def __init__(
        self,
        model_name: str,
        src_language: str,
        tgt_language: str,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
        save_to_disk=True,
        max_input_length: int = 128,
        max_target_length: int = 128,
    ):
        super().__init__(model_name, src_language, tgt_language, model, save_to_disk)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        if save_to_disk:
            self.save_args()
            tokenizer.save_pretrained(self.dir_path)
            model.save_pretrained(self.dir_path)

    def translate(self, src_sentence: str, config: GenerationConfig, output_confidence=False):
        inputs = self.tokenizer.encode(src_sentence, return_tensors="pt")
        output_scores, return_dict_in_generate = output_confidence, output_confidence
        outputs = self.model.generate(
            inputs[:, : self.tokenizer.model_max_length],
            max_length=config.max_length,
            max_new_tokens=config.max_new_tokens,
            min_length=config.min_length,
            min_new_tokens=config.min_new_tokens,
            early_stopping=config.early_stopping,
            do_sample=config.do_sample,
            num_beams=config.num_beams,
            num_beam_groups=config.num_beam_groups,
            top_k=config.top_k,
            top_p=config.top_p,
            temperature=config.temperature,
            diversity_penalty=config.diversity_penalty,
            output_scores=output_scores,
            return_dict_in_generate=True
        )
        translated_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

        if output_scores:
            scores = outputs.scores
            confidences = [softmax(score, dim=-1).max().item() for score in scores]
            num_words = len(translated_text.split())
            # scale the predicition probability by the number of words in the sentence
            scaled_probability = np.exp(sum(np.log(confidences)) / num_words)
            return translated_text, scaled_probability
            
        return translated_text

    def preprocess_function(self, data):
        inputs = [ex[self.src_language] for ex in data["translation"]]
        targets = [ex[self.tgt_language] for ex in data["translation"]]
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.max_input_length,
            truncation=True,
            padding="max_length",
        )

        # Setup the tokenizer for targets
        labels = self.tokenizer(
            text_target=targets,
            max_length=self.max_target_length,
            truncation=True,
            padding="max_length",
        )

        model_inputs["labels"] = labels["input_ids"]

        # Add decoder input ids
        model_inputs["decoder_input_ids"] = labels["input_ids"]

        return model_inputs

    def postprocess_text(self, preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    def train(
        self, dataset: Dataset, train_config: HuggingFaceTranslationModelTrainingConfig
    ):
        tokenized_datasets = dataset.map(self.preprocess_function, batched=True, load_from_cache_file=False)
        args = Seq2SeqTrainingArguments(
            self.dir_path,
            evaluation_strategy=train_config.evaluation_strategy,
            learning_rate=train_config.learning_rate,
            per_device_train_batch_size=train_config.per_device_train_batch_size,
            per_device_eval_batch_size=train_config.per_device_eval_batch_size,
            weight_decay=train_config.weight_decay,
            save_total_limit=train_config.save_total_limit,
            num_train_epochs=train_config.num_train_epochs,
            label_smoothing_factor=train_config.label_smoothing_factor,
            predict_with_generate=train_config.predict_with_generate,
            eval_steps=train_config.eval_steps,
            logging_steps=train_config.logging_steps,
            save_strategy="epoch" if self.save_to_disk else "no",
        )


        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer, model=self.model, return_tensors="pt"
        )

        def compute_metrics(eval_preds):
            metric = evaluate.load("sacrebleu")
            preds, labels = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]
            decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
            decoded_labels = self.tokenizer.batch_decode(
                labels, skip_special_tokens=True
            )
            decoded_preds, decoded_labels = self.postprocess_text(
                decoded_preds, decoded_labels
            )
            result = metric.compute(
                predictions=decoded_preds, references=decoded_labels
            )
            print("Decoded labels:", decoded_labels[:10])
            print("")
            print("Using preds: ", decoded_preds[:10])
            result = {"bleu": result["score"]}
            prediction_lens = [
                np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds
            ]
            result["gen_len"] = np.mean(prediction_lens)
            result = {k: round(v, 4) for k, v in result.items()}
            return result

        trainer = Seq2SeqTrainer(
            self.model,
            args=args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,
        )
        print(f"Beginning training model {self.model_name}...")
        trainer.train()
        if self.save_to_disk:
            train_config.save(os.path.join(self.dir_path, "train_config.json"))
            trainer.save_model(self.dir_path)
        self.model.to(torch.device("cpu"))

    def save_args(self):
        filename = os.path.join(self.dir_path, "args.json")
        if os.path.exists(filename):
            return
        with open(filename, "w") as f:
            info = {
                "model_name": self.model_name,
                "src_language": self.src_language,
                "tgt_language": self.tgt_language,
                "max_input_length": self.max_input_length,
                "max_target_length": self.max_target_length,
            }
            json.dump(info, f)

    @classmethod
    def from_pretrained(cls, path, checkpoint=None):
        args = json.load(open(os.path.join(path, "args.json")))

        if checkpoint:
            path = os.path.join(path, f"checkpoint-{checkpoint}")

        tokenizer = AutoTokenizer.from_pretrained(path)
        model: AutoModelForSeq2SeqLM = AutoModelForSeq2SeqLM.from_pretrained(path)

        return HuggingFaceTranslationModel(
            args["model_name"],
            args["src_language"],
            args["tgt_language"],
            tokenizer,
            model,
            save_to_disk=False,
            max_input_length=args["max_input_length"],
            max_target_length=args["max_target_length"],
        )
