import json
from typing import Union

from attr import dataclass
from architecture import *

import copy
import os
from base_model import BaseTranslationModel, GenerationConfig
from iterators import collate_batch_huggingface
from testing_utils import SimpleLossCompute, greedy_decode
from training_utils import (
    Batch,
    DummyOptimizer,
    DummyScheduler,
    LabelSmoothing,
    TrainState,
    rate,
    run_epoch,
)
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets import IterableDatasetDict, DatasetDict, Dataset
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast, AutoTokenizer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import GPUtil


@dataclass
class TranslationTrainingConfig:
    job_hash: str
    batch_size: int = 32
    distributed: bool = False
    num_epochs: int = 8
    accum_iter: int = 10
    base_lr: float = 1.0
    max_padding: int = 128
    warmup: int = 3000
    commit_hash: str = get_git_revision_short_hash()

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.__dict__, f)


class TranslationModel(BaseTranslationModel):
    """
    A neural machine translator between a source and target language. Upon initialization, the translator
    will either fetched cached model parameters between the two languages or it will translate using parallel
    train, validation, and test data located within the models/{src}-{target}/data directory.

    To initialize a new translation model on a language pair src-target, ensure that the models/{src}-{target}/data
    directory has files train.src, train.tgt, valid.src, valid.tgt, test.src, and test.tgt.

    PUBLIC API:
    translate(src_sentence): Translates a sentence from the source language to the target language.
    """

    def __init__(
        self,
        model_name: str,
        src_language: str,
        tgt_language: str,
        src_tokenizer: Optional[PreTrainedTokenizerBase] = None,
        tgt_tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model: EncoderDecoder = None,
        distributed=False,
        save_to_disk=True,
        N: int = 6,
        d_model: int = 512,
        d_ff: int = 2048,
        heads: int = 8,
        dropout: float = 0.1,
        model_max_len: int = 5000,
        **kwargs,
    ):
        super().__init__(model_name, src_language, tgt_language, model, save_to_disk)
        self.save_to_disk = save_to_disk
        self.N = N
        self.d_model = d_model
        self.d_ff = d_ff
        self.heads = heads
        self.dropout = dropout
        self.model_max_len = model_max_len
        self.src_tokenizer = (
            self._load_tokenizer(src_language) if not src_tokenizer else src_tokenizer
        )
        self.tgt_tokenizer = (
            self._load_tokenizer(tgt_language) if not tgt_tokenizer else tgt_tokenizer
        )
        self.add_special_tokens(self.src_tokenizer, self.tgt_tokenizer)

        self.src_vocab = self.src_tokenizer.get_vocab()
        self.tgt_vocab = self.tgt_tokenizer.get_vocab()
        print(f"Initialized {src_language} vocab with {len(self.src_vocab)} tokens.")
        print(f"Initialized {tgt_language} vocab with {len(self.tgt_vocab)} tokens.")

        if not self.model:
            self.model = self.__class__.make_model(
                len(self.src_vocab),
                len(self.tgt_vocab),
                N,
                d_model,
                d_ff,
                heads,
                dropout,
                model_max_len,
            )
        if self.save_to_disk:
            self.save_args()

    def translate(self, src_sentence: str, config=GenerationConfig()):
        if not src_sentence or src_sentence.isspace():
            return src_sentence

        self.model.eval()
        encoded = self.src_tokenizer(src_sentence, return_tensors="pt")
        src = encoded["input_ids"]
        src_mask = (src != self.src_tokenizer.pad_token_id).unsqueeze(-2)
        out = greedy_decode(
            self.model,
            src,
            src_mask,
            max_len=config.max_new_tokens,
            start_symbol=self.tgt_tokenizer.bos_token_id,
            end_symbol=self.tgt_tokenizer.eos_token_id,
        )
        return self.tgt_tokenizer.decode(out[0].tolist(), skip_special_tokens=True)

    @classmethod
    def make_model(
        cls,
        len_src_vocab: int,
        len_tgt_vocab: int,
        N: int,
        d_model: int,
        d_ff: int,
        heads: int,
        dropout: float,
        model_max_len: int,
    ):
        "Helper: Construct a model from hyperparameters."
        c = copy.deepcopy
        attn = MultiHeadedAttention(heads, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout, model_max_len)
        architecture = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
            nn.Sequential(Embeddings(d_model, len_src_vocab), c(position)),
            nn.Sequential(Embeddings(d_model, len_tgt_vocab), c(position)),
            Generator(d_model, len_tgt_vocab),
        )

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in architecture.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return architecture

    def train(self, dataset: Dataset, train_config: TranslationTrainingConfig):
        """
        Trains on a dataset with test and validation data.
        """
        print(f"Training translation model on {self.src_language}-{self.tgt_language}.")
        if train_config.distributed:
            self._train_distributed_model(train_config)
        else:
            self._train_worker(0, dataset, 1, train_config, False)
        self.model.to(torch.device("cpu"))

    def save(self, train_config: TranslationTrainingConfig):
        self.save_model_info()
        train_config.save(os.path.join(self.dir_path, "train_config.json"))
        torch.save(
            self.model.state_dict(),
            os.path.join(self.dir_path, "model_final.pt"),
        )

    def save_args(self):
        # save tokenizers
        src_tokenizer_path = os.path.join(self.dir_path, "src_tokenizer")
        tgt_tokenizer_path = os.path.join(self.dir_path, "tgt_tokenizer")
        self.src_tokenizer.save_pretrained(src_tokenizer_path)
        self.tgt_tokenizer.save_pretrained(tgt_tokenizer_path)
        # save model args
        args = {
            "model_name": self.model_name,
            "src_language": self.src_language,
            "tgt_language": self.tgt_language,
        }
        with open(os.path.join(self.dir_path, "args.json"), "w") as f:
            json.dump(args, f)

    def save_model_info(self):
        # Use for backwards compatibility
        model_info = {
            "len_src_vocab": len(self.src_vocab),
            "len_tgt_vocab": len(self.tgt_vocab),
            "N": self.N,
            "d_model": self.d_model,
            "d_ff": self.d_ff,
            "heads": self.heads,
            "dropout": self.dropout,
            "model_max_len": self.model_max_len,
        }
        with open(os.path.join(self.dir_path, "model_info.json"), "w") as f:
            json.dump(model_info, f)

    @classmethod
    def from_pretrained(cls, path: str, checkpoint=None):
        print(f"Loading model from path {path}.")
        args = json.load(open(os.path.join(path, "args.json"), "r"))
        src_tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(path, "src_tokenizer")
        )
        tgt_tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(path, "tgt_tokenizer")
        )
        model_info = json.load(open(os.path.join(path, "model_info.json"), "r"))
        model = cls.make_model(**model_info)
        pt_path = (
            os.path.join(path, "model_final.pt")
            if checkpoint is None
            else os.path.join(path, "checkpoints", checkpoint + ".pt")
        )
        model.load_state_dict(torch.load(pt_path))
        return TranslationModel(
            args["model_name"],
            args["src_language"],
            args["tgt_language"],
            src_tokenizer,
            tgt_tokenizer,
            model,
            save_to_disk=False,
        )

    def _load_tokenizer(self, language: str):
        tokenizer_path = os.path.join(self.dir_path, f"{language}_tokenizer.json")

        if not os.path.exists(tokenizer_path):
            self._train_tokenizer(language)

        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
        tokenizer.model_max_length = self.model_max_len
        return tokenizer

    def add_special_tokens(
        self,
        src_tokenizer: PreTrainedTokenizerBase,
        tgt_tokenizer: PreTrainedTokenizerBase,
    ):
        for tokenizer in [src_tokenizer, tgt_tokenizer]:
            if not tokenizer.bos_token_id:
                tokenizer.add_special_tokens({"bos_token": "[BOS]"})
            if not tokenizer.eos_token_id:
                tokenizer.add_special_tokens({"eos_token": "[EOS]"})
            if not tokenizer.pad_token_id:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def _train_tokenizer(self, language: str, vocab_size=10000):
        print(f"Training new BPE tokenizer for language {language}.")

        data_path = f"{self.dir_path}data/train.{language}"
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"File {data_path} does not exist; could not train tokenizer on language {language}."
            )

        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["[UNK]", "[EOS]", "[BOS]", "[PAD]", "[MASK]"],
        )

        # Train the tokenizer
        tokenizer.train(files=[data_path], trainer=trainer)
        tokenizer.save(f"{self.dir_path}{language}_tokenizer.json")

    def _create_dataloaders(
        self,
        dataset: Dataset,
        device,
        batch_size=12000,
        max_padding=128,
        is_distributed=True,
    ):
        def collate_fn(batch):
            return collate_batch_huggingface(
                batch,
                self.src_language,
                self.tgt_language,
                self.src_tokenizer,
                self.tgt_tokenizer,
                device,
                max_padding,
            )

        train_sampler = DistributedSampler(dataset["train"]) if is_distributed else None
        valid_sampler = (
            DistributedSampler(dataset["validation"]) if is_distributed else None
        )

        train_dataloader = DataLoader(
            dataset["train"],
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            collate_fn=collate_fn,
        )
        valid_dataloader = DataLoader(
            dataset["validation"],
            batch_size=batch_size,
            shuffle=(valid_sampler is None),
            sampler=valid_sampler,
            collate_fn=collate_fn,
        )
        return train_dataloader, valid_dataloader

    def _train_worker(
        self,
        gpu: int,
        dataset: Dataset,
        ngpus_per_node: int,
        train_config: TranslationTrainingConfig,
        is_distributed: bool,
    ):
        print(f"Train worker process using GPU: {gpu} for training", flush=True)
        torch.cuda.set_device(gpu)
        pad_idx = self.tgt_tokenizer.pad_token_id
        model = self.model
        model.cuda(gpu)
        module = model
        is_main_process = True
        if is_distributed:
            dist.init_process_group(
                "nccl", init_method="env://", rank=gpu, world_size=ngpus_per_node
            )
            model = DDP(model, device_ids=[gpu])
            module = model.module
            is_main_process = gpu == 0

        criterion = LabelSmoothing(
            size=len(self.tgt_vocab), padding_idx=pad_idx, smoothing=0.1
        )
        criterion.cuda(gpu)

        train_dataloader, valid_dataloader = self._create_dataloaders(
            dataset,
            gpu,
            batch_size=train_config.batch_size // ngpus_per_node,
            max_padding=train_config.max_padding,
            is_distributed=is_distributed,
        )

        optimizer = torch.optim.Adam(
            model.parameters(), lr=train_config.base_lr, betas=(0.9, 0.98), eps=1e-9
        )
        lr_scheduler = LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda step: rate(
                step, self.d_model, factor=1, warmup=train_config.warmup
            ),
        )
        train_state = TrainState()

        for epoch in range(train_config.num_epochs):
            if is_distributed:
                train_dataloader.sampler.set_epoch(epoch)
                valid_dataloader.sampler.set_epoch(epoch)

            model.train()
            print(f"[GPU{gpu}] Epoch {epoch} Training ====", flush=True)
            _, train_state = run_epoch(
                (Batch(b[0], b[1], pad_idx) for b in train_dataloader),
                model,
                SimpleLossCompute(module.generator, criterion),
                optimizer,
                lr_scheduler,
                mode="train+log",
                accum_iter=train_config.accum_iter,
                train_state=train_state,
            )

            GPUtil.showUtilization()
            if is_main_process and self.save_to_disk:
                checkpoint_dir = os.path.join(self.dir_path, "checkpoints/")
                if not os.path.exists(checkpoint_dir):
                    os.mkdir(checkpoint_dir)
                file_path = os.path.join(checkpoint_dir, f"{epoch}.pt")
                torch.save(module.state_dict(), file_path)
            torch.cuda.empty_cache()

            print(f"[GPU{gpu}] Epoch {epoch} Validation ====", flush=True)
            model.eval()
            sloss = run_epoch(
                (Batch(b[0], b[1], pad_idx) for b in valid_dataloader),
                model,
                SimpleLossCompute(module.generator, criterion),
                DummyOptimizer(),
                DummyScheduler(),
                mode="eval",
            )
            print(sloss)
            torch.cuda.empty_cache()

        if is_main_process and self.save_to_disk:
            self.save(train_config)

    def _train_distributed_model(self, train_config: TranslationTrainingConfig):
        ngpus = torch.cuda.device_count()
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12356"
        print(f"Number of GPUs detected: {ngpus}")
        print("Spawning training processes ...")
        mp.spawn(
            self._train_worker,
            nprocs=ngpus,
            args=(ngpus, train_config, True),
        )
