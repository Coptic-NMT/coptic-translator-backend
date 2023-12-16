
import pandas as pd
import sacrebleu
from tqdm import tqdm
from architecture import EncoderDecoder, Generator
from utils import subsequent_mask

import torch
import torch.nn as nn
from torch import Tensor




def loss(x: float, crit: nn.Module) -> Tensor: 
    d = x + 3 * 1
    predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d]])
    return crit(predict.log(), torch.LongTensor([1])).data


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator: Generator, criterion: nn.Module):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x: Tensor, y: Tensor, norm: float):
        x = self.generator(x)
        sloss = (
            self.criterion(
                x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
            )
            / norm
        )
        return sloss.data * norm, sloss
    


def greedy_decode(model: EncoderDecoder, src: Tensor, src_mask: Tensor, max_len: int, start_symbol: int, end_symbol: int):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for _ in range(max_len - 1):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
        
        if next_word == end_symbol:
            break

    return ys

