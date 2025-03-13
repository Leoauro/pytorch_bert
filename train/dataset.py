import os
from typing import Tuple

import datasets
import numpy as np
import torch
from torch import Tensor
from torch.utils import data

from config import config
from tokenizer.tokenizer import CustomTokenizer
from util import util


def _truncate_seq_pair(a, b, max_length):
    # 动态截断策略：优先截断较长的句子
    while True:
        total_length = len(a) + len(b)
        if total_length <= max_length:
            break
        if len(a) > len(b):
            a.pop()
        else:
            b.pop()


class PytorchBertDataset(data.Dataset):
    def __init__(self, csv_path: str) -> None:
        super().__init__()
        self.tokenizer = CustomTokenizer()
        self.cls_token = self.tokenizer.tokenize("[CLS]")
        self.sep_token = self.tokenizer.tokenize("[SEP]")
        self.mask_token = self.tokenizer.tokenize("[MASK]")
        self.pad_token = self.tokenizer.tokenize("[PAD]")
        self.dataset = datasets.Dataset.from_csv(csv_path)

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        data_item = self.dataset.__getitem__(index)
        sent_a, sent_b, is_next = data_item["sentence1"], data_item["sentence2"], data_item["label"]
        tokens_a = self.tokenizer.tokenize(sent_a)
        tokens_b = self.tokenizer.tokenize(sent_b)
        # 1. 拼接句子并添加特殊标记
        _truncate_seq_pair(tokens_a, tokens_b, config.SEQ_TOKEN_LEN - 3)
        input_tokens = self.cls_token + tokens_a + self.sep_token + tokens_b + self.sep_token
        segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)

        # 2. 生成MLM任务的Mask
        labels = input_tokens.copy()
        masked_positions = [0] * len(input_tokens)
        for i in range(1, len(input_tokens) - 1):  # 跳过[CLS]和[SEP]
            if np.random.rand() < config.MASK_PROP:
                masked_positions[i] = 1
                # BERT的80-10-10策略
                rand = np.random.rand()
                if rand < 0.8:
                    input_tokens[i] = self.mask_token[0]
                elif rand < 0.9:
                    input_tokens[i] = np.random.randint(5, config.VOCAB_SIZE)

        # 3. Padding处理
        padding = self.pad_token * (config.SEQ_TOKEN_LEN - len(input_tokens))
        attention_mask = [False] * len(input_tokens) + [True] * len(padding)
        input_tokens += padding
        labels += padding
        segment_ids += [0] * len(padding)

        masked_positions += [0] * len(padding)

        ret = (torch.tensor(input_tokens),
               torch.tensor(segment_ids),
               torch.tensor(attention_mask),
               torch.tensor(labels),
               torch.tensor(is_next),
               torch.tensor(masked_positions))

        return ret


if __name__ == '__main__':
    project_path = util.project_path()
    train_csv = os.path.join(project_path, "data", "train.csv")
    bert_data = PytorchBertDataset(train_csv)
    item = bert_data[0]
    print(item)
    pass
