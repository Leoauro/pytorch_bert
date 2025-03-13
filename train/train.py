import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW

from config import config
from dataset import PytorchBertDataset
from model.model import BertModel
from util import util

device = "cuda:0"
model = BertModel().cuda()
model.train()
# 获取数据集
project_path = util.project_path()
train_csv = os.path.join(project_path, "data", "train.csv")
dataset = PytorchBertDataset(train_csv)
# 装载到 DataLoader，
dataloader = DataLoader(dataset, config.BATCH_SIZE, shuffle=True)
# 损失函数 CrossEntropyLoss 已经隐式包含 softmax
mlm_criterion = nn.CrossEntropyLoss(ignore_index=3).cuda()
nsp_criterion = nn.CrossEntropyLoss()
# 优化器
optimizer = AdamW(params=model.parameters(), lr=config.LEARNING_RATE)

for epoch in range(config.EPOCHS):
    t = tqdm(dataloader)
    for token_ids, segment_ids, attention_mask, mlm_labels, labels, masked_positions in t:
        token_ids = token_ids.to(device)
        segment_ids = segment_ids.to(device)
        attention_mask = attention_mask.to(device)

        mlm_labels = mlm_labels.to(device)
        labels = labels.to(device)
        masked_positions = masked_positions.to(device)

        mlm_logits, nsp_logits = model(token_ids, attention_mask, segment_ids)
        mlm_logits = mlm_logits.view(-1, config.D_MODEL) @ model.emb.embedding.weight.T
        # 计算MLM损失
        # view重塑张量形状, 不改变数据在内存中的存储， reshape 可能会返回数据的副本
        mlm_loss = mlm_criterion(
            mlm_logits.view(-1, config.VOCAB_SIZE),
            mlm_labels.view(-1)
        )
        # masked_loss = (mlm_loss * masked_positions.view(-1).float()).sum()
        # avg_loss = masked_loss / masked_positions.sum()

        # 计算NSP损失
        nsp_loss = nsp_criterion(
            nsp_logits,
            labels
        )
        total_loss = mlm_loss + nsp_loss
        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
        optimizer.step()

        t.set_description(str(total_loss.item()))

torch.save(model.state_dict(), "model.pth")
