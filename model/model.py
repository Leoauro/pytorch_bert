from typing import Tuple

import torch
from torch import nn

from config import config


class BertModel(nn.Module):
    def __init__(self):
        super(BertModel, self).__init__()
        self.emb = Embedding()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.D_MODEL,
            nhead=config.N_HEAD,
            batch_first=True,
            dim_feedforward=1024,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.ENCODER_LAYERS,
                                             norm=nn.LayerNorm(config.D_MODEL))

        self.nsp_head = nn.Sequential(
            nn.Linear(config.D_MODEL, config.D_MODEL),
            nn.GELU(),
            nn.LayerNorm(config.D_MODEL),
            nn.Linear(config.D_MODEL, 2)  # 二分类
        )

    def forward(self, seq: torch.Tensor, mask: torch.Tensor, segment_ids: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """

        :param seq:
        :param mask: (batch_size,max_seq_len) bool 类型的 tensor
        :param segment_ids:
        :return:
        """
        ebd = self.emb(seq, segment_ids)
        out = self.encoder(ebd, src_key_padding_mask=mask)
        # 获取[CLS]位置的表示（用于NSP）
        cls_output = out[:, 0, :]  # [batch_size,d_model]
        nsp_logits = self.nsp_head(cls_output)
        return out, nsp_logits


class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(config.VOCAB_SIZE, config.D_MODEL)
        self.segment_ebd = nn.Embedding(2, config.D_MODEL)
        self.pos_ebd = nn.Embedding(config.SEQ_TOKEN_LEN, config.D_MODEL)
        self.pos_t = torch.arange(0, config.SEQ_TOKEN_LEN).reshape(1, config.SEQ_TOKEN_LEN)

        self.layer_norm = nn.LayerNorm(config.D_MODEL)
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, seq: torch.Tensor, segment: torch.Tensor) -> torch.Tensor:
        """"
            seq: (batch_size, max_seq_len)
            segment: (batch_size, max_seq_len)
        """
        ebd = self.embedding(seq)
        position = self.pos_t[:, :seq.shape[-1]].to(seq.device)
        position = self.pos_ebd(position)
        segment = self.segment_ebd(segment)
        # 层归一化和dropout
        out = self.layer_norm(ebd + position + segment)
        out = self.dropout(out)
        return out
