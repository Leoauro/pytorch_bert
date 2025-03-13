import os

import torch

from config import config
from model.model import BertModel
from tokenizer.tokenizer import CustomTokenizer
from train.dataset import PytorchBertDataset
from util import util

device = "cuda:0"
if __name__ == "__main__":
    model = BertModel().cuda()
    model.load_state_dict(torch.load("./train/model.pth"))
    model.eval()
    project_path = util.project_path()
    train_csv = os.path.join(project_path, "data", "test.csv")
    bert_data = PytorchBertDataset(train_csv)
    tokenizer = CustomTokenizer()
    right_predict_count = 0
    total_predict_count = 0
    nsp_right_predict_count = 0
    nsp_total_predict_count = 0
    for token_ids, segment_ids, attention_mask, mlm_labels, is_next, masked_positions in bert_data:
        token_ids = token_ids.unsqueeze(0).to(device)
        segment_ids = segment_ids.unsqueeze(0).to(device)
        attention_mask = attention_mask.unsqueeze(0).to(device)
        nsp_total_predict_count += 1
        mlm_labels = mlm_labels.to(device)
        is_next = is_next.to(device)
        masked_positions = masked_positions.to(device)
        mlm_logits, nsp_logits = model(token_ids, attention_mask, segment_ids)
        mlm_logits = mlm_logits.view(-1, config.D_MODEL) @ model.emb.embedding.weight.T
        mlm_logits = mlm_logits.squeeze()
        index_list = torch.where(masked_positions == 1)
        for index in index_list[0]:
            total_predict_count += 1
            true_token = mlm_labels[index]
            t = mlm_logits[index]
            predict_token = torch.argmax(t)
            # print("掩盖第:{}词，该词的token为:{}，该词为:{},预测的token:{}，预测的词为:{},".format(
            #     index,
            #     true_token,
            #     tokenizer.detokenize([true_token.item()]),
            #     predict_token,
            #     tokenizer.detokenize([predict_token.item()]),
            # ))
            if predict_token == true_token:
                right_predict_count += 1
        nsp_logits = nsp_logits.squeeze()
        if torch.argmax(nsp_logits) == is_next:
            nsp_right_predict_count += 1
    percentage = right_predict_count / total_predict_count * 100
    print("共预测{}，正确{}，预测的正确率为:{:.2f}%".format(total_predict_count, right_predict_count, percentage))
    print("是否为下一个句子：共预测{}，正确{}，预测的正确率为:{:.2f}%".format(nsp_total_predict_count,
                                                                           nsp_right_predict_count,
                                                                           nsp_right_predict_count / nsp_total_predict_count * 100))
