import os

from tokenizers import Tokenizer, pre_tokenizers, models, trainers

from config import config
from util import util

if __name__ == '__main__':
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.UnicodeScripts()
    trainer = trainers.BpeTrainer(
        vocab_size=config.VOCAB_SIZE,  # 目标词表大小
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],  # 特殊Token
        min_frequency=2,  # 最小词频阈值
        show_progress=True  # 显示进度条
    )
    project_path = util.project_path()
    train_csv = os.path.join(project_path, "data", "train.csv")
    tokenizer.train(files=[train_csv], trainer=trainer)
    # 保存词表和相关配置
    save_dir = os.path.join(project_path, "data", "tokenizer.json")
    tokenizer.save(save_dir)
