import os

from tokenizers import Tokenizer

from util import util


class CustomTokenizer:
    def __init__(self):
        project_path = util.project_path()
        tokenizer_file = os.path.join(project_path, "data", "tokenizer.json")
        self.tokenizer = Tokenizer.from_file(tokenizer_file)

    def tokenize(self, text: str) -> list[int]:
        encoded = self.tokenizer.encode(text)
        return encoded.ids

    def token_num(self) -> int:
        size = self.tokenizer.get_vocab_size()
        return size

    def detokenize(self, tokens: list[int]) -> str:
        decoded = ""
        for item in tokens:
            c = self.tokenizer.decode([item])
            decoded += c
        return decoded
