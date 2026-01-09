import torch
from torch.utils.data import Dataset
from .data_generator import OperationDataGenerator

class OperationDataset(Dataset):
    def __init__(self, operator, vocab=None):
        self.data = OperationDataGenerator._make_binary_operation_data(operator)
        self.vocab = vocab or self._build_vocab()
        self.vocab_size = len(self.vocab)
        self.seq_len = 3  # a + op + b 序列长度
    
    def _build_vocab(self):
        tokens = set()
        for item in self.data:
            tokens.add(item["a"])
            tokens.add(item["op"])
            tokens.add(item["b"])
            tokens.add(item["c"])
        tokens = sorted(list(tokens))
        return {tok: idx for idx, tok in enumerate(tokens)}
    
    def _token2idx(self, token):
        return self.vocab.get(token, 0)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        input_seq = torch.tensor([
            self._token2idx(item["a"]),
            self._token2idx(item["op"]),
            self._token2idx(item["b"])
        ], dtype=torch.long)
        label = torch.tensor(self._token2idx(item["c"]), dtype=torch.long)
        return input_seq, label