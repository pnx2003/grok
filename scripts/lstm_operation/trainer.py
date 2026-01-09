import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .dataset import OperationDataset
from .model import LSTMOperationModel

def train_lstm(operator, epochs=10, batch_size=32, lr=1e-3):
    """
    训练LSTM模型（对外暴露的唯一接口）
    :param operator: 运算类型（如"+", "s5", "x+y_mod_10"）
    :param epochs: 训练轮数
    :param batch_size: 批次大小
    :param lr: 学习率
    :return: 训练好的模型 + 词汇表
    """
    # 1. 初始化数据集
    dataset = OperationDataset(operator)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 2. 初始化模型
    model = LSTMOperationModel(vocab_size=dataset.vocab_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 3. 训练循环
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for input_seq, label in dataloader:
            optimizer.zero_grad()
            logits = model(input_seq)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")
    
    # 4. 返回模型和词汇表（用于推理）
    return model, dataset.vocab

def predict_lstm(model, vocab, a, op, b):
    """
    LSTM推理接口（对外暴露）
    :param model: 训练好的模型
    :param vocab: 数据集词汇表
    :param a: 操作数a（字符串）
    :param op: 操作符（字符串）
    :param b: 操作数b（字符串）
    :return: 预测的结果c（字符串）
    """
    # 构建反向词汇表
    idx2vocab = {v: k for k, v in vocab.items()}
    # 转换输入为索引序列
    input_seq = torch.tensor([
        vocab.get(a, 0),
        vocab.get(op, 0),
        vocab.get(b, 0)
    ], dtype=torch.long).unsqueeze(0)  # 增加batch维度
    
    # 推理
    model.eval()
    with torch.no_grad():
        logits = model(input_seq)
        pred_idx = torch.argmax(logits, dim=1).item()
    
    # 转换回字符串
    return idx2vocab.get(pred_idx, "unknown")