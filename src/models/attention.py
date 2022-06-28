from typing import Dict

import torch
from torch import nn

Tensor = torch.Tensor


class AttentionModel(nn.Module):
    def __init__(
        self,
        config: Dict,
    ) -> None:
        super().__init__()
        self.emb = nn.Embedding(config["vocab"], config["hidden_size"])
        self.lstm = nn.LSTM(
            input_size=config["hidden_size"],
            hidden_size=config["hidden_size"],
            dropout=config["dropout"],
            batch_first=True,
            num_layers=config["num_layers"],
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=config["hidden_size"],
            num_heads=4,
            dropout=config["dropout"],
            batch_first=True,
        )        
        self.linear = nn.Linear(config["hidden_size"], config["output_size"])

    def forward(self, x: Tensor) -> Tensor:
        x = self.emb(x)
        #x, _ = self.lstm(x)        
        x, _ = self.attention(x.clone(), x.clone(), x)
        last_step = x[:, -1, :]
        yhat = self.linear(last_step)
        return yhat