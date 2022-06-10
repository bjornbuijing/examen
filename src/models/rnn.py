from typing import Dict

import torch
from torch import nn

Tensor = torch.Tensor


class NLPmodel(nn.Module):
    def __init__(
        self,
        config: Dict,
    ) -> None:
        super().__init__()
        self.emb = nn.Embedding(config["vocab"], config["hidden_size"])
        self.rnn = nn.GRU(
            input_size=config["hidden_size"],
            hidden_size=config["hidden_size"],
            dropout=config["dropout"],
            batch_first=True,
            num_layers=config["num_layers"],
        )
        self.linear = nn.Linear(config["hidden_size"], config["output_size"])

    def forward(self, x: Tensor) -> Tensor:
        x = self.emb(x)
        x, _ = self.rnn(x)
        last_step = x[:, -1, :]
        yhat = self.linear(last_step)
        return yhat
