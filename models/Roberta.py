from transformers import AutoModel
from torch import tensor
import torch.nn as nn


class Roberta(nn.Module):
    def __init__(self,
                 transformer_name: str,
                 out_dim: int = 1,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.transformer = AutoModel.from_pretrained(transformer_name)
        self.drop = nn.Dropout(p=dropout)
        self.linear = nn.Linear(768, out_dim)

    def forward(self, input_ids: tensor, attention_mask: tensor) -> tensor:
        transformer_output = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask)[1]
        output = self.drop(transformer_output)
        output = self.linear(output).squeeze(1)

        return output


if __name__ == '__main__':
    pass
