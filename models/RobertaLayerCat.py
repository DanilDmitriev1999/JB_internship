import torch
from transformers import AutoModel
from torch import tensor
import torch.nn as nn


class RobertaLayerCat(nn.Module):
    def __init__(self,
                 transformer_name: str,
                 layer_cat: list,
                 out_dim: int = 1,
                 dropout: float = 0.1) -> None:
        super().__init__()

        self.layers = layer_cat

        self.transformer = AutoModel.from_pretrained(transformer_name, output_hidden_states=True)
        self.drop = nn.Dropout(p=dropout)
        self.linear = nn.Linear(768*len(layer_cat), out_dim)

    def forward(self, input_ids: tensor, attention_mask: tensor) -> tensor:
        batch_size = input_ids.size(0)
        transformer_output = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask).hidden_states
        pooled_output = torch.cat(tuple([transformer_output[i] for i in self.layers]), dim=-1)
        pooled_output = pooled_output[:, 0, :]
        output = self.drop(pooled_output)
        output = self.linear(output)

        return output.view(batch_size)