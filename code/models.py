import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Полносвязная нейронная сеть.
    Используется для контроля числа параметров W.
    """

    def __init__(self, input_dim, hidden_dims, num_classes):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h

        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)


def count_parameters(model):
    """Общее число обучаемых параметров W."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
