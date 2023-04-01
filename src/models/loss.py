from collections import Iterable

import torch
import torch.nn as nn
from torch.nn import init

from utils import settings


class LossBinary(nn.Module):

    def make_linear(self, in_features, out_features, bias=True, std=0.02):
        linear = nn.Linear(in_features, out_features, bias)
        if std is not None:
            init.normal_(linear.weight, std=std)
            if bias:
                init.zeros_(linear.bias)
            return linear

    def make_ffnn(self, feat_size, output_size, hidden_size=None):
        if hidden_size is None:
            return nn.Sequential(self.make_linear(feat_size, output_size), nn.Sigmoid())

        if not isinstance(hidden_size, Iterable):
            hidden_size = [hidden_size]
        ffnn = [self.make_linear(feat_size, hidden_size[0]), nn.ReLU(), self.dropout]
        for i in range(1, len(hidden_size)):
            ffnn += [self.make_linear(hidden_size[i - 1], hidden_size[i]), nn.ReLU(), self.dropout]
        ffnn += [self.make_linear(hidden_size[-1], output_size), nn.Sigmoid()]
        return nn.Sequential(*ffnn)

    def __init__(self, config):
        super(LossBinary, self).__init__()
        self.loss = nn.BCELoss(reduction='none')
        self.dropout = nn.Dropout(config["dropout"])
        self.linear = self.make_ffnn(768 * 2, 1, [320])

    def forward(self, input, event_spans_index, predict=False):
        scores = self.linear(input)
        target = torch.zeros(scores.size(0)).float().to(device=settings.device)
        if event_spans_index.size(0) != 0:
            target[event_spans_index] = 1

        if predict:
            pred = torch.where(scores.squeeze(-1) > 0.5, 1, 0).tolist()
            return pred

        else:
            loss = self.loss(scores.squeeze(-1), target)

            weight = torch.where(target > 0, 1.0, 0.25)
            loss = torch.sum(loss * weight)

            return loss
