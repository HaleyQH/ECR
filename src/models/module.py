from collections import Iterable

import torch
import torch.nn as nn
import torch.nn.init as init
from transformers import BertModel

from utils import settings

class TextEmbedder(nn.Module):
    def __init__(self, config):
        super(TextEmbedder, self).__init__()
        self.config = config
        self.dim_output = 0

        if 'bert' in config:
            self.bert_embedder = BertModel.from_pretrained(config['bert']).to(device=settings.device)
            self.dim_output = config["dim"]
            self.embedding = self.bert_embedder.get_input_embeddings()
            self.embedding.requires_grad = False

        else:
            raise RuntimeError("TextEmbedder for only bert")

    def forward(self, input_ids, attention_mask):
        return self.bert_embedder(input_ids, attention_mask)[-2][:, 1:-1, :], self.embedding(input_ids)[:, 1:-1, :]


class SpanExtractor(nn.Module):
    """
    The original idea is that, unlike SpanEndpoint, it accepts directly the masked spans.
    Also some extra stuff from https://github.com/lxucs/coref-hoi
    """

    def make_linear(self, in_features, out_features, bias=True, std=0.02):
        linear = nn.Linear(in_features, out_features, bias)
        init.normal_(linear.weight, std=std)
        if bias:
            init.zeros_(linear.bias)
        return linear

    def make_ffnn(self, feat_size, output_size, hidden_size=None):
        if hidden_size is None:
            return self.make_linear(feat_size, output_size)

        if not isinstance(hidden_size, Iterable):
            hidden_size = [hidden_size]
        ffnn = [self.make_linear(feat_size, hidden_size[0]), nn.ReLU(), self.dropout]
        for i in range(1, len(hidden_size)):
            ffnn += [self.make_linear(hidden_size[i - 1], hidden_size[i]), nn.ReLU(), self.dropout]
        ffnn.append(self.make_linear(hidden_size[-1], output_size))
        return nn.Sequential(*ffnn)

    def __init__(self, dim_input, config):
        super(SpanExtractor, self).__init__()
        self.span_embed = 'span_embed' in config
        self.dim_output = dim_input
        self.dim_input = dim_input
        self.dropout = nn.Dropout(p=config['dropout'])

        self.ff = self.make_ffnn(self.dim_output, self.dim_output)

    def forward(self, inputs, b, e):
        b_vec = inputs[b, :]
        e_vec = inputs[e, :]
        vec = b_vec + e_vec

        return vec


class Distance(nn.Module):
    def __init__(self, config):
        super(Distance, self).__init__()
        self.dist_embed = config['dist_embed']
        self.category = config['category']
        self.distance_embeddings = nn.Embedding(config['category'], self.dist_embed)
        self.init_embeddings_std = config['init_embeddings_std']
        if self.init_embeddings_std is not None:
            init.normal_(self.distance_embeddings.weight, std=self.init_embeddings_std)

    def distance_function(self, span_sentence_index, pn_sentence_index):
        span_dist = torch.abs(span_sentence_index - pn_sentence_index)
        return span_dist

    def forward(self, span_sentence_index, np_sentence_index):
        span_dist = self.distance_function(np_sentence_index, span_sentence_index)
        span_dist = torch.where(span_dist < self.category, span_dist, self.category - 1)
        return self.distance_embeddings(span_dist)
