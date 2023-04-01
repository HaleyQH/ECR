import logging
import torch
import torch.nn as nn
import numpy as np
from utils import settings
from models.module import TextEmbedder, Distance
from models.module import SpanExtractor

from models.loss import LossBinary

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger()


def collate_bert(batch):
    """

    :param model:
    :param batch:
    :param device:
    :param collate_api: if in True, means that the input comes from a client, possibly as a free text
    (i.e., no gold mentions, relations, concepts, spans, etc.). If in False (default), the input comes for training
    or evaluating using internal function located in traintool.train for instance.
    :return:
    """
    assert len(batch) == 1  # module for only batch size 1

    inputs = {
        'attention_mask': batch[0]['content']['attention_mask'],
        'input_ids': batch[0]['content']['input_ids'],
    }

    metadata = {
        'id': batch[0]["id"],
        'pn': batch[0]['pn'].unsqueeze(0),
        'eventspan_index': batch[0]['eventspan_index'],
        'allspan': batch[0]['allspan'],
        'sentence_map': batch[0]['sentence_map'],
        'signal': batch[0]['signal']
    }

    return {
        'inputs': inputs,
        'metadata': metadata
    }


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()

        self.embedder = TextEmbedder(config['text_embedder'])

        self.lstm = nn.LSTM(input_size=self.embedder.dim_output, hidden_size=self.embedder.dim_output, num_layers=1)
        self.c0 = torch.zeros(1, 1, self.embedder.dim_output).to(device=settings.device)
        self.h0 = torch.zeros(1, 1, self.embedder.dim_output).to(device=settings.device)

        self.span_extractor = SpanExtractor(self.embedder.dim_output,
                                            config['span-extractor'])
        self.distance = Distance(config['distance'])

        self.binary_task = LossBinary(config['loss'])

    def forward(self, inputs, metadata):
        pn = metadata['pn']
        sentence_map = metadata['sentence_map']

        self.event_spans_index = metadata['eventspan_index']
        all_spans = metadata['allspan']
        if all_spans.size(0) == 0:
            return None

        # MODEL MODULES

        for index in range(len(inputs["input_ids"])):
            if index == 0:
                embeddings, token = self.embedder(inputs['input_ids'][index],
                                                  inputs['attention_mask'][index])[0], self.embedder(
                    inputs['input_ids'][index],
                    inputs['attention_mask'][index])[1]
            else:
                embeddings, token = torch.cat(
                    (embeddings, self.embedder(inputs['input_ids'][index],
                                               inputs['attention_mask'][index])[0]), 1), torch.cat(
                    (token, self.embedder(inputs['input_ids'][index],
                                          inputs['attention_mask'][index])[1]), 1)
        # we work only with batch size 1 in this module
        assert embeddings.shape[0] == 1

        embeddings = embeddings.permute(1, 0, 2)
        token = token.permute(1, 0, 2).squeeze()

        hidden = self.lstm(embeddings, (self.h0, self.c0))[0].squeeze()

        self.all_span_begin = (all_spans[:, :1].T)[0]
        self.all_span_end = (all_spans[:, 1:2].T)[0]
        all_span_sentence_index = sentence_map[self.all_span_begin].unsqueeze(1)

        self.pn_begin = pn[0, :1].T
        pn_end = pn[0, 1:2].T

        span_vecs = self.span_extractor(hidden, self.all_span_begin, self.all_span_end)

        hidden = hidden + 0.1 * token
        pn_vecs = self.span_extractor(hidden, self.pn_begin, pn_end)

        pn_vecs = pn_vecs.repeat(span_vecs.size(0), 1)
        self.pn_begin = self.pn_begin.repeat(all_span_sentence_index.size(0))

        input = torch.cat((pn_vecs, span_vecs), 1)

        output = self.binary_task(
            input,
            self.event_spans_index,
            predict=not self.training,
        )

        return output

    def predict(self, inputs, metadata):

        pred = self.forward(inputs, metadata)

        pred_event_index = self.decode(pred, metadata)

        return {'id': metadata['id'], "pred": pred_event_index, "target": self.event_spans_index.cpu().numpy().tolist()}

    def decode(self, pred, metadata):
        signal = metadata['signal']
        pred = np.multiply(signal, pred).tolist()
        pred_ = torch.tensor(pred)
        pred_ = torch.where(pred_ == 0, 1000000, pred_).cuda()

        if sum(pred) == 0:
            return []
        if sum(pred) == 1:
            return [pred.index(1)]

        if sum(pred) > 1:

            dis = self.pn_begin - self.all_span_end
            dis = torch.where(dis < 0, 1000000, dis)

            dis = torch.mul(pred_, dis)

            index = [i for i, x in enumerate(dis) if x == min(dis)]
            if len(index) == 1:
                return index
            span = (self.all_span_end - self.all_span_begin).cpu().numpy().tolist()
            selected = [span[i] for i in index]
            return [selected.index(max(selected))]

    def collate_func(self):
        return lambda x: collate_bert(x)

    def get_params(self, named=False):
        bert_based_param, task_param = [], []
        for name, param in self.named_parameters():
            if name.startswith('embedder.bert_embedder'):
                to_add = (name, param) if named else param
                bert_based_param.append(to_add)
            else:
                to_add = (name, param) if named else param
                task_param.append(to_add)
        return bert_based_param, task_param

    def load_model(self, filename, to_cpu=False):
        logger.info('to_cpu IN LOAD_MODEL: %s' % to_cpu)
        if to_cpu:
            partial = torch.load(filename, map_location=torch.device('cpu'))
        else:
            partial = torch.load(filename, map_location=torch.device(settings.device))

        state = self.state_dict()
        state.update(partial)
        self.load_state_dict(state)

    def write_model(self, filename):
        logger.info('write model: %s ' % filename)
        mydict = {}
        for k, v in self.state_dict().items():
            mydict[k] = v
        torch.save(mydict, filename)
