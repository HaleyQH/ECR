import json
import logging
import os
import numpy
from tkinter import _flatten

from utils import settings
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger()

class DatasetRead(Dataset):
    def __init__(self, name, config, dictionaries):
        super().__init__()
        self.name = name
        self.file_name = []
        self.instance = []
        self.bert_tokenizer = dictionaries['bert_tokens']

        path = config['dataset']['filename']

        if os.path.isdir(path):
            for filename in tqdm(os.listdir(path)):
                f = os.path.join(path, filename)
                self.load_json(f)

        else:
            logging.error('NO MORE TERIES LEFT, FAILING')
            raise BaseException('Need a directory for the data')

    def convert(self, data):
        content = dict()
        content["input_ids"] = []
        content["attention_mask"] = []
        doc = data['doc'].split('\n')
        sentence_map = []
        for index, d in enumerate(doc):
            tokenize = self.bert_tokenizer(d, return_tensors='pt')
            sentence_map += [index] * ((tokenize["input_ids"]).size(1) - 2)
            content["input_ids"].append(tokenize["input_ids"].to(device=settings.device))
            content["attention_mask"].append(tokenize["attention_mask"].to(device=settings.device))
        pn = data['eventpn']

        allspan = data['allspan']

        try:
            event_index = [allspan.index(data['eventspan'])]
        except:
            event_index = []

        try:
            signal = data['flag']
        except:
            signal = []


        allspan = torch.tensor(allspan).to(device=settings.device)

        event_index = torch.tensor(event_index).to(device=settings.device)

        pn = torch.tensor(pn).to(device=settings.device)

        to_ret = {
            'id': data['id'],
            'doc': doc,
            'pn': pn,
            'allspan': allspan,
            'eventspan_index':event_index,
            'sentence_map': torch.tensor(list(_flatten(sentence_map))).to(device=settings.device),
            'content': content,
            'signal': signal
        }

        return to_ret

    def load_json(self, filename):
        for line in open(filename, 'r'):
            self.instance.append(self.convert(json.loads(line)))

    def __getitem__(self, index):
        return self.instance[index]

    def __len__(self):
        return len(self.instance)
