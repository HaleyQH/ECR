import argparse
import json
import logging
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.utils import create_datasets
from metrics.f1 import MetricF1
from utils import settings
from utils.utils import create_dictionaries, create_model

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger()

def predict(model, datasets, config):
    device = torch.device(settings.device)
    model = model.to(device)

    collate_fn = model.collate_func()
    batch_size = config['optimizer']['batch_size']

    model.load_model(config['model_path'])

    # evaluate

    logging.info('Start evaluating %s' % config['trainer']['evaluate'])

    loader = DataLoader(datasets[config['trainer']['evaluate']], collate_fn=collate_fn, batch_size=batch_size,
                        shuffle=False)

    model.eval()

    if hasattr(model, 'begin_epoch'):
        model.begin_epoch()

    output_path = os.path.join(config['output_path'], config['trainer']['evaluate'],
                               ('%s.jsonl' % config['trainer']['evaluate']))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pred = []
    with open(output_path, 'w') as file:
        for i, minibatch in enumerate(tqdm(loader)):
            predictions = model.predict(**minibatch)

            json.dump(predictions, file)
            file.write('\n')
            pred.append(predictions)
        metric = MetricF1()
        m = metric.update(pred)
        logging.info('-f1: %s -pr: %s -re: %s' % (m[0], m[1], m[2]))

    if hasattr(model, 'end_epoch'):
        model.end_epoch(config['trainer']['evaluate'])


def load_model(config, training=False, load_datasets_from_config=True):
    dictionaries = create_dictionaries(config, training)
    datasets = None

    if load_datasets_from_config:
        datasets, _, evaluate = create_datasets(config, dictionaries)
    model, parameters = create_model(config)

    to_ret = {
        'dictionaries': dictionaries,
        'datasets': datasets,
        'model': model,
        'parameters': parameters
    }
    return to_ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', help='configuration file')
    parser.add_argument('--model_path', help='path to the model to be evaluated', type=str, default=None,
                        required=True)
    parser.add_argument('--output_path', help='path to where the output predicted files will be saved',
                        type=str, default=None, required=True)
    parser.add_argument('--device', dest='device', type=str, default='cuda')
    args = parser.parse_args()

    settings.device = args.device

    with open(args.config_file) as f:
        config = json.load(f)

    config['model_path'] = args.model_path
    config['output_path'] = args.output_path
    config['dictionaries_path'] = os.path.join(os.path.dirname(config['model_path']), 'dictionaries')

    os.makedirs(config['output_path'], exist_ok=True)
    loaded_model_dict = load_model(config, training=False, load_datasets_from_config=True)
    dictionaries = loaded_model_dict['dictionaries']
    datasets = loaded_model_dict['datasets']
    model = loaded_model_dict['model']
    parameters = loaded_model_dict['parameters']

    predict(model=model, datasets=datasets, config=config)
