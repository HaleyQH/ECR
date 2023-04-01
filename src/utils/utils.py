import logging

from transformers import BertTokenizer

from data_processing.data_reader import DatasetRead
from models import model_create

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger()



def load_dictionary(config, path):
    type = config['type']
    filename = config['filename']
    filename = filename if ('/' in filename) else '{}/{}'.format(path, filename)
    logger.debug('load_dictionary of config %s and path %s and filename %s' % (config, path, filename))
    if type == 'bert':
        dictionary = BertTokenizer.from_pretrained(config['filename'])
    else:
        raise BaseException('no such type', type)

    return dictionary


def create_dictionaries(config, training):
    path = config['dictionaries_path']
    dictionaries = {}

    for name, dict_config in config['dictionaries'].items():
        if training:
            if 'init' in dict_config:
                dictionary = load_dictionary(dict_config['init'], path)
            else:
                raise BaseException("no dictionary in config")

        else:
            if 'init' in dict_config:
                dictionary = load_dictionary(dict_config['init'], path)
            else:
                raise BaseException("no dictionary in config")

        dictionaries[name] = dictionary

    return dictionaries


def create_model(config):
    model = model_create(config['model'])

    logger.debug('Model: %s' % model)

    regularization = config['optimizer']['regularization'] if 'regularization' in config['optimizer'] else {}

    logger.debug('Parameters:')
    parameters = []
    num_params = 0
    for key, value in dict(model.named_parameters()).items():
        if not value.requires_grad:
            logger.debug('skip %s' % key)
            continue
        else:
            if key in regularization:
                logger.debug('param {} size={} l2={}'.format(key, value.numel(), regularization[key]))
                parameters += [{'params': value, 'weight_decay': regularization[key]}]
            else:
                logger.debug('param {} size={}'.format(key, value.numel()))
                parameters += [{'params': value}]
        num_params += value.numel()
    logger.debug('total number of params: {} = {}M'.format(num_params, num_params / 1024 / 1024 * 4))

    return model, parameters


def create_datasets(config, dictionaries):
    datasets = {name: DatasetRead(name, {'dataset': value,
                                         'model': config['model'],
                                         }, dictionaries)
                for name, value in config['datasets'].items()}

    if 'train' in config['trainer']:
        train = datasets[config['trainer']['train']]
        train.train = True
    else:
        train = None

    valid = config['trainer']['valid']

    return datasets, train, valid
