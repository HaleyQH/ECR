import argparse
import json
import logging
import os

import time
from pathlib import Path
import random

import numpy as np
import torch

from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AdamW

from utils.utils import create_datasets
from metrics.f1 import MetricF1
from utils import settings
from models import Model
from utils.utils import create_dictionaries, create_model

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger()


def do_evaluate(model, dataset, batch_size):
    collate_fn = model.collate_func()

    loader = DataLoader(dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=False)
    name = dataset.name

    model.eval()

    if hasattr(model, 'begin_epoch'):
        model.begin_epoch()

    pred = []
    for i, minibatch in enumerate(loader):
        # the config , how they are passed also a hack, TODO: make this cleaner
        predictions = model.predict(**minibatch)
        pred.append(predictions)
    metric = MetricF1()
    m = metric.update(pred)

    logging.info('%s-f1: %s -pr: %s -re: %s' % (name, m[0], m[1], m[2]))

    return pred, m


class Runner:
    def __init__(self, config):
        self.config = config
        self.bert_train_steps = config['lr-scheduler']['nr_iters_bert_training']

    def get_optimizer(self, model):
        no_decay = ['bias', 'LayerNorm.weight']
        bert_param, task_param = model.get_params(named=True)

        logger.debug('===bert_params in DECAY: ' +
                     str([n for n, p in bert_param if not any(nd in n for nd in no_decay)]))

        logger.debug('===bert_params in NO DECAY: ' +
                     str([n for n, p in bert_param if any(nd in n for nd in no_decay)]))

        logger.debug('=== task params: ' + str([n for n, p in task_param]))

        grouped_bert_param = [
            {
                'params': [p for n, p in bert_param if not any(nd in n for nd in no_decay)],
                'lr': self.config['lr-scheduler']['bert_learning_rate_start'],
                'weight_decay': self.config['optimizer']['adam_weight_decay']
            }, {
                'params': [p for n, p in bert_param if any(nd in n for nd in no_decay)],
                'lr': self.config['lr-scheduler']['bert_learning_rate_start'],
                'weight_decay': 0.0
            }
        ]
        optimizers = [
            AdamW(grouped_bert_param, lr=self.config['lr-scheduler']['bert_learning_rate_start'],
                  eps=self.config['optimizer']['adam_eps']),
            Adam(model.get_params()[1], lr=self.config['lr-scheduler']['task_learning_rate_start'],
                 eps=self.config['optimizer']['adam_eps'], weight_decay=0)
        ]
        return optimizers

    def get_scheduler_v2(self, optimizers, bert_start_step, bert_end_step, task_start_step, task_end_step,
                         min_lambda_bert, min_lambda_tasks):
        """
        The version _v2 intends to incorporate the nr of update steps for which bert and the tasks will be updated.
        """

        # Only warm up bert lr
        total_update_steps_bert = bert_end_step - bert_start_step
        total_update_steps_tasks = task_end_step - task_start_step
        warmup_steps = int(total_update_steps_bert * self.config['lr-scheduler']['bert_warmup_ratio'])

        ratio_increase_per_task_step = (min_lambda_tasks - 1.0) / total_update_steps_tasks
        ratio_increase_per_bert_step = (min_lambda_bert - 1.0) / (total_update_steps_bert - warmup_steps)

        def lr_lambda_bert(current_step):
            if current_step < bert_start_step:
                return 1.0  # no changes to learning rate

            if (current_step - bert_start_step) < warmup_steps and current_step >= bert_start_step:
                to_ret = float(current_step - bert_start_step + 1) / float(max(1, warmup_steps))
                return to_ret
            to_ret = max(min_lambda_bert,
                         ratio_increase_per_bert_step * (current_step - bert_start_step - warmup_steps + 1) + 1.0)
            return to_ret

        def lr_lambda_task(current_step):
            if current_step < task_start_step:
                return 1.0  # no changes to learning rate

            to_ret = max(min_lambda_tasks, ratio_increase_per_task_step * (current_step - task_start_step) + 1.0)
            return to_ret

        schedulers = [
            LambdaLR(optimizers[0], lr_lambda_bert),
            LambdaLR(optimizers[1], lr_lambda_task)
        ]
        return schedulers

    def train_spanbert(self, model: Model, datasets):
        logging.info('CURRENT settings.device VALUE IS: %s' % settings.device)
        logging.info('TRAINING, PLEASE WAIT...')
        conf = self.config
        epochs = conf['optimizer']['iters']

        grad_accum = conf['optimizer']['gradient_accumulation_steps']
        batch_size = conf['optimizer']['batch_size']

        device_name = settings.device
        device = torch.device(device_name)
        model = model.to(device)
        # model.load_model("/data/hzg2/ECR/experiments/hzg/save/stored_models/best.model")
        collate_fn = model.collate_func()
        train = DataLoader(datasets[conf['trainer']['train']], collate_fn=collate_fn, batch_size=batch_size,
                           shuffle=True)

        examples_train = datasets['train']

        optimizers = self.get_optimizer(model)

        bert_start_epoch = config['lr-scheduler']['bert_start_epoch']
        bert_end_epoch = config['lr-scheduler']['bert_end_epoch']
        task_start_epoch = config['lr-scheduler']['task_start_epoch']
        task_end_epoch = config['lr-scheduler']['task_end_epoch']

        bert_start_step = len(examples_train) * bert_start_epoch // grad_accum
        bert_end_step = len(examples_train) * bert_end_epoch // grad_accum
        task_start_step = len(examples_train) * task_start_epoch // grad_accum
        task_end_step = len(examples_train) * task_end_epoch // grad_accum

        bert_learning_rate_start = config['lr-scheduler']['bert_learning_rate_start']
        bert_learning_rate_end = config['lr-scheduler']['bert_learning_rate_end']

        task_learning_rate_start = config['lr-scheduler']['task_learning_rate_start']
        task_learning_rate_end = config['lr-scheduler']['task_learning_rate_end']

        min_lambda_bert = bert_learning_rate_end / bert_learning_rate_start
        min_lambda_task = task_learning_rate_end / task_learning_rate_start

        schedulers = self.get_scheduler_v2(optimizers, bert_start_step, bert_end_step, task_start_step, task_end_step,
                                           min_lambda_bert, min_lambda_task)

        bert_param, task_param = model.get_params()

        loss_during_accum = []  # To compute effective loss at each update
        loss_during_report = 0.0  # Effective loss during logging step
        loss_history = []  # Full history of effective loss; length equals total update steps


        model.zero_grad()
        optimizer_steps = 0
        f1_val = -1
        earlystop = 0
        for epo in range(1, 1 + epochs):

            settings.epoch = epo
            max_norm_bert = 0
            max_norm_tasks = 0

            logger.info('EPOCH: ' + str(epo))

            for i, minibatch in enumerate(train):
                model.train()

                loss = model.forward(**minibatch)
                if loss != None:
                    if grad_accum > 1:
                        loss /= grad_accum
                    loss.backward()

                    if 'clip-norm' in conf['optimizer']:
                        norm_bert = torch.nn.utils.clip_grad_norm_(bert_param, conf['optimizer']['clip-norm'])
                        norm_tasks = torch.nn.utils.clip_grad_norm_(task_param, conf['optimizer']['clip-norm'])

                        if norm_bert > max_norm_bert:
                            max_norm_bert = norm_bert

                        if norm_tasks > max_norm_tasks:
                            max_norm_tasks = norm_tasks

                    loss_during_accum.append(loss.item())

                    # Update
                    if len(loss_during_accum) % grad_accum == 0:
                        for optimizer in optimizers:
                            optimizer.step()
                        model.zero_grad()
                        for scheduler in schedulers:
                            scheduler.step()

                        # Compute effective loss
                        effective_loss = np.sum(loss_during_accum).item()
                        loss_during_accum = []
                        loss_during_report += effective_loss
                        loss_history.append(effective_loss)
                        optimizer_steps += 1

                        # Report
                        if optimizer_steps % conf['optimizer']['report_frequency'] == 0:
                            # Show avg loss during last report interval
                            avg_loss = loss_during_report
                            loss_during_report = 0.0
                            logger.info('\tStep %d: accumulated loss %.2f; for %d steps' %
                                        (optimizer_steps, avg_loss, conf['optimizer']['report_frequency']))

            with torch.no_grad():
                name = conf["trainer"]["valid"]
                if epo % conf['optimizer']['per'] == 0:
                    logger.info('valid every %s epoch' % conf['optimizer']['per'])
                    pred, m = do_evaluate(model, datasets[name], batch_size)
                    if m[0] > f1_val:
                        f1_val = m[0]
                        file_name = '{}.jsonl'.format(name)
                        base_dir = conf['path']
                        subdir = 'predictions/{}/'.format(name)
                        predict_file = os.path.join(base_dir, subdir, file_name)
                        dirname = os.path.dirname(predict_file)
                        os.makedirs(dirname, exist_ok=True)
                        with open(predict_file, 'w') as file:
                            for predictions in pred:
                                json.dump(predictions, file, ensure_ascii=False)
                                file.write('\n')
                        self.save_model()
                        earlystop = 0
                    else:
                        earlystop += 1
                        if earlystop > 5:
                            break

    def save_model(self):
        try:
            logger.info('writing best.model to %s' % config['path'])
            output_dir = os.path.join(config['path'], 'stored_models')
            os.makedirs(output_dir, exist_ok=True)
            model.write_model(os.path.join(output_dir, config['optimizer']['model']))
        except:
            logging.error('ERROR: failed to write model to disk')
            time.sleep(60)


def load_model(config, training=False, load_datasets_from_config=True):
    """
    creates the model from config, it can be either used to train it later, or to load the parameters of a saved model
    using state_dict() (https://pytorch.org/tutorials/beginner/saving_loading_models.html) .
    when using state_dict() to load saved model, the static embeddings are not loaded, only the trainable parameters are
    loaded, so it is necessary to use this method before the state_dict().

    :param config:
    :return:
    """
    dictionaries = create_dictionaries(config, training)
    datasets = None

    if load_datasets_from_config:
        datasets, data, evaluate = create_datasets(config, dictionaries)
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
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--device', dest='device', type=str, default='cuda')
    args = parser.parse_args()

    settings.device = args.device


    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        # 设置随机数种子


    setup_seed(20)

    with open(args.config_file) as f:
        config = json.load(f)

    if args.output_path is not None:
        logging.info('Setting path to %s' % args.output_path)
        config['path'] = args.output_path
        os.makedirs(config['path'], exist_ok=True)
    elif 'path' not in config:
        logging.info('set path=%s' % Path(args.config_file).parent)
        config['path'] = Path(args.config_file).parent

    config['dictionaries_path'] = os.path.join(config['path'], 'dictionaries')

    loaded_model_dict = load_model(config, True, load_datasets_from_config=True)
    dictionaries = loaded_model_dict['dictionaries']
    datasets = loaded_model_dict['datasets']
    model = loaded_model_dict['model']
    parameters = loaded_model_dict['parameters']

    if config['trainer']['version'] == 'bert':
        runner = Runner(config=config)
        runner.train_spanbert(model, datasets)
    else:
        raise RuntimeError('no implementation for the following trainer version: ' +
                           config['trainer']['version'])
