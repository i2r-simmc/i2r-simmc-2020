import json
import os
import wordninja
from string import punctuation

import ujson

import numpy as np
import torch
import yaml


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_config(config_path="config.yml"):
    with open(config_path, "r") as setting:
        config = yaml.load(setting, Loader=yaml.FullLoader)
    return config


def use_cuda(var):
    if torch.cuda.is_available():
        return var.cuda()
    else:
        return var


def save_model(the_model, path):
    if os.path.exists(path):
        path = path + '_copy'
    print("saving model to ...", path)
    torch.save(the_model, path)


def load_model(path):
    if not os.path.exists(path):
        assert False, 'cannot find model: ' + path
    print("loading model from ...", path)
    return torch.load(path)


def load_dict(filename):
    word2id = dict()
    with open(filename, encoding='utf-8') as f_in:
        for line in f_in:
            word = line.strip()
            word2id[word] = len(word2id)
    return word2id


def cal_accuracy(pred, answer_dist):
    """
    pred: batch_size
    answer_dist: batch_size, max_local_entity
    """
    num_hit_at_one = 0.0
    num_precision = 0.0
    num_recall = 0.0
    num_f1 = 0.0
    num_answerable = 0.0
    for i, l in enumerate(pred):
        num_hit_at_one += (answer_dist[i, l[0]] != 0)
        precision = 0
        for e in l:
            precision += (answer_dist[i, e] != 0)
        precision = float(precision) / len(l)
        recall = 0
        for j, answer in enumerate(answer_dist[i]):
            if answer != 0 and j in l:
                recall += 1
        recall = float(recall) / np.sum(answer_dist[i]) if np.sum(answer_dist[i]) > 0 else 0
        f1 = 0
        if precision + recall > 0:
            f1 = 2 * recall * precision / (precision + recall)
        num_precision += precision
        num_recall += recall
        num_f1 += f1
    for i, dist in enumerate(answer_dist):
        if np.sum(dist) != 0:
            num_answerable += 1
    return num_hit_at_one / len(pred), num_precision / len(pred), num_recall / len(pred), num_f1 / len(
        pred), num_answerable / len(pred)


def cal_accuracy_seq2seq(pred, answer_dist):
    """
    pred: batch_size
    answer_dist: batch_size, max_local_entity
    """
    num_hit_at_one = 0.0
    num_precision = 0.0
    num_recall = 0.0
    num_f1 = 0.0
    num_answerable = 0.0
    for i, l in enumerate(pred):
        num_hit_at_one += np.sum(l == answer_dist[i]) // l.shape[0]
    for i, dist in enumerate(answer_dist):
        if np.sum(dist) != 0:
            num_answerable += 1
    return num_hit_at_one / len(pred), num_precision / len(pred), num_recall / len(pred), num_f1 / len(
        pred), num_answerable / len(pred)


def save_json(list_, name):
    with open(name, 'w') as f:
        json.dump(list_, f, sort_keys=True, indent=4)


def load_json(file):
    if not file:
        return None
    with open(file, 'r') as f:
        data = ujson.load(f)
    return data


def load_fact(file):
    kb = dict()
    with open(file) as f:
        for line in f.readlines():
            line = line.strip()
            triple = line.split('|')
            triple = list(map(str.strip, triple))
            s, p, o = triple
            if s not in kb:
                kb[s] = dict()
            if p not in kb[s]:
                kb[s][p] = dict()
            if o not in kb[s][p]:
                kb[s][p][o] = 0
            if o not in kb:
                kb[o] = dict()
            if p not in kb[o]:
                kb[o][p] = dict()
            if s not in kb[o][p]:
                kb[o][p][s] = 1
    return kb


def print_model_parameters(model):
    for p in model.parameters():
        if p.requires_grad:
            print(p.name, p.numel())


def clean_text(text, filter_dot=False):
    text = text.replace('.', ' . ').lower()
    for punc in punctuation:
        if punc != '.':
            text = text.replace(punc, " ")
    text = text.split()
    output = []
    for i in text:
        # if len(i) < 10:
        #     output.append(i)
        # else:
        output.extend(wordninja.split(i))
    if filter_dot:
        return [e for e in text if e != '.']
    return text


def words2ids(str_in, vocab):
    words = str_in.split()
    sentence = np.ndarray(len(words) + 2, dtype=np.int32)
    sentence[0] = vocab['<sos>']
    for i, w in enumerate(words):
        if w in vocab:
            sentence[i + 1] = vocab[w]
        else:
            sentence[i + 1] = vocab['<unk>']
    sentence[-1] = vocab['<eos>']
    return sentence
