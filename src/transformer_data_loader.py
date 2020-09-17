import torch
from torch.utils.data import Dataset

from util import load_json


class SimmcDataset(Dataset):
    def __init__(self, data_file_src, data_file_tgt, tokenizer_enc, tokenizer_dec, is_train=True):
        self.src = []
        self.src_mask = []
        self.tgt = []

        print('loading data from', data_file_src)
        with open(data_file_src) as f_src:
            lines_src = f_src.readlines()
        with open(data_file_tgt) as f_tgt:
            lines_tgt = f_tgt.readlines()
        for idx in range(len(lines_src)):
            line_tgt = lines_tgt[idx]
            line_tgt = '<cls> %s <end>' % line_tgt
            src = tokenizer_enc(lines_src[idx], add_special_tokens=True)
            src_vec = src.input_ids
            src_mask = src.attention_mask
            tgt_vec = tokenizer_dec.encode(line_tgt, add_special_tokens=True)
            self.src.append(src_vec)
            self.src_mask.append(src_mask)
            self.tgt.append(tgt_vec)

    def __len__(self):
        return len(self.src)

    def __getitem__(self, item):
        return torch.tensor(self.src[item]), torch.tensor(self.src_mask[item]), torch.tensor(self.tgt[item])


class SimmcActDataset(Dataset):
    def __init__(self, data_file_src, data_file_tgt, tokenizer_enc, tokenizer_dec, is_train=True):
        self.src = []
        self.tgt = []
        self.src_mask = []

        print('loading data from', data_file_src)
        with open(data_file_src) as f_src:
            lines_src = f_src.readlines()
        lines_tgt = []
        tgt_dialogs = load_json(data_file_tgt)
        for d in tgt_dialogs['dialogs']:
            for turn in d['dialog']:
                target = turn['target']
                lines_tgt.append('<cls>' + target + ' <end>')
        for idx in range(len(lines_src)):
            src = tokenizer_enc(lines_src[idx], add_special_tokens=True)
            src_vec = src.input_ids
            src_mask = src.attention_mask
            tgt_vec = tokenizer_dec.encode(lines_tgt[idx], add_special_tokens=True)
            self.src.append(src_vec)
            self.src_mask.append(src_mask)
            self.tgt.append(tgt_vec)

    def __len__(self):
        return len(self.src)

    def __getitem__(self, item):
        return torch.tensor(self.src[item]), torch.tensor(self.src_mask[item]), torch.tensor(self.tgt[item])


class SimmcFusionDataset(Dataset):
    def __init__(self, data_file_src, data_file_tgt1, data_file_tgt2, data_file_tgt3, tokenizer_enc, tokenizer_dec, is_train=True):
        self.src = []
        self.tgt = []
        self.src_mask = []

        print('loading data from', data_file_src)
        with open(data_file_src) as f_src:
            lines_src = f_src.readlines()
        lines_tgt1 = []
        tgt_dialogs = load_json(data_file_tgt1)
        for d in tgt_dialogs['dialogs']:
            for turn in d['dialog']:
                target = turn['target']
                lines_tgt1.append(target)
        with open(data_file_tgt2) as f_tgt:
            lines_tgt2 = f_tgt.readlines()
        lines_tgt2 = [e.replace('<EOS>', '').strip()  for e in lines_tgt2]
        lines_tgt3 = []
        tgt_dialogs = load_json(data_file_tgt3)
        for d in tgt_dialogs['dialogs']:
            for turn in d['dialog']:
                target = turn['answer']
                lines_tgt3.append(target)
        lines_tgt = ['<cls> ' + lines_tgt1[idx] + ' <sep1> ' + lines_tgt2[idx] + ' <sep2> ' + lines_tgt3[idx] + ' <end>' for idx in range(len(lines_tgt1))]
        for idx in range(len(lines_src)):
            src = tokenizer_enc(lines_src[idx], add_special_tokens=True)
            src_vec = src.input_ids
            src_mask = src.attention_mask
            tgt_vec = tokenizer_dec.encode(lines_tgt[idx], add_special_tokens=True)
            self.src.append(src_vec)
            self.src_mask.append(src_mask)
            self.tgt.append(tgt_vec)

    def __len__(self):
        return len(self.src)

    def __getitem__(self, item):
        return torch.tensor(self.src[item]), torch.tensor(self.src_mask[item]), torch.tensor(self.tgt[item])
