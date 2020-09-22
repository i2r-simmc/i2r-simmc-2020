import argparse
import json
import os

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from transformers.modeling_encoder_decoder import EncoderDecoderModel

from model import BartLMHeadModel
from transformer_data_loader import SimmcFusionDataset
from util import get_config, load_json, save_json

START_BELIEF_STATE = '=> Belief State :'
END_OF_BELIEF = '<EOB>'


def _get_models(config):
    encoder_decoder_tuples = config['encoder_decoder_model_name_or_path'].split(',')
    encoder_decoder_tuples = tuple(encoder_decoder_tuples)
    enc_model = encoder_decoder_tuples[0]
    dec_model = encoder_decoder_tuples[0] if len(encoder_decoder_tuples) == 1 else encoder_decoder_tuples[1]
    share_model = 'share_model' in config and config['share_model']
    if 'bart' in enc_model:
        model = BartLMHeadModel.from_pretrained(enc_model, torchscript=True)
    else:
        model = EncoderDecoderModel.from_encoder_decoder_pretrained(enc_model, dec_model,
                                                                    share_model=share_model,
                                                                    torchscript=True,
                                                                    encoder_torchscript=True)
    return model


def _get_enc_dec_tokenizers(config, model):
    encoder_decoder_tuples = config['encoder_decoder_model_name_or_path'].split(',')
    enc_model = encoder_decoder_tuples[0]
    dec_model = encoder_decoder_tuples[0] if len(encoder_decoder_tuples) == 1 else encoder_decoder_tuples[1]
    cache_dir = config['cache_dir']
    use_same_tokenizer = 'bart' in enc_model or ('share_model' in config and config['share_model'])
    if use_same_tokenizer:
        if config['name'] in ('simmc', 'simmc-act', 'simmc-fusion'):
            tokenizer_dec = AutoTokenizer.from_pretrained(dec_model, cache_dir=cache_dir)
            tokenizer_dec.add_special_tokens({
                'additional_special_tokens': ['<sep>', '<cls>', '<end>', '<unk>', '<pad>', '<cst>', '<SOM>', '<EOM>']
            })
            if config['name'] == 'simmc-fusion':
                tokenizer_dec.add_special_tokens({
                    'additional_special_tokens': ['<sep1>', '<sep2>']
                })
        else:
            tokenizer_dec = None
        tokenizer_enc = tokenizer_dec
    else:
        if config['name'] in ('simmc', 'simmc-act', 'simmc-fusion'):
            tokenizer_enc = AutoTokenizer.from_pretrained(enc_model, cache_dir=cache_dir)
            tokenizer_enc.add_special_tokens({
                'additional_special_tokens': ['<SOM>', '<EOM>']
            })
            tokenizer_dec = AutoTokenizer.from_pretrained(dec_model, cache_dir=cache_dir)
            if os.path.exists(config['data_folder'] + config['decoder_special_token']):
                with open(config['data_folder'] + config['decoder_special_token'], "rb") as handle:
                    special_tokens_dict = json.load(handle)
                    num_added_toks = tokenizer_dec.add_special_tokens(special_tokens_dict)
            tokenizer_dec.add_special_tokens({
                'additional_special_tokens': ['<sep>', '<cls>', '<end>', '<unk>', '<pad>']
            })
            if config['name'] == 'simmc-fusion':
                tokenizer_dec.add_special_tokens({
                    'additional_special_tokens': ['<sep1>', '<sep2>']
                })
        else:
            tokenizer_enc = None
            tokenizer_dec = None
    tokenizer_dec.eos_token = '<end>'
    tokenizer_dec.cls_token = '<cls>'
    tokenizer_dec.unk_token = '<unk>'
    tokenizer_dec.pad_token = '<pad>'

    if 'bart' in enc_model:
        model.resize_token_embeddings(len(tokenizer_dec))
        model.vocab_size = len(tokenizer_dec)
    else:
        model.encoder.resize_token_embeddings(len(tokenizer_enc))
        model.decoder.resize_token_embeddings(len(tokenizer_dec))
        model.config.encoder.vocab_size = len(tokenizer_enc)
        model.config.decoder.vocab_size = len(tokenizer_dec)
        # if 'gpt' in dec_model:
        model.decoder.config.use_cache = False
    return tokenizer_enc, tokenizer_dec


def _clean_special_characters(lines, tokenizer_dec, remove_space=True):
    eos_token = tokenizer_dec.eos_token
    pad_token = tokenizer_dec.pad_token
    lines = [rp[:(rp.index(eos_token) if eos_token in rp else len(rp))] for rp in lines]
    lines = [
        rp.replace(r'<|endoftext|>', '').replace('!', '').replace('<cls>', '').replace('<CLS>', '').replace('[sep]',
                                                                                                            '').replace(
            '[SEP]', '').replace(
            '<end>', '').replace('<END>', '').replace('<eod>', '').replace('<EOD>', '') for rp in lines]
    if remove_space:
        lines = [rp.replace(' ', '') for rp in lines]
    lines = [rp.replace(pad_token, '') for rp in lines]
    lines = [rp.strip() for rp in lines]
    return lines


def _train_epoch(model, optimizer, scheduler, train_data_loader, dev_data_loader, config):
    train_iterator = range(0, int(config['num_epoch']))
    best_loss = 1000000000000000000
    t_total = len(train_data_loader)
    for epoch in train_iterator:
        model.train()
        train_loss = []
        for step, batch in enumerate(tqdm(train_data_loader, total=t_total, desc="Train Iteration",
                                          disable=config['local_rank'] not in [-1, 0])):
            src = batch[0].to(config['device'])
            src_mask = batch[1].to(config['device'])
            tgt = batch[2].to(config['device'])
            loss = model(input_ids=src, attention_mask=src_mask, decoder_input_ids=tgt, labels=tgt)[0]
            model.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
            optimizer.step()
            scheduler.step()
            train_loss.append(loss.item())

        print('Rank:', config['local_rank'], 'Epoch:', epoch, 'Train Loss:', np.mean(train_loss))

        # Evaluation
        with torch.no_grad():
            valid_loss = []
            for step, batch in enumerate(
                    tqdm(dev_data_loader, desc="Dev Iteration", disable=config['local_rank'] not in [-1, 0])):
                src = batch[0].to(config['device'])
                src_mask = batch[1].to(config['device'])
                tgt = batch[2].to(config['device'])
                outputs = model(input_ids=src, attention_mask=src_mask, decoder_input_ids=tgt, labels=tgt)
                loss = outputs[0]
                valid_loss.append(loss.item())
            mean_valid_loss = np.mean(valid_loss)
            if mean_valid_loss < best_loss and bool(config['to_save_model']):
                if config['save_model_file'] and config['save_model_file'].strip():
                    model.save_pretrained(config['save_model_file'])
                best_loss = mean_valid_loss
                print('rank: %d Best dev loss:' % config['local_rank'], best_loss)


def post_process(config):
    def _split(line):
        if '<sep1>' in line:
            a, b = line.split('<sep1>', maxsplit=1)
            if '<sep2>' in b:
                b, c = b.split('<sep2>', maxsplit=1)
                return a.strip(), b.strip(), c.strip()
            return a.strip(), b.strip(), ""
        return line.strip(), "", ""

    input_path_pred = load_json(config['test_output_pred'])
    input_path_pred = list(map(_split, input_path_pred))
    is_fashion = config['domain'] == 'fashion'

    subtask1 = list(map(lambda x: x[0], input_path_pred))
    subtask3 = list(map(lambda x: x[1], input_path_pred))
    subtask2 = list(map(lambda x: x[2], input_path_pred))

    original_data = load_json(config['test_data_original_file'])
    subtask1_dialog_output = []
    count = 0
    for d in original_data['dialogue_data']:
        predictions = []
        for turn in d['dialogue']:
            action_attribute_str = subtask1[count].strip()
            idx_split_action = action_attribute_str.index('[') if \
                '[' in action_attribute_str else len(action_attribute_str)
            action = action_attribute_str[:idx_split_action].strip()
            attributes = action_attribute_str[idx_split_action:].replace('[', '').replace(']', '').strip()
            if is_fashion:
                attributes = list(filter(lambda x: x, map(str.strip, attributes.split(' '))))
                predictions.append({
                    'turn_idx': turn['turn_idx'],
                    'action': action,
                    'attributes': {
                        'attributes': attributes
                    }
                })
            else:
                attribute_dict = dict()
                for kv_pair in attributes.split(','):
                    kv_pair = kv_pair.strip()
                    if not kv_pair:
                        continue
                    if ':' in kv_pair:
                        kv_pair_split = kv_pair.split(':')
                        kv_pair_split = list(filter(lambda x: x.strip(), kv_pair_split))
                        if len(kv_pair_split) == 1:
                            attribute_dict[kv_pair_split[0].strip()] = None
                        else:
                            attribute_dict[kv_pair_split[0].strip()] = kv_pair_split[1].strip()
                predictions.append({
                    'turn_idx': turn['turn_idx'],
                    'action': action,
                    'attributes': attribute_dict
                })
            count += 1
        subtask1_dialog_output.append({
            'dialog_id': d['dialogue_idx'],
            'predictions': predictions
        })

    # Compatible with official simmc evaluation script
    subtask3 = list(map(lambda x: '%s %s %s' % (START_BELIEF_STATE, x, END_OF_BELIEF), subtask3))

    json.dump(subtask1_dialog_output, open(config['test_data_output_subtask1'], 'w'))
    with open(config['test_data_output_subtask2'], 'w') as f:
        for line in subtask2:
            f.writelines(line + '\n')
    with open(config['test_data_output_subtask3'], 'w') as f:
        for line in subtask3:
            f.writelines(line + '\n')
    print('post preprocess complete')


def generation(config):
    arguments = config['args']
    model = _get_models(config)
    tokenizer_enc, tokenizer_dec = _get_enc_dec_tokenizers(config, model)
    if config['load_model_file'] is not None and os.path.exists(config['load_model_file']):
        pretrained_model_states = torch.load(config['load_model_file'] + "/pytorch_model.bin",
                                             map_location=config['device'])
        model.load_state_dict(pretrained_model_states, strict=False)
        print('Loaded model:', config['load_model_file'])
    model.to(config['device'])

    def collate(examples):
        src_list = list(map(lambda x: x[0], examples))
        src_mask_list = list(map(lambda x: x[1], examples))
        tgt_list = list(map(lambda x: x[2], examples))
        if tokenizer_enc._pad_token is None:
            src_pad = pad_sequence(src_list, batch_first=True)
        else:
            src_pad = pad_sequence(src_list, batch_first=True, padding_value=tokenizer_enc.pad_token_id)
        src_mask_pad = pad_sequence(src_mask_list, batch_first=True, padding_value=0)
        if tokenizer_dec._pad_token is None:
            tgt_pad = pad_sequence(tgt_list, batch_first=True)
        else:
            tgt_pad = pad_sequence(tgt_list, batch_first=True, padding_value=tokenizer_dec.pad_token_id)
        return src_pad, src_mask_pad, tgt_pad

    test_dataset = SimmcFusionDataset(arguments.test_data_src,
                                      None,
                                      None,
                                      None,
                                      tokenizer_enc,
                                      tokenizer_dec)

    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(
        test_dataset, sampler=test_sampler, batch_size=config['test_batch_size'], collate_fn=collate
    )
    results = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_dataloader, desc="Test Iteration")):
            src = batch[0].to(config['device'])
            src_mask = batch[1].to(config['device'])
            tgt = batch[2].to(config['device'])
            generated = model.generate(src,
                                       max_length=200 if config['name'] == 'simmc-fusion' else 60,
                                       decoder_start_token_id=tokenizer_dec.pad_token_id,
                                       attention_mask=src_mask,
                                       early_stopping=True)
            decoded = tokenizer_dec.batch_decode(generated)
            pred_rp = _clean_special_characters(decoded, tokenizer_dec, remove_space=config['remove_space'] if \
                'remove_space' in config else True)
            results += pred_rp
    save_json(results, config['test_output_pred'])


def train(config):
    arguments = config['args']
    model = _get_models(config)
    tokenizer_enc, tokenizer_dec = _get_enc_dec_tokenizers(config, model)
    model.to(config['device'])

    print('training on device', config['device'])
    config['tokenizer_enc'] = tokenizer_enc
    config['tokenizer_dec'] = tokenizer_dec

    def collate(examples):
        src_list = list(map(lambda x: x[0], examples))
        src_mask_list = list(map(lambda x: x[1], examples))
        tgt_list = list(map(lambda x: x[2], examples))
        if tokenizer_enc._pad_token is None:
            src_pad = pad_sequence(src_list, batch_first=True)
        else:
            src_pad = pad_sequence(src_list, batch_first=True, padding_value=tokenizer_enc.pad_token_id)
        src_mask_pad = pad_sequence(src_mask_list, batch_first=True, padding_value=0)
        if tokenizer_dec._pad_token is None:
            tgt_pad = pad_sequence(tgt_list, batch_first=True)
        else:
            tgt_pad = pad_sequence(tgt_list, batch_first=True, padding_value=tokenizer_dec.pad_token_id)
        return src_pad, src_mask_pad, tgt_pad

    train_dataset = SimmcFusionDataset(arguments.train_data_src,
                                       arguments.train_data_tgt_subtask1,
                                       arguments.train_data_tgt_subtask2,
                                       arguments.train_data_tgt_subtask3,
                                       tokenizer_enc,
                                       tokenizer_dec)
    dev_dataset = SimmcFusionDataset(arguments.dev_data_src,
                                     arguments.dev_data_tgt_subtask1,
                                     arguments.dev_data_tgt_subtask2,
                                     arguments.dev_data_tgt_subtask3,
                                     tokenizer_enc,
                                     tokenizer_dec)

    dev_sampler = SequentialSampler(dev_dataset)
    dev_dataloader = DataLoader(
        dev_dataset, sampler=dev_sampler, batch_size=config['batch_size'] * 4, collate_fn=collate
    )

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=config['batch_size'], collate_fn=collate
    )
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(parameters, lr=float(arguments.learning_rate), eps=float(config['adam_epsilon']))
    total_steps = len(train_dataloader) * config['num_epoch']
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)

    _train_epoch(model, optimizer, scheduler, train_dataloader, dev_dataloader, config)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', default=None, type=str, required=True,
                        help="Can be one of train, generate and evaluate.")
    parser.add_argument('--config_file', default=None, type=str, required=True,
                        help="The config file for the action.")
    parser.add_argument('--train_data_src', type=str, required=False,
                        help='training data source file')
    parser.add_argument('--train_data_tgt_subtask1', type=str, required=False,
                        help='training data target for subtask #1')
    parser.add_argument('--train_data_tgt_subtask2', type=str, required=False,
                        help='training data target for subtask #2')
    parser.add_argument('--train_data_tgt_subtask3', type=str, required=False,
                        help='training data target for subtask #3')
    parser.add_argument('--dev_data_src', type=str, required=False,
                        help='dev data source file')
    parser.add_argument('--dev_data_tgt_subtask1', type=str, required=False,
                        help='dev data target for subtask #1')
    parser.add_argument('--dev_data_tgt_subtask2', type=str, required=False,
                        help='dev data target for subtask #2')
    parser.add_argument('--dev_data_tgt_subtask3', type=str, required=False,
                        help='dev data target for subtask #3')
    parser.add_argument('--test_data_src', type=str, required=False,
                        help='test data source file')
    parser.add_argument('--test_data_tgt_subtask1', type=str, required=False,
                        help='test data target for subtask #1')
    parser.add_argument('--test_data_tgt_subtask2', type=str, required=False,
                        help='test data target for subtask #2')
    parser.add_argument('--test_data_tgt_subtask3', type=str, required=False,
                        help='test data target for subtask #3')
    parser.add_argument('--test_data_output_subtask1', type=str, required=False,
                        help='test data output for subtask #1')
    parser.add_argument('--test_data_output_subtask2', type=str, required=False,
                        help='test data output for subtask #2')
    parser.add_argument('--test_data_output_subtask3', type=str, required=False,
                        help='test data output for subtask #3')
    parser.add_argument('--test_data_original_file', type=str, required=False,
                        help='test data original file json')
    parser.add_argument('--test_output_pred', type=str, required=False,
                        help='output prediction for subtask #1,#3,#2')
    parser.add_argument('--encoder_decoder_model_name_or_path', type=str, required=True,
                        help='model name')
    parser.add_argument('--domain', type=str, required=False,
                        help='domain name')
    parser.add_argument('--learning_rate', default=1e-5, type=float, required=False,
                        help='learning rate')
    parser.add_argument("--load_model_index", type=int, default=0, required=False,
                        help="which index of the model to load")

    parser.add_argument("--local_rank", type=int, default=0, help="For distributed training: local_rank")
    parser.add_argument("--batch_size", type=int, required=False, help="batch size for training")
    parser.add_argument("--test_batch_size", type=int, default=100, required=False, help="test batch size")

    args = parser.parse_args()
    config_file = args.config_file
    cfg = get_config(config_file)
    cfg['args'] = args
    cfg['local_rank'] = args.local_rank
    cfg['batch_size'] = args.batch_size
    cfg['test_batch_size'] = args.test_batch_size
    device = torch.device("cuda", args.local_rank)
    cfg['device'] = device
    cfg['domain'] = args.domain
    cfg['test_data_original_file'] = args.test_data_original_file
    cfg['save_model_file'] = '%s_%d' % (cfg['save_model_file'], args.load_model_index)
    cfg['load_model_file'] = '%s_%d' % (cfg['load_model_file'], args.load_model_index)
    cfg['encoder_decoder_model_name_or_path'] = args.encoder_decoder_model_name_or_path
    cfg['test_output_pred'] = '%s_%d' % (args.test_output_pred, args.load_model_index)
    cfg['test_data_output_subtask1'] = '%s_%d' % (args.test_data_output_subtask1, args.load_model_index)
    cfg['test_data_output_subtask2'] = '%s_%d' % (args.test_data_output_subtask2, args.load_model_index)
    cfg['test_data_output_subtask3'] = '%s_%d' % (args.test_data_output_subtask3, args.load_model_index)
    if 'train' == args.action:
        train(cfg)
    elif 'generate' == args.action:
        generation(cfg)
    elif 'postprocess' == args.action:
        post_process(cfg)
