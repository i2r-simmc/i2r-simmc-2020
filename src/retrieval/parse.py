import json
from collections import defaultdict
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str)
parser.add_argument('--domain', type=str)
parser.add_argument('--train_dir', default = '../data/", type = str)
args = parser.parse_args()

with open(os.path.join(args.train_dir, 'simmc_{}/{}_{}_dials.json'.format(args.domain, args.domain, args.mode)) as infile:
    data = json.load(infile)
    
with open(os.path.join(args.train_dir, 'simmc_{}/{}_{}_dials_retrieval_candidates.json'.format(args.domain, args.domain, args.mode)) as infile:
    responses = json.load(infile)
    
dial_index = 0

print("Preprocessing " + os.path.join(args.train_dir, 'simmc_{}/{}_{}_dials.json'.format(args.domain, args.domain, args.mode))
print("Writing to " + os.path.join(args.train_dir, 'simmc_{}/{}.txt'.format(args.domain, args.mode))

outfile = open(os.path.join(args.train_dir, 'simmc_{}/{}.txt'.format(args.domain, args.mode), 'w')
for content in data['dialogue_data']:
    turn_index = 0
    turns = content['dialogue']
    context = []
    for turn in turns:
        options = responses['retrieval_candidates'][dial_index]['retrieval_candidates'][turn_index]['retrieval_candidates']
        correct_id = options[0]
        option = responses['system_transcript_pool'][correct_id]
        text = 'participant_1'.replace('_', ' ') + ': ' + turn['transcript']
        context.append(text)
        outfile.write('{}\t{}\t{}\n'.format(1, '\t'.join(context), option.strip()))
        
        if args.mode != 'train':
            negs = options[1:]
            for neg in negs:
                outfile.write('{}\t{}\t{}\n'.format(0, '\t'.join(context), responses['system_transcript_pool'][neg].strip()))
        
        text = 'participant_2'.replace('_', ' ') + ': ' + turn['system_transcript']
        context.append(text)               
        
        turn_index += 1
        
    dial_index += 1

print("Preprocessing completed.")
