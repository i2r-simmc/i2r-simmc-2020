import json
import argparse
import os
import re

# DSTC style dataset fieldnames
from utils.convert_utils import represent_visual_objects
from preprocessing.convert_simmc import convert_action

FIELDNAME_DIALOG = 'dialogue'
FIELDNAME_USER_UTTR = 'transcript'
FIELDNAME_ASST_UTTR = 'system_transcript'
FIELDNAME_BELIEF_STATE = 'belief_state'
FIELDNAME_VISUAL_OBJECTS = 'visual_objects'

# Templates for GPT-2 formatting
START_OF_MULTIMODAL_CONTEXTS = '<som>'
END_OF_MULTIMODAL_CONTEXTS = '<eom>'
START_BELIEF_STATE = '=> Belief State :'
END_OF_BELIEF = '<EOB>'
END_OF_SENTENCE = '<EOS>'

TEMPLATE_PREDICT = '{context} {START_BELIEF_STATE} '
TEMPLATE_TARGET = '{context} {START_BELIEF_STATE} {belief_state} ' \
                  '{END_OF_BELIEF} {response} {END_OF_SENTENCE}'
TEMPLATE_ACTION = '%s [ %s ]'

META_DATA_PATH = 'data/simmc_furniture/furniture_metadata.csv'

SUBTASK = 'dominant-action'
IGNORE_ATTRIBUTES = [
    "minPrice",
    "maxPrice",
    "furniture_id",
    "material",
    "decorStyle",
    "intendedRoom",
    "raw_matches",
    "focus"  # fashion
]


def convert(data_path, generate_belief_state=False, use_multimodal_contexts=False,
            use_action_prediction=False, test_split_name='test-std'):
    """
        Input: JSON representation of the dialogs
        Output: line-by-line stringified representation of each turn
    """
    for category in ['fashion', 'furniture']:
        for split in ['train', 'dev', 'devtest', test_split_name]:
            # Load the dialogue data
            filepath = os.path.join(data_path, 'simmc_%s' % category,
                                               '%s_%s_dials.json' % (category, split))
            if not os.path.exists(filepath):
                continue
            with open(filepath, 'r') as f_in:
                data = json.load(f_in)

            if category == 'furniture' and split != test_split_name:
                if not os.path.exists(os.path.join(data_path, 'simmc_%s' % category, 'furniture_%s_dials_api_calls.json' % split)):
                    continue
                with open(os.path.join(data_path, 'simmc_%s' % category, 'furniture_%s_dials_api_calls.json' % split), 'r') as f_in:
                    api_call_data = json.load(f_in)
            if category == 'furniture':
                task_mapping = None
            else:
                task_mapping = {ii["task_id"]: ii for ii in data["task_mapping"]}
            data = data['dialogue_data']

            converted_data = []
            oov = set()
            for idx_diag, each_dialogue in enumerate(data):
                if category == 'furniture' and split != test_split_name:
                    api_d = api_call_data[idx_diag]
                dialogue_dict = {'dialog': []}
                if category == 'furniture':
                    mm_state = None
                else:
                    dialog_id = each_dialogue["dialogue_idx"]
                    if "dialogue_task_id" not in each_dialogue:
                        # Assign a random task for missing ids.
                        print("Dialogue task Id missing: {}".format(dialog_id))
                        mm_state = task_mapping[1874]
                    else:
                        mm_state = task_mapping[each_dialogue["dialogue_task_id"]]

                for idx_c, conversation in enumerate(each_dialogue['dialogue']):
                    if not generate_belief_state and not use_action_prediction:
                        if 'system_transcript' in conversation:
                            dialogue_dict['dialog'].append({
                                'answer': conversation['system_transcript'],
                                'question': conversation['transcript'],
                            })
                        else:
                            dialogue_dict['dialog'].append({
                                'question': conversation['transcript'],
                            })
                    elif use_action_prediction and split != test_split_name:
                        if category == 'fashion':
                            action = convert_action(conversation, mm_state)
                            if action['action_supervision'] and action['action_supervision']['attributes']:
                                action_str = TEMPLATE_ACTION % (action['action'],
                                                                ' '.join(action['action_supervision']['attributes']))
                                oov.update(action['action_supervision']['attributes'])
                            else:
                                action_str = action['action'] if action['action'] else 'none'
                            if action['action']:
                                oov.add(action['action'])
                            else:
                                oov.add('none')
                        else:
                            action_str = api_d['actions'][idx_c]['action']
                            args = api_d['actions'][idx_c]['action_supervision']['args']
                            if args:
                                args = {k: v for k, v in args.items() if k not in IGNORE_ATTRIBUTES}
                            args_str = ' [ ]'
                            if args:
                                args_str = ' , '.join([' : '.join(list(map(str, e))) for e in list(args.items())])
                                attribute_tokens = args_str.split()
                                oov.update(attribute_tokens)
                                args_str = ' [ %s ]' % args_str
                            oov.add(action_str)
                            action_str += args_str

                        if use_multimodal_contexts:
                            visual_objects = conversation[FIELDNAME_VISUAL_OBJECTS]
                            visual_object_context = represent_visual_objects(visual_objects)
                        else:
                            visual_object_context = None
                        
                        if split != test_split_name:
                            dialogue_dict['dialog'].append({
                                'answer': conversation['system_transcript'],
                                'question': conversation['transcript'],
                                'visual_objects': visual_object_context,
                                'target': action_str,
                            })
                        else:
                            dialogue_dict['dialog'].append({
                                'question': conversation['transcript'],
                                'visual_objects': visual_object_context
                            })
                    else:
                        if FIELDNAME_BELIEF_STATE in conversation:
                            belief_state = []
                            user_belief = conversation[FIELDNAME_BELIEF_STATE]
                            for bs_per_frame in user_belief:
                                str_belief_state_per_frame = "{act} [ {slot_values} ]".format(
                                    act=bs_per_frame['act'].strip(),
                                    slot_values=', '.join(
                                        [f'{kv[0].strip()} = {kv[1].strip()}'
                                         for kv in bs_per_frame['slots']])
                                )
                                belief_state.append(str_belief_state_per_frame)

                                oov.update(re.split(r'[:.]', bs_per_frame['act']))
                                for kv in bs_per_frame['slots']:
                                    slot_name = kv[0]
                                    oov.add(slot_name)
                            str_belief_state = ' '.join(belief_state)

                        if use_multimodal_contexts:
                            visual_objects = conversation[FIELDNAME_VISUAL_OBJECTS]
                            visual_object_context = represent_visual_objects(visual_objects)
                        else:
                            visual_object_context = None

                        if split != test_split_name:
                            dialogue_dict['dialog'].append({
                                'answer': conversation['system_transcript'],
                                'question': conversation['transcript'],
                                'visual_objects': visual_object_context,
                                'target': str_belief_state,
                            })
                        else:
                            dialogue_dict['dialog'].append({
                                'question': conversation['transcript'],
                                'visual_objects': visual_object_context
                            })

                converted_data.append(dialogue_dict)

            converted_data = {
                'dialogs': converted_data,
                'version': '0.1',
                'type': 'scene_aware_dialog',
            }
            if not generate_belief_state and not use_action_prediction:
                json.dump(converted_data, open(os.path.join(data_path, 'simmc_%s' % category, 'simmc_%s_resp_%s.json' %
                                                            (category, split)), 'w'))
            elif use_action_prediction:
                json.dump(converted_data, open(os.path.join(data_path, 'simmc_%s' % category, 'simmc_%s_api_%s.json' %
                                                            (category, split)), 'w'))
            else:
                json.dump(converted_data, open(os.path.join(data_path, 'simmc_%s' % category, 'simmc_%s_dst_%s.json' %
                                                            (category, split)), 'w'))


def main():
    # Parse input args
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',
                        help='data path of the training data',
                        type=str)
    parser.add_argument('--generate_belief_state',
                        action='store_true',
                        help='determine whether to generate belief state each turn')
    parser.add_argument('--use_multimodal_contexts',
                        action='store_true',
                        help='determine whether to use multimodel contexts')
    parser.add_argument('--use_action_prediction',
                        action='store_true',
                        help='determine whether to use action prediction')
    parser.add_argument('--test_split_name', type=str, default='test-std',
                        help='name of test split')

    args = parser.parse_args()
    data_path = args.data_path
    generate_belief_state = bool(args.generate_belief_state)
    use_multimodal_contexts = bool(args.use_multimodal_contexts)
    use_action_prediction = bool(args.use_action_prediction)

    # Convert the data into MTN friendly format
    convert(data_path, generate_belief_state=generate_belief_state,
            use_multimodal_contexts=use_multimodal_contexts,
            use_action_prediction=use_action_prediction, 
            test_split_name=args.test_split_name)


if __name__ == '__main__':
    main()
