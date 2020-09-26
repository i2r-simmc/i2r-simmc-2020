#!/usr/bin/env python3
"""
    Script for converting the main SIMMC datasets (.JSON format)
    into the line-by-line stringified format (and back).

    The reformatted data is used as input for the GPT-2 based
    DST model baseline.
"""
import json
import re
import os

# DSTC style dataset fieldnames
FIELDNAME_DIALOG = 'dialogue'
FIELDNAME_USER_UTTR = 'transcript'
FIELDNAME_ASST_UTTR = 'system_transcript'
FIELDNAME_BELIEF_STATE = 'belief_state'
FIELDNAME_STATE_GRAPH_0 = 'state_graph_0'
FIELDNAME_VISUAL_OBJECTS = 'visual_objects'

# Templates for GPT-2 formatting
START_OF_MULTIMODAL_CONTEXTS = '<SOM>'
END_OF_MULTIMODAL_CONTEXTS = '<EOM>'
START_BELIEF_STATE = '=> Belief State :'
END_OF_BELIEF = '<EOB>'
END_OF_SENTENCE = '<EOS>'

TEMPLATE_PREDICT = '{context} {START_BELIEF_STATE} '


def convert_json_to_flattened(
        input_path_json,
        output_path_predict,
        output_path_target,
        len_context=2,
        use_multimodal_contexts=True,
        test_data=False):
    """
        Input: JSON representation of the dialogs
        Output: line-by-line stringified representation of each turn
    """

    if not os.path.exists(input_path_json):
        return
    with open(input_path_json, 'r') as f_in:
        data = json.load(f_in)['dialogue_data']

    predicts = []
    targets = []
    for _, dialog in enumerate(data):

        prev_asst_uttr = None
        lst_context = []

        for turn in dialog[FIELDNAME_DIALOG]:
            user_uttr = turn[FIELDNAME_USER_UTTR].replace('\n', ' ').strip()
            asst_uttr = turn[FIELDNAME_ASST_UTTR].replace('\n', ' ').strip()
            if not test_data:
                user_belief = turn[FIELDNAME_BELIEF_STATE]
                

            # Format main input context
            context = ''
            if prev_asst_uttr:
                context += f'System : {prev_asst_uttr} '
            context += f'User : {user_uttr}'
            prev_asst_uttr = asst_uttr

            # Add multimodal contexts
            if use_multimodal_contexts:
                visual_objects = turn[FIELDNAME_VISUAL_OBJECTS]
                context += ' ' + represent_visual_objects(visual_objects)

            # Concat with previous contexts
            lst_context.append(context)
            context = ' '.join(lst_context[-len_context:])

            # Format belief state
            if not test_data:
                belief_state = []
                for bs_per_frame in user_belief:
                    str_belief_state_per_frame = "{act} [ {slot_values} ]".format(
                        act=bs_per_frame['act'].strip(),
                        slot_values=', '.join(
                            [f'{kv[0].strip()} = {kv[1].strip()}'
                                for kv in bs_per_frame['slots']])
                    )
                    belief_state.append(str_belief_state_per_frame)

                str_belief_state = ' '.join(belief_state)

            # Format the main input
            predict = TEMPLATE_PREDICT.format(
                context=context,
                START_BELIEF_STATE=START_BELIEF_STATE,
            )
            predicts.append(predict)

            if not test_data:
                # Format the main output
                target = '%s %s %s' % (START_BELIEF_STATE, str_belief_state, END_OF_BELIEF)
                targets.append(target)

    # Create a directory if it does not exist
    directory = os.path.dirname(output_path_predict)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # Output into text files
    with open(output_path_predict, 'w') as f_predict:
        X = '\n'.join(predicts)
        f_predict.write(X)

    if len(targets) > 0:
        directory = os.path.dirname(output_path_target)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        
        with open(output_path_target, 'w') as f_target:
            Y = '\n'.join(targets)
            f_target.write(Y)


def represent_visual_objects(visual_objects):
    # Stringify visual objects (JSON)
    target_attributes = ['pos', 'color', 'type', 'class_name', 'decor_style']

    list_str_objects = []
    for obj_name, obj in visual_objects.items():
        s = obj_name + ' :'
        for target_attribute in target_attributes:
            if target_attribute in obj:
                target_value = obj.get(target_attribute)
                if target_value == '' or target_value == []:
                    pass
                else:
                    s += f' {target_attribute} {str(target_value)}'
        list_str_objects.append(s)

    str_objects = ' '.join(list_str_objects)
    return f'{START_OF_MULTIMODAL_CONTEXTS} {str_objects} {END_OF_MULTIMODAL_CONTEXTS}'


def parse_flattened_results_from_file(path):
    results = []
    with open(path, 'r') as f_in:
        for line in f_in:
            # Remove eos token and restore action
            line = line.replace('<EOS>', '').strip()
            parsed = parse_flattened_result(line)
            results.append(parsed)

    return results


def parse_flattened_result(to_parse):
    """
        Parse out the belief state from the raw text.
        Return an empty list if the belief state can't be parsed

        Input:
        - A single <str> of flattened result
          e.g. 'User: Show me something else => Belief State : DA:REQUEST ...'

        Output:
        - Parsed result in a JSON format, where the format is:
            [
                {
                    'act': <str>  # e.g. 'DA:REQUEST',
                    'slots': [
                        <str> slot_name,
                        <str> slot_value
                    ]
                }, ...  # End of a frame
            ]  # End of a dialog
    """
    dialog_act_regex = re.compile(r'([\w:?.?]*)  *\[([^\]]*)\]')
    slot_regex = re.compile(r'([A-Za-z0-9_.-:]*)  *= ([^,]*)')

    belief = []

    # Parse
    # to_parse: 'DIALOG_ACT_1 : [ SLOT_NAME = SLOT_VALUE, ... ] ...'

    for dialog_act in dialog_act_regex.finditer(to_parse):
        d = {
            'act': dialog_act.group(1),
            'slots': []
        }

        for slot in slot_regex.finditer(dialog_act.group(2)):
            d['slots'].append(
                [
                    slot.group(1).strip(),
                    slot.group(2).strip()
                ]
            )

        if d != {}:
            belief.append(d)

    return belief
