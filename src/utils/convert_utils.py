import json
import re

from utils.action_evaluation import evaluate_action_prediction

START_OF_MULTIMODAL_CONTEXTS = '<som>'
END_OF_MULTIMODAL_CONTEXTS = '<eom>'


def parse_furniture_attribute_str(attr_list):
    if not attr_list:
        return dict()
    res = dict()
    attr_str = ' '.join(attr_list).replace(']', '').replace('[', '').split(',')
    for kv_str in attr_str:
        kv_str = kv_str.strip()
        if ':' in kv_str:
            key, value = kv_str.split(':')[:2]
            key = key.strip()
            value = value.strip()
            res[key] = value
        else:
            res[kv_str] = ""
    return res


def parse_flattened_results_from_file(path, key, use_act_pred):
    results = []
    dialogs = json.load(open(path, 'r'))
    for dialog in dialogs['dialogs']:
        for turn in dialog['dialog']:
            if use_act_pred:
                split_turn = turn[key].split()
                act = split_turn[0] if len(split_turn) > 0 else ""
                attr = split_turn[2:-1] if len(split_turn) > 3 else None
                parsed = {
                    'act': act,
                    'attr': attr
                }
            else:
                parsed = parse_flattened_result(turn[key])
            results.append(parsed)

    return results


def parse_action_results_from_data(dialogs_gt, dialogs_pred, is_furniture, key):
    results = []
    for idx_d, dialog in enumerate(dialogs_gt):
        dialog_pred = []
        for idx_turn, turn in enumerate(dialogs_pred[idx_d]['dialog']):
            split_turn = turn[key].split()
            act = split_turn[0] if len(split_turn) > 0 else ""
            attr = split_turn[2:-1] if len(split_turn) > 3 else []
            if is_furniture:
                parsed = {
                    'action': act,
                    'attributes': parse_furniture_attribute_str(attr)
                }
            else:
                parsed = {
                    'action': act,
                    'attributes': {'attributes': attr}
                }
            dialog_pred.append(parsed)
        results.append({
            'dialog_id': dialog['dialog_id'],
            'predictions': dialog_pred
        })

    action_metrics = evaluate_action_prediction(dialogs_gt, results)

    return action_metrics


def parse_action_results_from_file(path_pred, path_target, key):
    dialogs_gt = json.load(open(path_target, 'r'))
    dialogs_pred = json.load(open(path_pred, 'r'))['dialogs']
    is_furniture = 'furniture' in path_target
    return parse_action_results_from_data(dialogs_gt, dialogs_pred, is_furniture, key)


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
    dialog_act_regex = re.compile('([\w:?.?]*)  *\[([^\]]*)\]')
    slot_regex = re.compile('([A-Za-z0-9_.-:]*)  *= ([^,]*)')

    belief = []

    # Parse
    # to_parse: 'DIALOG_ACT_1 : [ SLOT_NAME = SLOT_VALUE, ... ] ...'
    to_parse = to_parse.strip()

    for dialog_act in dialog_act_regex.finditer(to_parse):
        d = dict()
        d['act'] = dialog_act.group(1)
        d['slots'] = []

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
    str_objects = str_objects.replace('\'', ' \' ').replace('[', ' [ ').replace(']', ' ] ')
    return f'{START_OF_MULTIMODAL_CONTEXTS} {str_objects} {END_OF_MULTIMODAL_CONTEXTS}'
