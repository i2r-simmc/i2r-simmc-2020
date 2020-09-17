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
import math

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
TEMPLATE_TARGET = '{context} {START_BELIEF_STATE} {belief_state} ' \
                  '{END_OF_BELIEF} {response} {END_OF_SENTENCE}'

# ---------------Action Variables-------------------------'
# sub-tasks
DOMINANT_ACTION = "dominant-action"
MULTI_ACTION = "multi-action"

### log field names ###
DOMAIN = "furniture"
ACTION_NAME = "actionName"
RAW_MATCHES = "raw_matches"
MATCHES = "matches"
ACTION_METADATA = "actionMetadata"
FURNITURE_TYPE = "furnitureType"
MIN_PRICE = "minPrice"
MAX_PRICE = "maxPrice"
NEXT_STATE = "nextState"
PREVIOUS_STATE = "previousState"
TEXT_PREFABS_CAROUSEL = "textPrefabsInCarousel"
PREFAB_IN_FOCUS = "prefabInFocus"
PREFABS_IN_CAROUSEL = "prefabsInCarousel"
SHARED_FOCUS = "sharedPrefabInFocus"
SHARED_CAROUSEL = "sharedPrefabsInCarousel"

# Keystroke names
SEARCH_FURNITURE = "SearchFurniture"
BRING_OBJECT_TO_FOCUS = "BringObjectToFocus"
REMOVE_OBJECT_FROM_FOCUS = "RemoveObjectInFocus"
ROTATE = "Rotate"  # RotateRight, RotateUp, RotateDown, RotateBack, etc.
FURNITURE_CLICK = "FurnitureClick"
PREVIOUS = "Previous"
NEXT = "Next"
SHARE = "Share"

# API Action names
SEARCH_FURNITURE_ACTION = "SearchFurniture"
NAVIGATE_CAROUSEL_ACTION = "NavigateCarousel"
FOCUS_ON_FURNITURE_ACTION = "FocusOnFurniture"
ROTATE_ACTION = "Rotate"
ADD_TO_CART_ACTION = "AddToCart"
GET_INFO_ACTION = "SpecifyInfo"
NONE_ACTION = "None"

# Keep only these matching attributes for GET_INFO
FILTER_MATCHES = ["price", "dimensions", "info", "material", "color"]

# Action preference order for dominant action sub-task
PREFERENCE_ORDER = [
    ADD_TO_CART_ACTION,
    GET_INFO_ACTION,
    ROTATE_ACTION,
    SEARCH_FURNITURE_ACTION,
    FOCUS_ON_FURNITURE_ACTION,
    NAVIGATE_CAROUSEL_ACTION,
    NONE_ACTION,
]

# api / arg field names
API = "api"
ARGS = "args"
DIRECTION = "direction"
FURNITURE_ID = "furniture_id"
NAVIGATE_DIRECTION = "navigate_direction"
POSITION = "position"


def extract_info_attributes(turn):
    """Extract information attributes for current round using NLU annotations.

    Args:
        turn: Current round information

    Returns:
        get_attribute_matches: Information attributes
    """
    user_annotation = eval(turn["transcript_annotated"])
    assistant_annotation = eval(turn["transcript_annotated"])
    annotation = user_annotation + assistant_annotation
    # annotation = user_annotation
    all_intents = [ii["intent"] for ii in annotation]
    get_attribute_matches = []
    for index, intent in enumerate(all_intents):
        if any(
                ii in intent
                for ii in ("DA:ASK:GET", "DA:ASK:CHECK", "DA:INFORM:GET")
        ):
            # If there is no attribute added, default to info.
            if "." not in intent:
                get_attribute_matches.append("info")
                continue

            attribute = intent.split(".")[-1]
            if attribute == "info":
                new_matches = [
                    ii["id"].split(".")[-1]
                    for ii in annotation[index]["slots"]
                    if "INFO" in ii["id"]
                ]
                if len(new_matches):
                    get_attribute_matches.extend(new_matches)
                else:
                    get_attribute_matches.append("info")
            elif attribute != "":
                get_attribute_matches.append(attribute)
    return sorted(set(get_attribute_matches))


def convert_action(turn, mm_state):
    insert_item = {
        "turn_idx": turn["turn_idx"],
        "action": "None",
        "action_supervision": None
    }
    keystrokes = turn.get("raw_assistant_keystrokes", [])
    # Get information attributes given the asset id.
    attributes = extract_info_attributes(turn)
    if keystrokes:
        focus_image = int(keystrokes[0]["image_id"])
        # Change of focus image -> Search in dataset or memory.
        if focus_image in mm_state["memory_images"]:
            insert_item["action"] = "SearchMemory"
            insert_item["action_supervision"] = {
                "focus": focus_image,
                "attributes": attributes,
            }
        elif focus_image in mm_state["database_images"]:
            insert_item["action"] = "SearchDatabase"
            insert_item["action_supervision"] = {
                "focus": focus_image,
                "attributes": attributes,
            }
        else:
            print("Undefined action; using None instead")
    else:
        # Check for SpecifyInfo action.
        # Get information attributes given the asset id.
        attributes = extract_info_attributes(turn)
        if len(attributes):
            insert_item["action"] = "SpecifyInfo"
            insert_item["action_supervision"] = {
                "attributes": attributes
            }
        else:
            # AddToCart action.
            for intent_info in eval(turn["transcript_annotated"]):
                if "DA:REQUEST:ADD_TO_CART" in intent_info["intent"]:
                    insert_item["action"] = "AddToCart"
                    insert_item["action_supervision"] = None
    return insert_item


def get_args_for_furniture_click(stroke):
    entry = json.loads(stroke)
    text_next_state = entry[NEXT_STATE][TEXT_PREFABS_CAROUSEL]
    text_prev_state = entry[PREVIOUS_STATE][TEXT_PREFABS_CAROUSEL]
    arg = [text for text in text_next_state if text not in text_prev_state]
    return arg


def get_keystrokes_with_args(raw_keystroke_list, price_dict):
    """Gets the keystrokes + args from the raw keystrokes in the logs after some processing

    Args:
        raw_keystroke_list: list extracted from the logs
        price_dict : dict from furniture type to the default min & max prices

    Returns:
        list of keystrokes where each keystroke is an api name + corresponding args
    """
    keystrokes_with_args = []
    for stroke in raw_keystroke_list:
        keystroke = json.loads(stroke)[ACTION_NAME]

        if keystroke == SEARCH_FURNITURE:
            furniture_type_arg = json.loads(stroke)[ACTION_METADATA][FURNITURE_TYPE]

            if furniture_type_arg != "":
                action_metadata = json.loads(stroke)[ACTION_METADATA]

                # check if the prices are close to the default populated prices.
                # if yes, replace by -1
                min_price_arg = action_metadata[MIN_PRICE]
                max_price_arg = action_metadata[MAX_PRICE]
                if math.isclose(
                        min_price_arg, price_dict[furniture_type_arg][0], abs_tol=0.9
                ):
                    action_metadata[MIN_PRICE] = -1
                if math.isclose(
                        max_price_arg, price_dict[furniture_type_arg][1], abs_tol=0.9
                ):
                    action_metadata[MAX_PRICE] = -1
                keystrokes_with_args.append(
                    {
                        API: keystroke,
                        NEXT_STATE: json.loads(stroke)[NEXT_STATE],
                        PREVIOUS_STATE: json.loads(stroke)[PREVIOUS_STATE],
                        ARGS: action_metadata,
                    }
                )

        elif keystroke == BRING_OBJECT_TO_FOCUS:
            keystrokes_with_args.append(
                {
                    API: keystroke,
                    NEXT_STATE: json.loads(stroke)[NEXT_STATE],
                    PREVIOUS_STATE: json.loads(stroke)[PREVIOUS_STATE],
                    ARGS: json.loads(stroke)[NEXT_STATE][PREFAB_IN_FOCUS],
                }
            )
        elif keystroke.startswith(ROTATE):
            if (
                    json.loads(stroke)[NEXT_STATE][PREFAB_IN_FOCUS] is not None
                    and json.loads(stroke)[NEXT_STATE][PREFAB_IN_FOCUS] != ""
            ):
                keystrokes_with_args.append(
                    {
                        API: keystroke,
                        NEXT_STATE: json.loads(stroke)[NEXT_STATE],
                        PREVIOUS_STATE: json.loads(stroke)[PREVIOUS_STATE],
                        ARGS: json.loads(stroke)[NEXT_STATE][PREFAB_IN_FOCUS],
                    }
                )
            else:
                keystrokes_with_args.append(
                    {
                        API: keystroke,
                        NEXT_STATE: json.loads(stroke)[NEXT_STATE],
                        PREVIOUS_STATE: json.loads(stroke)[PREVIOUS_STATE],
                        ARGS: json.loads(stroke)[NEXT_STATE]["sharedPrefabInFocus"],
                    }
                )
        elif keystroke == FURNITURE_CLICK:
            keystrokes_with_args.append(
                {
                    API: keystroke,
                    NEXT_STATE: json.loads(stroke)[NEXT_STATE],
                    PREVIOUS_STATE: json.loads(stroke)[PREVIOUS_STATE],
                    ARGS: get_args_for_furniture_click(stroke),
                }
            )
        elif keystroke == SHARE:
            keystrokes_with_args.append(
                {
                    API: keystroke,
                    NEXT_STATE: json.loads(stroke)[NEXT_STATE],
                    PREVIOUS_STATE: json.loads(stroke)[PREVIOUS_STATE],
                    ARGS: None,
                }
            )
            # fix some PREVIOUS_STATE log strangeness
            keystrokes_with_args[-1][PREVIOUS_STATE][PREFAB_IN_FOCUS] = \
                keystrokes_with_args[-1][NEXT_STATE][SHARED_FOCUS]
            keystrokes_with_args[-1][PREVIOUS_STATE][PREFABS_IN_CAROUSEL] = \
                keystrokes_with_args[-1][NEXT_STATE][SHARED_CAROUSEL]
        else:
            keystrokes_with_args.append(
                {
                    API: keystroke,
                    NEXT_STATE: json.loads(stroke)[NEXT_STATE],
                    PREVIOUS_STATE: json.loads(stroke)[PREVIOUS_STATE],
                    ARGS: None,
                }
            )
    return keystrokes_with_args


def convert_json_to_flattened(
        input_path_json,
        output_path_predict,
        output_path_target,
        len_context=2,
        use_multimodal_contexts=True,
        input_path_special_tokens='',
        output_path_special_tokens='',
        is_fashion=True,
        pred_action=False,
        token_level=False):
    """
        Input: JSON representation of the dialogs
        Output: line-by-line stringified representation of each turn
    """

    # Load the dialogue data
    with open(input_path_json, 'r') as f_in:
        data = json.load(f_in)

    if 'task_mapping' in data:
        task_mapping = {ii["task_id"]: ii for ii in data["task_mapping"]}
    else:
        task_mapping = None
    data = data['dialogue_data']

    # Load special tokens
    predicts = []
    targets = []
    if input_path_special_tokens != '':
        with open(input_path_special_tokens, 'r') as f_in:
            special_tokens = json.load(f_in)
    else:
        special_tokens = {
            "eos_token": END_OF_SENTENCE,
            "additional_special_tokens": [
                END_OF_BELIEF
            ]
        }
        if use_multimodal_contexts:
            special_tokens = {
                "eos_token": END_OF_SENTENCE,
                "additional_special_tokens": [
                    END_OF_BELIEF,
                    START_OF_MULTIMODAL_CONTEXTS,
                    END_OF_MULTIMODAL_CONTEXTS
                ]
            }

    if output_path_special_tokens != '':
        # If a new output path for special tokens is given,
        # we track new OOVs
        oov = set()
    for _, dialog in enumerate(data):

        prev_asst_uttr = None
        lst_context = []
        dialog_id = dialog["dialogue_idx"]
        if task_mapping:
            if "dialogue_task_id" not in dialog:
                # Assign a random task for missing ids.
                print("Dialogue task Id missing: {}".format(dialog_id))
                mm_state = task_mapping[1874]
            else:
                mm_state = task_mapping[dialog["dialogue_task_id"]]

        for turn in dialog[FIELDNAME_DIALOG]:
            # User Transcript
            user_uttr = turn[FIELDNAME_USER_UTTR].replace('\n', ' ').strip()
            # User belief state (targets)
            user_belief = turn[FIELDNAME_BELIEF_STATE]
            # System assistant Transcript
            asst_uttr = turn[FIELDNAME_ASST_UTTR].replace('\n', ' ').strip()

            if pred_action:
                action = convert_action(turn, mm_state)
                print
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
            belief_state = []
            for bs_per_frame in user_belief:
                bs_per_frame['act'] = bs_per_frame['act'].replace('.', ':')
                str_belief_state_per_frame = "{act} [ {slot_values} ]".format(
                    act=bs_per_frame['act'].strip(),
                    slot_values=', '.join(
                        [f'{kv[0].strip()} = {kv[1].strip()}'
                         for kv in bs_per_frame['slots']])
                )
                belief_state.append(str_belief_state_per_frame)

                # Track OOVs
                if output_path_special_tokens != '' and not token_level:
                    oov.add(bs_per_frame['act'])
                    for kv in bs_per_frame['slots']:
                        slot_name = kv[0]
                        oov.add(slot_name)
                        # slot_name, slot_value = kv[0].strip(), kv[1].strip()
                        # oov.add(slot_name)
                        # oov.add(slot_value)
                else:
                    oov.add('DA')
                    oov.add('da')

            str_belief_state = ' '.join(belief_state)

            # Format the main input
            predict = TEMPLATE_PREDICT.format(
                context=context,
                START_BELIEF_STATE=START_BELIEF_STATE,
            )
            predicts.append(predict)

            # Format the main output
            target = '%s %s' % (str_belief_state, END_OF_SENTENCE)
            targets.append(target)

    # Create a directory if it does not exist
    directory = os.path.dirname(output_path_predict)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    directory = os.path.dirname(output_path_target)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # Output into text files
    with open(output_path_predict, 'w') as f_predict:
        X = '\n'.join(predicts)
        f_predict.write(X)

    with open(output_path_target, 'w') as f_target:
        Y = '\n'.join(targets)
        f_target.write(Y)

    if output_path_special_tokens != '':
        # Create a directory if it does not exist
        directory = os.path.dirname(output_path_special_tokens)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        with open(output_path_special_tokens, 'w') as f_special_tokens:
            # Add oov's (acts and slot names, etc.) to special tokens as well
            special_tokens['additional_special_tokens'].extend(list(oov))
            json.dump(special_tokens, f_special_tokens)


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


def parse_flattened_results_from_file(path, key):
    results = []
    dialogs = json.load(open(path, 'r'))
    for dialog in dialogs['dialogs']:
        for turn in dialog['dialog']:
            parsed = parse_flattened_result(turn[key].lower())
            results.append(parsed)

    return results


def parse_flattened_results_from_flattened_file(flatten_list):
    flatten_list = [e.replace(' ', '').replace('[', ' [ ').replace(']', ' ] ').replace(',', ' , ').replace(
        '=', ' = ').replace('<eos>', '').replace('<eod>', '').replace('[sep]', '').replace('</s>', '').replace('<s>',
                                                                                                               '').lower()
                    for e in flatten_list]
    return [parse_flattened_result(e.lower()) for e in flatten_list]


def parse_flattened_act_results_from_file(flatten_list):
    res = []
    for e in flatten_list:
        token_spt = e.split(' ')
        token_spt_set = set()
        if len(token_spt) == 1:
            res.append(e)
        else:
            token_e = []
            for each_token in token_spt:
                if not each_token.strip():
                    continue
                if each_token in token_spt_set and each_token not in (',', ':', '.'):
                    break
                token_e.append(each_token)
                token_spt_set.add(each_token)
            if token_e[-1] == ':':
                token_e = token_e[:-1]
            if token_e[-1] != ']':
                token_e.append(']')
            token_e_str = ' '.join(token_e)
            # Compatible with furniture data
            token_e_str = token_e_str.replace('navigatedirection', 'navigate_direction')
            res.append(token_e_str)
    return res


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
        d['act'] = dialog_act.group(1).replace('.', ':')
        d['slots'] = []

        for slot in slot_regex.finditer(dialog_act.group(2)):
            d['slots'].append(
                [
                    slot.group(1).strip(),
                    slot.group(2).strip().replace(' ', '')
                ]
            )

        if d != {}:
            belief.append(d)

    return belief


def parse_flattened_result_only_intent(to_parse):
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

    def parse_intent(intent_str):
        pos_list = []
        for i in range(len(intent_str)):
            for j in range(i + 1, min(len(intent_str), i + 4)):
                sub_str = intent_str[i:j + 1]
                if sub_str == 'da:' or sub_str == 'err:':
                    pos_list.append(i)
                    break
        if not pos_list or len(pos_list) == 1:
            return [intent_str]
        return [intent_str[:pos] for pos in pos_list[1:]] + [intent_str[pos_list[-1]:]]

    belief = []

    # Parse
    # to_parse: 'DIALOG_ACT_1 : [ SLOT_NAME = SLOT_VALUE, ... ] ...'
    to_parse = to_parse.strip()

    intents = parse_intent(to_parse)
    for idx, dialog_act in enumerate(intents):
        d = dict()
        d['act'] = dialog_act.replace('.', ':')
        d['slots'] = []

        if d != {}:
            belief.append(d)

    return belief
