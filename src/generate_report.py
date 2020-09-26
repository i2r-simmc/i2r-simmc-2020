import os
import json
import argparse
import pandas as pd

REPORT_FILE_NAMES = [
    'dstc9-simmc-%s-%s-subtask-1-report.json',
    'dstc9-simmc-%s-%s-subtask-2-generation-report.json',
    'dstc9-simmc-%s-%s-subtask-3-report.json'
]


def gen_report(domain, test_split_name, model_type_combined):
    report_data_frame = pd.DataFrame(columns=['Domain', 'Action Accuracy', 'Action Perplexity', 'Attribute Accuracy',
                                              'BLEU-4', 'Dialog Act F1', 'Slot F1'])
    action_accuracy=None
    action_perplexity=None
    attribute_accuracy = None
    blue4 = None
    dialog_act_f1 = None
    slot_f1 = None
    output_report_file_path = os.path.join('..', 'output', domain, model_type_combined, test_split_name, 'report.joint-learning.csv')
    for task_id in range(1, 4):
        report_json_file_name = REPORT_FILE_NAMES[task_id - 1] % (test_split_name, domain)
        file_path = os.path.join('..', 'output', domain, model_type_combined, test_split_name, report_json_file_name)
        if not os.path.exists(file_path):
            return
        report = json.load(open(file_path))
        if 'action_accuracy' in report:
            action_accuracy = report['action_accuracy']
        if 'action_perplexity' in report:
            action_perplexity = report['action_perplexity']
        if 'attribute_accuracy' in report:
            attribute_accuracy = report['attribute_accuracy']
        if 'bleu' in report:
            blue4 = report['bleu']
        if 'act_f1' in report:
            dialog_act_f1 = report['act_f1']
        if 'slot_f1' in report:
            slot_f1 = report['slot_f1']
    report_data_frame.loc[0] = [domain, action_accuracy, action_perplexity, attribute_accuracy, blue4, dialog_act_f1, slot_f1]
    report_data_frame.to_csv(output_report_file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', default=None, type=str, required=True,
                        help="domain of the model")
    parser.add_argument('--test_split_name', default=None, type=str, required=True,
                        help="name of the test split")
    parser.add_argument('--model_type_combined', default=None, type=str, required=True,
                        help="combined model name")

    args = parser.parse_args()
    domain = args.domain
    test_split_name = args.test_split_name
    model_type_combined = args.model_type_combined
    gen_report(domain, test_split_name, model_type_combined)