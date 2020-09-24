import os
import json
import argparse
import pandas as pd


def gen_report(domain):
    report_data_frame = pd.DataFrame(columns=['Domain', 'Action Accuracy', 'Action Perplexity', 'Attribute Accuracy',
                                              'BLEU-4', 'Dialog Act F1', 'Slot F1'])
    action_accuracy=None
    action_perplexity=None
    attribute_accuracy = None
    blue4 = None
    dialog_act_f1 = None
    slot_f1 = None
    output_report_file_path = os.path.join('..', 'output', domain, 'prediction_report.csv')
    for task_id in range(1, 4):
        report_json_file_name = 'output_subtask%d_report.json' % task_id
        file_path = os.path.join('..', 'output', domain, report_json_file_name)
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
    args = parser.parse_args()
    domain = args.domain
    gen_report(domain)