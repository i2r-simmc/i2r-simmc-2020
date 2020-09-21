import argparse
import json

from mm_action_prediction.tools.response_evaluation import evaluate_response_generation


def compute_metrics_by_data(gt_data, predictions):
    model_responses = []
    line_num = 0
    for d in gt_data["dialogue_data"]:
        model_response = {}
        model_response["dialog_id"] = d["dialogue_idx"]
        model_response["predictions"] = []

        for gt_dialog in d["dialogue"]:
            pred = predictions[line_num]
            model_response["predictions"].append({"response": pred})
            line_num += 1

        model_responses.append(model_response)

    return evaluate_response_generation(gt_data, model_responses)


if __name__ == '__main__':
    # Parse input args
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path_target',
                        help='path for target response json (*.json)')
    parser.add_argument('--input_path_predicted',
                        help='path for model prediction output, line-separated format (.txt)')
    parser.add_argument('--output_path_report',
                        help='path for saving evaluation summary (.json)')

    args = parser.parse_args()
    input_path_target = args.input_path_target
    input_path_predicted = args.input_path_predicted
    output_path_report = args.output_path_report

    # Evaluate SubTask 2
    predict_response = []
    with open(args.input_path_predicted) as f:
        for line in f.readlines():
            predict_response.append(line)
    gold_response = json.load(open(args.input_path_target))  #load_json(config['data_folder'] + gold_data_file2)
    report = compute_metrics_by_data(gold_response, predict_response)

    # Save report
    with open(output_path_report, 'w') as f_out:
        json.dump(report, f_out)