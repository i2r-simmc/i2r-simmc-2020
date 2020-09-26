import json
from collections import defaultdict
import argparse
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('--domain', type=str)
parser.add_argument('--output_dir', required=True, type=str)
parser.add_argument('--train_dir', default='../data/', type=str)
parser.add_argument('--testset', default='devtest', type=str)
parser.add_argument('--mode', default='devtest', type=str)
args = parser.parse_args()

def evaluate_response_retrieval(gt_responses, model_scores, single_round_eval=False):
    """Evaluates response retrieval using the raw data and model predictions.
    Args:
        gt_responses: Ground truth responses.
        model_scores: Scores assigned by the model to the candidates.
        single_round_eval: Evaluate only for the last turn.
    If in single round evaluation model (mostly for hidden test-std split),
    use hidden gt_index field. Else, 0th element is the ground truth for other
    splits.
    """
    gt_index_pool = {
        ii["dialogue_idx"]: ii for ii in gt_responses["retrieval_candidates"]
    }
    gt_ranks = []
    for model_datum in model_scores:
        dialog_id = model_datum["dialog_id"]
        gt_datum = gt_index_pool[dialog_id]["retrieval_candidates"]
        num_gt_rounds = len(gt_datum)
        for round_id, round_datum in enumerate(model_datum["candidate_scores"]):
            round_id = round_datum["turn_id"]
            # Skip if single_round_eval and this is not the last round.
            if single_round_eval and round_id != num_gt_rounds - 1:
                continue

            gt_index = gt_datum[round_id]["gt_index"]
            current_turn = round_datum["turn_id"]
            round_scores = round_datum["scores"]
            gt_score = round_scores[gt_index]
            gt_ranks.append(np.sum(np.array(round_scores) > gt_score) + 1)
    gt_ranks = np.array(gt_ranks)
    print("#Instances evaluated retrieval: {}".format(gt_ranks.size))

    return {
        "r1": np.mean(gt_ranks <= 1),
        "r5": np.mean(gt_ranks <= 5),
        "r10": np.mean(gt_ranks <= 10),
        "mean": np.mean(gt_ranks),
        "mrr": np.mean(1 / gt_ranks),
    }

def export_results(results):
    results_path = os.path.join(args.output_dir, '{}-subtask-2-retrieval-results.csv'.format(args.domain))                        
    if not os.path.isfile(results_path):
        with open(results_path, 'w') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)   
            filewriter.writerow(['Model', 'r@1', 'r@5', 'r@10', 'Mean Rank', 'MRR'])
            filewriter.writerow(['{}_{}'.format(args.architecture, args.poly_m), results['r1'], results['r5'], results['r10'], results['mean'], results['mrr']])
    else:
        with open(results_path, 'a') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)   
            filewriter.writerow(['{}_{}'.format(args.architecture, args.poly_m), results['r1'], results['r5'], results['r10'], results['mean'], results['mrr']])

scores_json_path = os.path.join(args.output_dir, 'dstc9-simmc-{}-{}-subtask-2-retrieval.json'.format(args.testset, args.domain))
gt_json_path = os.path.join(args.train_dir, '{}_{}_dials_retrieval_candidates.json'.format(args.domain, args.mode))

print("Reading: {}".format(scores_json_path))

with open(scores_json_path, "r") as infile:
    model_scores = json.load(infile)

print("Reading: {}".format(gt_json_path))

with open(gt_json_path, "r") as infile:
    gt_responses = json.load(infile)

if (args.testset=='teststd'):
    single_round_evaluation = True
else:
    single_round_evaluation = False
          
retrieval_metrics = evaluate_response_retrieval(gt_responses, model_scores, single_round_evaluation)
print(retrieval_metrics)

if (args.mode=='devtest'):
    export_results(retrieval_metrics)
    print('Devtest results exported to csv at {}'.format(os.path.join(args.output_dir, '{}-subtask-2-retrieval-results.csv'.format(args.domain))))
