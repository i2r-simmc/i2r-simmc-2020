import argparse
import json

import nltk
import numpy as np


################################################################
# from official script
################################################################
def normalize_sentence(sentence):
    """Normalize the sentences and tokenize.
    """
    return nltk.tokenize.word_tokenize(sentence.lower())


def evaluate_response_generation(gt_responses, model_responses):
    """Evaluates response generation using the raw data and model predictions.
    """
    gt_responses_pool = {
        ii["dialogue_idx"]: ii for ii in gt_responses["dialogue_data"]
    }
    bleu_scores = []
    # Smoothing function.
    chencherry = nltk.translate.bleu_score.SmoothingFunction()
    for model_datum in model_responses:
        dialog_id = model_datum["dialog_id"]
        for round_id, round_datum in enumerate(model_datum["predictions"]):
            response = round_datum["response"]
            gt_datum = gt_responses_pool[dialog_id]["dialogue"][round_id]
            gt_response = gt_datum["system_transcript"]

            bleu_score = nltk.translate.bleu_score.sentence_bleu(
                [normalize_sentence(gt_response)],
                normalize_sentence(response),
                smoothing_function=chencherry.method1
            )
            bleu_scores.append(bleu_score)
    return np.mean(bleu_scores)


################################################################

def read_predictions(filename):
    predictions = []
    with open(filename, 'r') as f:
        dialogs = json.load(f)["dialogs"]

    for d in dialogs:
        for turn in d["dialog"]:
            predictions.append(turn["answer"])

    return predictions


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


def compute_metrics(input_path_target, input_path_predicted):
    with open(input_path_target, 'r') as f:
        gt_data = json.load(f)

    model_responses = []
    predictions = read_predictions(input_path_predicted)

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