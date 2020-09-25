#!/usr/bin/env python3
"""
    Scripts for converting the main SIMMC datasets (.JSON format)
    into the line-by-line stringified format (and back).

    The reformatted data is used as input for the GPT-2 based
    DST model baseline.
"""
from utils.convert import convert_json_to_flattened
import argparse

if __name__ == '__main__':
    # Parse input args
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path_json',
                        help='input path to the original dialog data')
    parser.add_argument('--output_path_predict',
                        help='output path for model input')
    parser.add_argument('--output_path_target',
                        help='output path for full target')
    parser.add_argument('--len_context',
                        help='# of turns to include as dialog context',
                        type=int, default=2)
    parser.add_argument('--use_multimodal_contexts',
                        help='determine whether to use the multimodal contexts each turn',
                        type=int, default=1)
    parser.add_argument('--test_data',
                        action='store_true',
                        help='used for test data (no labels)')

    args = parser.parse_args()
    input_path_json = args.input_path_json
    output_path_predict = args.output_path_predict
    output_path_target = args.output_path_target
    len_context = args.len_context
    use_multimodal_contexts = bool(args.use_multimodal_contexts)

    # Convert the data into GPT-2 friendly format
    convert_json_to_flattened(
        input_path_json,
        output_path_predict,
        output_path_target,
        len_context=len_context,
        use_multimodal_contexts=use_multimodal_contexts,
        test_data=args.test_data)
