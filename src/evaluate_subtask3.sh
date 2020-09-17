# Evaluate (furniture, multi-modal)
python evaluate_subtask3.py \
    --input_path_target=../data/simmc/simmc_furniture_fusion/furniture_devtest_dials_target.txt \
    --input_path_predicted=../output/simmc/furniture_devtest_subtask3.txt \
    --output_path_report=../output/simmc/furniture_devtest_subtask3_report.json

## Evaluate (Fashion, non-multimodal)
#python -m gpt2_dst.scripts.evaluate \
#    --input_path_target="${PATH_DIR}"/gpt2_dst/data/fashion_to/fashion_devtest_dials_target.txt \
#    --input_path_predicted="${PATH_DIR}"/gpt2_dst/results/fashion_to/fashion_devtest_dials_predicted.txt \
#    --output_path_report="${PATH_DIR}"/gpt2_dst/results/fashion_to/fashion_devtest_dials_report.json
#
## Evaluate (Fashion, multi-modal)
#python -m gpt2_dst.scripts.evaluate \
#    --input_path_target="${PATH_DIR}"/gpt2_dst/data/fashion/fashion_devtest_dials_target.txt \
#    --input_path_predicted="${PATH_DIR}"/gpt2_dst/results/fashion/fashion_devtest_dials_predicted.txt \
#    --output_path_report="${PATH_DIR}"/gpt2_dst/results/f