# Step 1 preprocess API call data for subtask#1
for DOMAIN in "furniture" "fashion"
do
  # DOMAIN="fashion"
  ROOT="../data/simmc_${DOMAIN}/"

  # Input files.
  TRAIN_JSON_FILE="${ROOT}${DOMAIN}_train_dials.json"
  DEV_JSON_FILE="${ROOT}${DOMAIN}_dev_dials.json"
  DEVTEST_JSON_FILE="${ROOT}${DOMAIN}_devtest_dials.json"
  FURNITURE_METADATA_FILE="${ROOT}furniture_metadata.csv"
  FASHION_METADATA_FILE="${ROOT}fashion_metadata.json"

  # Step 1: Extract assistant API.
  INPUT_FILES="${TRAIN_JSON_FILE} ${DEV_JSON_FILE} ${DEVTEST_JSON_FILE}"
  if [ $DOMAIN == "furniture" ]
  then
    python tools/extract_actions.py \
        --json_path="${INPUT_FILES}" \
        --save_root="${ROOT}" \
        --metadata_path="${FURNITURE_METADATA_FILE}"
  else
    python tools/extract_actions_fashion.py \
        --json_path="${INPUT_FILES}" \
        --save_root="${ROOT}" \
        --metadata_path="${FASHION_METADATA_FILE}"
  fi
done

# Step 2 preprocess data for subtask #1
## Preprocess for subtask#1 (fashion and furniture, multi-modal)
python preprocess.py \
    --data_path=../data \
    --use_multimodal_contexts \
    --use_action_prediction

## Preprocess for subtask#2 (fashion and furniture, multi-modal)
python preprocess.py \
    --data_path=../data \
    --use_multimodal_contexts

## Preprocess for subtask#3 (fashion and furniture, multi-modal)
python preprocess.py \
    --data_path=../data \
    --use_multimodal_contexts \
    --generate_belief_state

# Step 3 flatten subtask #3 and flatten input
# Fashion
# Multimodal Data
# Train split
for DOMAIN in "furniture" "fashion"
do
  ROOT="../data/simmc_${DOMAIN}/"
  for SPLIT in "train" "dev" "devtest"
  do
  python -m preprocess_input \
      --input_path_json="${ROOT}""$DOMAIN"_"$SPLIT"_dials.json \
      --output_path_predict="${ROOT}""$DOMAIN"_"$SPLIT"_dials_predict.txt \
      --output_path_target="${ROOT}""$DOMAIN"_"$SPLIT"_dials_target.txt \
      --len_context=2 \
      --use_multimodal_contexts=1
  done
done

for DOMAIN in "furniture" "fashion"
do
  ROOT="../data/simmc_${DOMAIN}/"
  for SPLIT in "teststd"
  do
  python -m preprocess_input \
      --input_path_json="${ROOT}""$DOMAIN"_"$SPLIT"_dials.json \
      --output_path_predict="${ROOT}""$DOMAIN"_"$SPLIT"_dials_predict.txt \
      --output_path_target="${ROOT}""$DOMAIN"_"$SPLIT"_dials_target.txt \
      --len_context=2 \
      --use_multimodal_contexts=1 \
      --test_data
  done
done
