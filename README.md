Codes submitted to SIMMC challenge (https://github.com/facebookresearch/simmc), a track of DSTC 9 (https://dstc9.dstc.community/home)

# Installation
- $ git clone https://github.com/i2r-simmc/i2r-simmc-2020.git && cd i2r-simmc-2020
- $ pip install -r requirements.txt
- $ mkdir model && mkdir model/fashion && mkdir model/furniture
- $ mkdir output && mkdir output/fashion && mkdir output/furniture

# Data pre-processing
- Place data files under data/simmc_fasion,furniture folders
- The testing data files should follow the same naming convention of training, dev and devtest dataset.
- Place the predefined action meta info from the official simmc repo to the data/simmc_fashion and data/simmc_furniture/ respectively
    - $ cp simmc/mm_action_prediction/models/fashion_model_metainfo.json data/simmc_fashion/
    - $ cp simmc/mm_action_prediction/models/furniture_model_metainfo.json data/simmc_furniture/
- $ cd src/
- $ bash preprocess.sh

# Training
## Training model for fashion and furniture dataset:
- $ cd src/
- Train model with specific domain, domain can be `fashion` and `furniture` 
- $ bash train.sh \<domain\>
- Optionally, you can train fashion and furniture dataset with specified setting including gpu_id, learning_rate and batch_size
- $ bash train.sh \<domain\> <gpu_id> <learning_rate> <batch_size>
- Eg: $ bash train.sh fashion 0 1e-5 3
- The default learning_rate is 1e-5, default batch size is 3, if you encounter CUDA memory issue, please reduce batch size to 2 or 1.

# Generation
## Generating output for fashion and furniture dataset:
- $ cd src/
- Generate model output with specific domain, domain can be `fashion` and `furniture` 
- $ bash generate.sh \<domain\>
- Optionally, you can generate with specified setting including gpu_id, testing batch size and testing split name
- Testing split name can be `devtest` or `test` based on the file you want to test. 
- $ bash generate.sh \<domain\> <gpu_id> <learning_rate> <testing_split_name>
- Eg: $ bash generate.sh fashion 0 20 devtest
- The default testing batch size is 20, if you encounter CUDA memory issue, please reduce testing batch size.
- After the generation, the generated files for subtask #1,#2,#3 can be found in `output/<domain>/output_subtask_1.json`,`output/<domain>/output_subtask_2.json`,`output/<domain>/output_subtask_3.json` respectively.

# Evaluation
- First go to the src folder
- $ cd src
## Testing for all subtasks
- Evaluate subtask#1,#2,#3 with specific domain
- $ bash evaluate_all.sh \<domain\> <test_split_name>
- Eg: $ bash evaluate_all.sh fashion devtest
- The final report can be retrieved from `output/<domain>/prediction_report.csv`

## (Optionally) Evaluation for subtasks separately
## Testing for SubTask #1
- Evaluation for subtask#1 with the official SIMMC script with specific domain, domain can be `fashion` and `furniture`, `test_split_name` can be `devtest` or `test-std`
- $ cd src/
- $ bash evaluate_subtask1.sh \<domain\> <test_split_name>
- Eg: $ bash evaluate_subtask1.sh fashion devtest
- The results can be retrieved from `output/<domain>/output_subtask1.json`

## Testing for SubTask #2
- Evaluation for subtask#2 with the official SIMMC script with specific domain, domain can be `fashion` and `furniture`, `test_split_name` can be `devtest` or `test-std`
- $ cd src/
- $ bash evaluate_subtask2.sh \<domain\> <test_split_name>
- Eg: $ bash evaluate_subtask2.sh fashion devtest
- The results can be retrieved from `output/<domain>/output_subtask2.json`

## Testing for SubTask #3
- Evaluation for subtask#3 with the official SIMMC script with specific domain, domain can be `fashion` and `furniture`, `test_split_name` can be `devtest` or `test`
- $ cd src/
- $ bash evaluate_subtask3.sh \<domain\> <test_split_name>
- Eg: $ bash evaluate_subtask3.sh fashion devtest
- The results can be retrieved from `output/<domain>/output_subtask3.json`
