Codes submitted to SIMMC challenge (https://github.com/facebookresearch/simmc), a track of DSTC 9 (https://dstc9.dstc.community/home)

# Overview
We developed an end-to-end encoder-decoder model based on BART (Lewis et al., 2020) for generating outputs of the tasks (Sub-Task #1, Sub-Task #2 Response, Sub-Task #3) in a single string, called joint learning model, and another model based on Poly-Encoder (Humeau et al., 2019) for generating outputs of the Sub-Task #2 Retrieval task, called retrieval model. The retrieval model utilizes the BART encoder fine-tuned by the joint learning model. The two models are trained and evaluated separately.

# Installation 
- $ git clone https://github.com/i2r-simmc/i2r-simmc-2020.git && cd i2r-simmc-2020
- Place SIMMC data files under data/simmc_fasion,furniture folders
	- $ git lfs install
	- $ git clone https://github.com/facebookresearch/simmc.git
	- $ cp -R simmc/data .
	- $ cp simmc/mm_action_prediction/models/fashion_model_metainfo.json data/simmc_fashion/
	- $ cp simmc/mm_action_prediction/models/furniture_model_metainfo.json data/simmc_furniture/
- $ mkdir -p model/fashion && mkdir model/furniture
	- Model files are saved at model/\<domain\>/<model_type>/best_model/
		- \<domain\> is either `fashion` or `furniture`
		- <model_type>: `bart-large`, `bart-base`, `poly-encoder`, or `bi-encoder`
- $ mkdir -p output/fashion && mkdir output/furniture
	- Output JSON files are stored at output/\<domain\>/<model_type_combined>/\<dataset\>/dstc9-simmc-\<dataset\>-\<domain\>-\<task\>.json
		- <model_type_combined>: `bart-large_poly-encoder`, `bart-large_ bi-encoder`, `bart-base_poly-encoder`, `bart-base_bi-encoder`
		- \<dataset\>: devtest, test-std
		- \<task\>: subtask-1, subtask-2-generation, subtask-2-retrieval, subtask-3
	- For `devtest` dataset, the performance reports are stored at output/\<domain\>/<model_type_combined>/\<dataset\>/report.joint-learning.csv,report.retrieval.csv

# Installation
- $ cd src
- $ pip install -r requirements.txt

# Joint learning
## Data pre-processing 
- $ cd src
- $ bash preprocess.sh
	- We assume that the files of the `test-std` set are named e.g. fashion_test-std_dials.json. If not, lines 11, 37, 43, 50 and 59 of preprocess.sh and the `test_split_name` in the evaluation script below should be changed accordingly.

## Training 
- $ cd src
- $ bash train.sh \<domain\>
	- \<domain\> is either `fashion` or `furniture`
	- Optionally, you can train `fashion` and `furniture` dataset with specified setting including gpu_id, learning_rate and batch_size
		- $ bash train.sh \<domain\> <gpu_id> <learning_rate> <batch_size>
		- e.g. $ bash train.sh fashion 0 1e-5 3
		- The default learning_rate is 1e-5, default batch size is 3, if you encounter CUDA memory issue, please reduce batch size to 2 or 1.

## Generation 
- Generate trained model outputs for Sub-Task #1, Sub-Task #2 Generation and Sub-Task #3 together with specific domain
- $ cd src/
- $ bash generate.sh \<domain\> <test_split_name>
	- e.g. $ bash generate.sh fashion
	- Optionally, you can generate with specified setting including gpu_id, testing batch size and testing split name
	- Testing split name can be `devtest` or `test-std` based on the file you want to test.
	- $ bash generate.sh \<domain\> <test_split_name> <gpu_id> <test_batch_size>
	- e.g. $ bash generate.sh fashion 0 20 devtest
	- The default testing batch size is 20, if you encounter CUDA memory issue, please reduce testing batch size.
- The generation output files of `devtest` dataset for subtasks #1,#2 (generation),#3 can be found at the followings:
	- output/\<domain\>/<combined_model_name>/<test_split_name>/dstc9-simmc-devtest-fashion-subtask-1.json
	- output/\<domain\>/<combined_model_name>/<test_split_name>/dstc9-simmc-devtest-fashion-subtask-2-generation.json
	- output/\<domain\>/<combined_model_name>/<test_split_name>/dstc9-simmc-devtest-fashion-subtask-3.json, respectively.

# Retrieval
## Data pre-processing 
- Edit src/preprocess_retrieval.sh ($TESTSET=`devtest` or `test-std`)
- $ bash src/preprocess_retrieval.sh 
	- We assume that the files of the `test-std` set are named e.g. fashion_test-std_dials.json. If not, lines 18 and 34 of preprocess_retrieval.sh should be changed accordingly.

## Training 
- Edit src/retrieval/train_all_models.sh ($DOMAIN=`fashion` or `furniture`)
- $ cd src/retrieval
- $ bash train_all_models.sh

## Generation

# Evaluation
## Evaluation (Joint learning)
- Evaluate Sub-Task #1, Sub-Task #2 Generation and Sub-Task #3 together with specific domain
- $ cd src/
- $ bash evaluate_all.sh \<domain\> <test_split_name>
	- e.g. $ bash evaluate_all.sh fashion devtest
- The performance report for the non-retrieval tasks can be found at output/\<domain\>/<combined_model_name>/<test_split_name>/report.joint-learning.csv

## (Optionally) Evaluation for subtasks individually (Joint learning)
### Testing for Sub-Task #1
- Evaluation for subtask#1 with the official SIMMC script with specific domain, domain can be `fashion` and `furniture`, `test_split_name` can be `devtest` or `test-std`
- $ cd src/
- $ bash evaluate_subtask1.sh \<domain\> <test_split_name>
- Eg: $ bash evaluate_subtask1.sh fashion devtest
- The results can be retrieved from `output/\<domain\>/<combined_model_name>/<test_split_name>/dstc9-simmc-devtest-fashion-subtask-1-report.json`

### Testing for Sub-Task #2 Generation
- Evaluation for subtask#2 generation with the official SIMMC script with specific domain, domain can be `fashion` and `furniture`, `test_split_name` can be `devtest` or `test-std`
- $ cd src/
- $ bash evaluate_subtask2.sh \<domain\> <test_split_name>
- Eg: $ bash evaluate_subtask2.sh fashion devtest
- The results can be retrieved from `output/\<domain\>/<combined_model_name>/<test_split_name>/dstc9-simmc-devtest-fashion-subtask-2-generation-report.json`

### Testing for Sub-Task #3
- Evaluation for subtask#3 with the official SIMMC script with specific domain, domain can be `fashion` and `furniture`, `test_split_name` can be `devtest` or `test-std`
- $ cd src/
- $ bash evaluate_subtask3.sh \<domain\> <test_split_name>
- Eg: $ bash evaluate_subtask3.sh fashion devtest
- The results can be retrieved from `output/\<domain\>/<combined_model_name>/<test_split_name>/dstc9-simmc-devtest-fashion-subtask-3-report.json`

## Evaluation (Retrieval)
- Edit src/retrieval/evaluate_all_models.sh ($DOMAIN=`fashion` or `furniture`, $TESTSET=`devtest` or `test-std`)
- $ cd src/retrieval
- $ bash evaluate_all_models.sh

## Evaluation outputs
- The output JSON files can be found at output/\<domain\>/outputs.\<TESTSET\>.json ($TESTSET=`devtest` or `test-std`)
- `devtest`: The performance results can be found at output/\<domain\>/reports.\<TESTSET\>.joint.csv,reports.\<TESTSET\>.retrieval.csv ($TESTSET=`devtest` or `test-std`)

# References
- Lewis, M., Liu, Y., Goyal, N., Ghazvininejad, M., Mohamed, A., Levy, O., … Zettlemoyer, L. (2020). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension. In ACL. Retrieved from http://arxiv.org/abs/1910.13461
- Humeau, S., Shuster, K., Lachaux, M.-A., & Weston, J. (2019). Poly-encoders: Transformer Architectures and Pre-training Strategies for Fast and Accurate Multi-sentence Scoring. Retrieved from http://arxiv.org/abs/1905.01969
