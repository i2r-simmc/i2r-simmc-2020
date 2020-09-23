Codes submitted to SIMMC challenge (https://github.com/facebookresearch/simmc), a track of DSTC 9 (https://dstc9.dstc.community/home)

# Overview
We developed an end-to-end encoder-decoder model based on BART (Lewis et al., 2020) for generating outputs of the tasks (Sub-Task #1, Sub-Task #2 Response, Sub-Task #3) in a single string, called joint learning model, and another model based on Poly-Encoder (Humeau et al., 2019) for generating outputs of the Sub-Task #2 Retrieval task, called retrieval model. The retrieval model utilizes the BART encoder fine-tuned by the joint learning model. The two models are trained and evaluated separately.

# Installation 
- $ git clone https://github.com/i2r-simmc/i2r-simmc-2020.git && cd i2r-simmc-2020
- Place data files under data/simmc_fasion,furniture folders
- $ mkdir model && mkdir model/fashion && mkdir model/furniture
- $ mkdir output && mkdir output/fashion && mkdir output/furniture
- The output JSON files are stored under src/output/fasion,furniture folders

# Installation (joint learning model)
- $ cd src
- $ pip install -r requirements.txt

# Installation (retrieval model)
- $ cd src/retrieval
- $ pip install -r requirements.txt

# Data pre-processing (joint learning model)
- $ cd src
- $ bash preprocess.sh

# Data pre-processing (retrieval model)
- Edit src/preprocess_retrieval.sh ($DOMAIN='fashion' or 'furniture')
- $ bash src/preprocess_retrieval.sh 

# Training (retrieval model)
- $ cd src
- $ bash train.sh

# Training (retrieval model)
- Edit src/retrieval/train_all_models.sh ($DOMAIN='fashion' or 'furniture')
- $ cd src/retrieval
- $ bash train_all_models.sh

# Evaluation (retrieval model)
- Edit src/retrieval/evaluate_all_models.sh ($DOMAIN='fashion' or 'furniture', $TESTSET='devtest' or 'test-std')
- $ cd src/retrieval
- $ bash evaluate_all_models.sh

# References
- Lewis, M., Liu, Y., Goyal, N., Ghazvininejad, M., Mohamed, A., Levy, O., … Zettlemoyer, L. (2020). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension. In ACL. Retrieved from http://arxiv.org/abs/1910.13461
- Humeau, S., Shuster, K., Lachaux, M.-A., & Weston, J. (2019). Poly-encoders: Transformer Architectures and Pre-training Strategies for Fast and Accurate Multi-sentence Scoring. Retrieved from http://arxiv.org/abs/1905.01969
