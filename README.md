Codes submitted to SIMMC challenge (https://github.com/facebookresearch/simmc), a track of DSTC 9 (https://dstc9.dstc.community/home)

# Installation
- $ git clone https://github.com/i2r-simmc/i2r-simmc-2020.git && cd i2r-simmc-2020
- $ git lfs pull
- $ pip install -r requirements.txt
- $ mkdir model && mkdir model/fashion && mkdir model/furniture
- $ mkdir output && mkdir output/fashion && mkdir output/furniture

# Data pre-processing
- Place data files under data/simmc_fasion,furniture folders
- Edit preprocessing.sh ($DOMAIN='fashion' or 'furniture')
- $ bash preprocessing.sh 

# Training and evaluation
- Edit train_all_models.sh and evaluate_all_models.sh ($DOMAIN='fashion' or 'furniture', $TESTSET='devtest' or 'test-std')
- $ bash train_all_models.sh
- $ bash evaluate_all_models.sh
- The output JSON files are stored under data/simmc_fasion,furniture folders
