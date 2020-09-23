Codes submitted to SIMMC challenge (https://github.com/facebookresearch/simmc), a track of DSTC 9 (https://dstc9.dstc.community/home)

# Installation
- $ cd src
- $ pip install -r requirements.txt

# Data pre-processing
- Place data files under data/simmc_fasion,furniture folders
- Edit preprocessing.sh ($DOMAIN='fashion' or 'furniture')
- $ cd ..
- $ bash src/preprocessing.sh 

# Training and evaluation
- Edit train_all_models.sh and evaluate_all_models.sh ($DOMAIN='fashion' or 'furniture', $TESTSET='devtest' or 'test-std')
- $ bash train_all_models.sh
- $ bash evaluate_all_models.sh
- The output JSON files are stored under data/simmc_fasion,furniture folders
