# Changes

This repo the following changes to the original huggingface transformers (repo: https://github.com/huggingface/transformers).

- Added a new class model called BartLMHeadModel in replace of original BartForConditionalGeneration to be our main model.
- Made some changes in the `transformers/generation_utils.py` to support action perplexity calculation.

The following changes were made to the origin Poly-Encoder code (repo: https://github.com/chijames/Poly-Encoder).

- Modified parse.py for different data preprocessing
- Modified run.py to support Bart and output results
- Added evaluation scripts
