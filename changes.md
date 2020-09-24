# Changes

This repo the following changes to the original huggingface transformers (repo: https://github.com/huggingface/transformers).

- Added a new class model called BartLMHeadModel in replace of original BartForConditionalGeneration to be our main model.
- Made some changes in the `transformers/generation_utils.py` to support action perplexity calculation.