# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```


date 5 Oct 2025, folder structure
cs336_basics/
  __init__.py

  nn/
    __init__.py            # expose: Linear, Embedding, functional, etc.
    functional.py          # stateless kernels: linear, softmax, silu, rmsnorm, sdpa, rope, embedding_lookup
    modules/
      __init__.py
      linear.py            # class Linear
      embedding.py
      attention.py
      feedforward.py       # SwiGLU/FFN
      normalization.py     # RMSNorm

  models/
    __init__.py
    transformer_block.py   # pre-norm block wiring (MHA+RoPE, FFN, residuals)
    transformer_lm.py      # embeddings, blocks, final norm/head

  tokenizer/
    __init__.py            # expose: Tokenizer, train_bpe
    bpe/
      __init__.py
      core.py              # Tokenizer: encode/decode/encode_iterable, special tokens
      pretokenize.py       # GPT-2 regex + bytes mapping
      training.py          # train_bpe algorithm (pure functions)
      serialization.py     # load/save vocab & merges
      types.py             # small dataclasses / type aliases
    cli/
      __init__.py
      tokenize_dataset.py
      compute_bytes_per_token.py

  optim/
    __init__.py
    adamw.py
    schedulers.py          # cosine schedule with warmup

  data/
    __init__.py
    batching.py            # get_batch

  training/
    __init__.py
    checkpointing.py       # save_checkpoint / load_checkpoint

  utils/
    __init__.py
    logging_config.py