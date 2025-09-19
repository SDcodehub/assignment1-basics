## Usage

Run the BPE preprocessing sample with logging from the repository root.

### Default run (INFO level)
```bash
cd assignment1-basics
uv run python -u cs336_basics/train_bpe.py | cat
```

### Debug run (verbose top‑K details)
```bash
cd assignment1-basics
LOG_LEVEL=DEBUG uv run python -u cs336_basics/train_bpe.py | cat
```

Notes:
- INFO shows high‑level counts. Set `LOG_LEVEL=DEBUG` to include top‑K splits and pair stats.
- Outputs go to stdout; no files are written during normal debugging.

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
```


### Benchmark (timing)
You can use [`hyperfine`](https://github.com/sharkdp/hyperfine) to benchmark the runtime of the BPE training script.
cd assignment1-basics
```bash
hyperfine --warmup 1 --runs 2 --show-output 'uv run -- python -u cs336_basics/train_bpe.py'
```

Note the commit id using following command
```bash
git rev-parse --short HEAD
```

### Performance results
Dataset used is `TinyStoriesV2-GPT4-train.txt`

| Commit  | Runs | Warmup | Mean ± σ              | Range (min…max)           | Command                                       |
|---------|------|--------|-----------------------|---------------------------|-----------------------------------------------|
| 00312fb | 2    | 1      | 166.923 s ± 0.600 s   | 166.498 s … 167.347 s     | `uv run -- python -u cs336_basics/train_bpe.py` |

#### TODO Using above setup think of optimising the train_BPE code
