"""
Train a BPE model on a text file
"""
import os
import logging
from heapq import nlargest
import regex as re




# GPT 2 tokenizer pattern
# This regex splits the text into chunks of letters numbers or punctuactions
# Its designed to keep the spaces attached to the words that follow them
split_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def pretokenise_text(input_path):
    with open(input_path, "r", encoding="utf-8") as read_file:
        text = read_file.read()
   
    pre_tokens = re.findall(split_pattern, text)
    # we will convert the pre tokens into the counts
    word_count = {}
    for word in pre_tokens:
        word_count[word] = word_count.get(word, 0) + 1

    # BPE generally works on the byte sequences to converting the strings into byte sequences
    splits = {word.encode("utf-8"): count for word, count in word_count.items()}
    return splits

def initialise_vocab(special_tokens):
    # vocab is a mapping from the integer ID to the byte sequence
    vocab = {i: bytes([i]) for i in range(256)}

    #add special tokens
    next_id = 256
    for token in special_tokens:
        vocab[next_id] = token.encode("utf-8")
        next_id += 1
    
    return vocab

def get_stats(splits):
    """
    Give n splits pre tokenised, return a dictionary of pairs of byte sequences and their counts
    """
    stats = {}
    for word_part, count in splits.items():
        # A word is represented as the byte sequences
        for i in range(len(word_part)-1):
            # form the pair of adjacent tokens
            pair = (word_part[i], word_part[i+1])
            # increment the count for the pair
            stats[pair] = stats.get(pair, 0) + count
    return stats

def merge_splits(splits, pair, new_token):
    """Replaces all the occuraces of pair in the splits with new_token"""
    new_splits = {}
    for words_parts, count in splits.items():
        new_words_parts = []
        i = 0
        while i < len(words_parts):
            if words_parts[i:i+2] == pair:
                new_words_parts.append(new_token)
                i += 2
            else:
                new_words_parts.append(words_parts[i])
                i += 1
        new_splits[tuple(new_words_parts)] = count
    return new_splits

def train_bpe(input_path, vocab_size, special_tokens):
    pass



def _get_logger() -> logging.Logger:
    logger_instance = logging.getLogger(__name__)
    if logger_instance.handlers:
        return logger_instance

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler()
    fmt = "%(asctime)s %(levelname)s:%(name)s:%(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"  # short date+time
    handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    logger_instance.addHandler(handler)
    logger_instance.setLevel(level)
    logger_instance.propagate = False
    return logger_instance
    



if __name__ == "__main__":
    log = _get_logger()
    # sample_text = "Hello, world! It's a test — with numbers 123 and spaces.  New line?\nYes!"
    # temp_path = "/tmp/pretokenise_sample.txt"
    # with open(temp_path, "w", encoding="utf-8") as f:
    #     f.write(sample_text)
    temp_path = "/Users/sagdesai/Desktop/work/building-transformer-lm/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt"
    raw_splits = pretokenise_text(temp_path)
    log.info("unique pretokenized byte-sequences: %d", len(raw_splits))
    result = {tuple(word): count for word, count in raw_splits.items()}

    # Debug-only: top-K splits
    if log.isEnabledFor(logging.DEBUG):
        top_splits = nlargest(10, result.items(), key=lambda kv: kv[1])
        for byte_seq, count in top_splits:
            hex_bytes = " ".join(f"{x:02x}" for x in byte_seq)
            log.debug("split %r [%s] -> %d", byte_seq, hex_bytes, count)

    vocab_map = initialise_vocab(["<|endoftext|>"])
    log.info("vocab size: %d", len(vocab_map))

    pair_stats = get_stats(result)
    log.info("unique adjacent-byte pairs: %d", len(pair_stats))

    # Debug-only: top-K pairs
    if log.isEnabledFor(logging.DEBUG):
        top_pairs = nlargest(20, pair_stats.items(), key=lambda kv: kv[1])
        for (a, b), count in top_pairs:
            log.debug("pair (%d,%d) [%02x %02x] -> %d", a, b, a, b, count)

