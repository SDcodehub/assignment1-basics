"""
Tokenizer class for tokenizing text
"""
import regex as re
import json
from typing import Dict, List, Tuple
from cs336_basics.utlis.logging_config import get_logger


log = get_logger()

# GPT 2 tokenizer pattern
# This regex splits the text into chunks of letters numbers or punctuations
# Its designed to keep the spaces attached to the words that follow them
split_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    """
    Tokenizer class for tokenizing text
    """
    def __init__(self, vocab:Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: List[str] = None):
        """
        Construct the Tokenizer
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        # Append any missing special tokens to the vocab with new IDs (UTF-8 bytes)
        if self.special_tokens:
            existing_values = set(self.vocab.values())
            next_id = (max(self.vocab.keys()) + 1) if self.vocab else 0
            for token in self.special_tokens:
                token_bytes = token.encode("utf-8")
                if token_bytes not in existing_values:
                    self.vocab[next_id] = token_bytes
                    existing_values.add(token_bytes)
                    next_id += 1

        if self.special_tokens:
            special_pattern = "|".join(re.escape(token) for token in self.special_tokens)
            self.special_token_pattern = re.compile(f"({special_pattern})")
        else:
            self.special_token_pattern = None


    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: List[str] = None):
        """
        class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges 
        (in the same format that your BPE training code output) and (optionally) a list of special 
        tokens. This method should accept the following additional parameters:
        """

        # load the vocabulary
        log.info(f"loading vocabulary from {vocab_filepath}")

        # This will load the file into a dictionary like:
        # {"256": [60, 124, 101, 110, 100, 111, 102, 116, 101, 120, 116, 124, 62], ...}
        with open(vocab_filepath, "r", encoding="utf-8") as vocab_file:
            vocab = json.load(vocab_file)

        # Show only the top 5 and bottom 5 items in the vocab for debug logging
        log.debug(f"top 5 vocab items are{vocab[:5]}")
        log.debug(f"bottom 5 vocab items are{vocab[-5:]}")

        # now lets convert the read file into the format expect in init dict{int: bytes}
        parsed_vocab = {int(token_id): bytes(byte_list) for token_id, byte_list in vocab.items()}

        log.debug(f"top 5 parsed vocab items are{parsed_vocab[:5]}")
        log.debug(f"bottom 5 parsed vocab items are{parsed_vocab[-5:]}")

        # lets start working on the merges
        log.info(f"loading merges from {merges_filepath}")
        with open(merges_filepath, "r", encoding="utf-8") as merges_files:
            merges = [tuple(bytes.fromhex(h) for h in line.split()) for line in merges_files if line.split()]
        
        log.debug(f"top 5 merges are{merges[:5]}")
        log.debug(f"bottom 5 merges are{merges[-5:]}")

        return cls(parsed_vocab, merges, special_tokens)

    def encode(self, text: str) -> List[int]:
        """
        Encode a text string into a list of integers
        """
        pass

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that 
        lazily yields token IDs. This is required for memory-eﬀicient tokenization of large 
        files that we cannot directly load into memory.
        """
        pass

    def decode(self, ids: List[int]) -> str:
        """
        Decode a list of integers into a text string
        """
        pass
