"""
Tokenizer class for tokenizing text
"""
import regex as re
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
        pass

    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: List[str] = None):
        """
        returns tokenizer from seialised vocabulary and merges files
        """
        pass

    def encode(self, text: str) -> List[int]:
        """
        Encode a text string into a list of integers
        """
        pass

    def decode(self, ids: List[int]) -> str:
        """
        Decode a list of integers into a text string
        """
        pass
