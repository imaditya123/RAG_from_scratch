import numpy as np
from typing import List
import re
from collections import Counter

class Simpletokenizer:
  """Basic tokenizer for text processing. """
  def __init__(self,vocab_size: int=10000):
    self.vocab_size = vocab_size
    self.word_to_id = {}
    self.id_to_word = {}
    self.word_freq = Counter()
    self.is_fitted= False

  def _preprocess(self,text: str) ->List[str]:
    """Basic text preprocessing"""

    text = text.lower()

    text = re.sub(r'[^\w\s]',' ',text)
    tokens = text.split()

    return tokens

  def fit(self,texts : List[str]):
    """Build vocabulary from text"""

    all_tokens= []

    for text in texts:
      tokens = self._preprocess(text)
      all_tokens.extend(tokens)
      self.word_freq.update(tokens)
    # Get most frequest words ..... -2 for UNK and PAD
    most_common = self.word_freq.most_common(self.vocab_size-2)

    #build vocab

    self.word_to_id = {'<PAD>':0,'UNK':1}
    self.id_to_word = {0: '<PAD>', 1: '<UNK>'}

    for i, (word, _) in enumerate(most_common,2):
      self.word_to_id[word]=i
      self.id_to_word[i]=word

    self.is_fitted=True


  def tokenize(self,text: str)->List[int]:
    """Convert text to token IDs."""

    if not self.is_fitted:
      raise ValueError("Tokenizer must be fitted before tokenizing")

    tokens=self._preprocess(text)
    return [self.word_to_id.get(token,1) for  token in tokens]
