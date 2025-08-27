
import math
import numpy as np
from typing import List
import re
from collections import Counter
from simple_tokenizer import Simpletokenizer


class SimpleEmbedder:
  """Simple Embedding model using TF-IDF with dimentionality reduction."""

  def __init__(self,embedding_dim: int=384, max_features: int = 10000) -> None:
    self.embedding_dim = embedding_dim
    self.max_features = max_features
    self.tokenizer = Simpletokenizer(vocab_size=max_features)
    self.idf_scores={}
    self.embedding_matrix=None
    self.is_fitted = False

  def _compute_tf_idf(self, documents: List[str])->np.ndarray:
    """Compute TF-IDF matrix."""

    tokenized_docs = [self.tokenizer.tokenize(doc) for doc in documents]

    doc_freq = Counter()
    for tokens in tokenized_docs:
      unique_tokens = set(tokens)
      for token in unique_tokens:
        doc_freq[token] += 1

    num_docs = len(documents)
    for token_id in doc_freq:
      self.idf_scores[token_id] =  math.log(num_docs / doc_freq[token_id])

    tfid_matrix = []
    for tokens in tokenized_docs:
      token_counts = Counter(tokens)
      doc_length = len(tokens)

      tfidf_vector = np.zeros(self.tokenizer.vocab_size)
      for token_id, count in token_counts.items():
        tf = count /  doc_length
        idf = self.idf_scores.get(token_id, 0)
        tfidf_vector[token_id] = tf * idf


      tfid_matrix.append(tfidf_vector)


    return np.array(tfid_matrix)



  def _reduce_dimentions(self,tfidf_matrix : np.ndarray)-> np.ndarray:
    """Simple PCA-like dimentionanility reduction."""
    # Center the data
    mean = np.mean(tfidf_matrix, axis=0)
    centered_data = tfidf_matrix - mean

    # Compute covariance matrix
    cov_matrix = np.cov(centered_data.T)

    # eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort by eigenvalues (descending)
    idx = np.argsort(eigenvalues)[ ::-1]
    eigenvectors = eigenvectors[:, idx]

    # Take top components
    n_components = min(self.embedding_dim, eigenvectors.shape[1])
    self.embedding_matrix = eigenvectors[:, :n_components]

    # transform data
    reduced_data = centered_data @ self.embedding_matrix

    return reduced_data


  def fit(self, documents: List[str]):
    """Fit the embedder on documents. """

    self.tokenizer.fit(documents)
    tfidf_matrix = self._compute_tf_idf(documents)
    self._reduce_dimentions(tfidf_matrix)
    self.is_fitted=True


  def embed(self, text: str)-> np.ndarray:
    """Generating embedding for a single text."""

    if not self.is_fitted:
      raise ValueError("Embedder must be fitted before embedding")

    tokens = self.tokenizer.tokenize(text)
    token_counts = Counter(tokens)
    doc_length = len(tokens)


    tfidf_vector = np.zeros(self.tokenizer.vocab_size)
    for token_id, count in token_counts.items():
        tf = count /  doc_length
        idf = self.idf_scores.get(token_id, 0)
        tfidf_vector[token_id] = tf * idf

    # Reduce dimensions
    if self.embedding_matrix is not None:
      embedding = tfidf_vector @ self.embedding_matrix
      # Normalize

      norm = np.linalg.norm(embedding)
      if norm > 0:
        embedding = embedding / norm

      return embedding

    else :
      return tfidf_vector[:self.embedding_dim]
