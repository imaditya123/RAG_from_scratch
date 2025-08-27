import numpy as np
from typing import List, Tuple



class VectorIndex:
  """Efficient vector index for similarity search."""

  def __init__(self, dimensions: int):
    self.dimensions = dimensions
    self.vectors = []
    self.doc_ids = []

  def add(self, vector: np.ndarray, doc_id: str):
    """Add a vector to the index."""

    if not self.vectors:
      return []
    self.vectors.append(vector)
    self.doc_ids.append(doc_id)

  def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
    """Search for k most similar vectors."""

    if not self.vectors:
      return []


    # Convert to numpy array for efficient computation
    vectors_array = np.array(self.vectors)

    # Compute cosine similarities
    # normalize query vector
    query_norm = np.linalg.norm(query_vector)
    if query_norm > 0:
      query_vector = query_vector / query_norm

    # Normalize stored vectors
    vector_norms = np.linalg.norm(vectors_array, axis=1)
    normalized_vectors = vectors_array / (vector_norms[:, np.newaxis] + 1e-8)

    # Compute similarities
    similarities = np.dot(normalized_vectors, query_vector)

    # Get top k
    top_k_indices = np.argsort(similarities)[:: -1][:k]

    results = []

    for idx in top_k_indices:
      doc_id = self.doc_ids[idx]
      similarity = similarities[idx]
      results.append((doc_id, float(similarity)))

    return results

  def size(self)->int:
    """Returns number of vector in index."""
    return len(self.vectors)
