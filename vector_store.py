import pickle
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from rag_document import RagDocument
from simple_embedder import SimpleEmbedder
from vector_index import VectorIndex


class VectorStore:
  """Complete vector store implementation."""

  def __init__(self, embedding_dim: int=384):
    self.embedding_dim = embedding_dim
    self.embedder = SimpleEmbedder(embedding_dim=embedding_dim)
    self.index = VectorIndex(dimensions=embedding_dim)
    self.documents = {}
    self.is_fitted = False

  def add_documents(self,documents: List[RagDocument]):
    """Add documents to the vector store"""

    if not self.is_fitted:
      # Fit embedder on all document content
      all_texts = [doc.content for doc in documents]
      self.embedder.fit(all_texts)
      self.is_fitted=True

    for doc in documents:
      # generate embedding 

      embedding = self.embedder.embed(doc.content)
      doc.embedding = embedding

      # Add to index
      self.index.add(embedding,doc.id)

      # store document

      self.documents[doc.id] = doc

  def search(self, query: str, k: int = 5, filter_metadata: Optional[Dict[str, Any]]=None )-> List[Tuple[RagDocument, float]]:
      """Search for similar documents."""
      if not self.is_fitted:
        raise ValueError("Vector store must have documents before searching.")

      # Generating query embedding 
      query_embedding = self.embedder.embed(query)

      # Search in index
      results = self.index.search(query_embedding, k = k*2)

      # Apply meta data filtering and return documents
      filtered_results = []

      for doc_id, similarity in results:
        doc = self.documents[doc_id]

        # Apply metadata filter id provided
        if filter_metadata:
          match = True

          for key, value in filter_metadata.items():
            if key not in doc.metadata or doc.metadata[key] != value:
              match = False
              break

          if not match:
            continue
        filtered_results.append((doc,similarity))

        if len(filtered_results) >= k:
          break

      return filtered_results


  def get_document(self, doc_id: str) -> Optional[RagDocument]:
    """Get document by ID."""
    return self.documents.get(doc_id)

  def delete_document(self,doc_id: str)-> bool:
    if doc_id in self.documents:
      del self.documents[doc_id]

      return True

    return False

  def save(self, filepath: str):
    """Save vector store to file."""
    data = {
        'embedding_dim': self.embedding_dim,
        'embedder': {
            'tokenizer': self.embedder.tokenizer.__dict__,
            'idf_scores': self.embedder.idf_scores,
            'embedding_matrix': self.embedder.embedding_matrix
        },
        'documents': {doc_id: {
            'id': doc.id,
            'content': doc.content,
            'metadata': doc.metadata,
            'embedding': doc.embedding.tolist() if doc.embedding is not None else None
        } for doc_id, doc in self.documents.items()},
        'index': {
            'vectors': [v.tolist() for v in self.index.vectors],
            'doc_ids': self.index.doc_ids
        }
    }
        
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    
  def load(self, filepath: str):
      """Load vector store from file."""
      with open(filepath, 'rb') as f:
          data = pickle.load(f)
      
      self.embedding_dim = data['embedding_dim']
      
      # Restore embedder
      self.embedder = SimpleEmbedder(embedding_dim=self.embedding_dim)
      self.embedder.tokenizer.__dict__.update(data['embedder']['tokenizer'])
      self.embedder.idf_scores = data['embedder']['idf_scores']
      self.embedder.embedding_matrix = np.array(data['embedder']['embedding_matrix']) if data['embedder']['embedding_matrix'] is not None else None
      self.embedder.is_fitted = True
        
      # Restore index
      self.index = VectorIndex(dimension=self.embedding_dim)
      self.index.vectors = [np.array(v) for v in data['index']['vectors']]
      self.index.doc_ids = data['index']['doc_ids']
        
      # Restore documents
      self.documents = {}
      for doc_id, doc_data in data['documents'].items():
          doc = RagDocument(
              id=doc_data['id'],
              content=doc_data['content'],
              metadata=doc_data['metadata'],
              embedding=np.array(doc_data['embedding']) if doc_data['embedding'] is not None else None
          )
          self.documents[doc_id] = doc
      
      self.is_fitted = True
    
  def stats(self) -> Dict[str, Any]:
      """Get statistics about the vector store."""
      return {
          'num_documents': len(self.documents),
          'embedding_dimension': self.embedding_dim,
          'index_size': self.index.size(),
          'vocabulary_size': len(self.embedder.tokenizer.word_to_id) if self.is_fitted else 0
      }
