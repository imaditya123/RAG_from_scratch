
from rag_document import RagDocument
from vector_store import VectorStore


if __name__ == "__main__":
    # Create sample documents
    sample_docs = [
        RagDocument(
            id="doc1",
            content="Machine learning is a subset of artificial intelligence that focuses on algorithms.",
            metadata={"category": "AI", "author": "John Doe"}
        ),
        RagDocument(
            id="doc2", 
            content="Deep learning uses neural networks with multiple layers to learn complex patterns.",
            metadata={"category": "AI", "author": "Jane Smith"}
        ),
        RagDocument(
            id="doc3",
            content="Natural language processing helps computers understand and generate human language.",
            metadata={"category": "NLP", "author": "John Doe"}
        ),
        RagDocument(
            id="doc4",
            content="Computer vision enables machines to interpret and understand visual information.",
            metadata={"category": "CV", "author": "Alice Johnson"}
        ),
        RagDocument(
            id="doc5",
            content="Python is a popular programming language for data science and machine learning.",
            metadata={"category": "Programming", "author": "Bob Wilson"}
        )
    ]
    
    # Create and populate vector store
    print("Creating vector store...")
    vector_store = VectorStore(embedding_dim=128)
    vector_store.add_documents(sample_docs)
    
    print(f"Vector store stats: {vector_store.stats()}")
    
    # Test search
    print("\n=== Search Results ===")
    query = "neural networks and deep learning"
    results = vector_store.search(query, k=3)
    
    print(f"Query: '{query}'")
    for i, (doc, similarity) in enumerate(results, 1):
        print(f"{i}. [Score: {similarity:.3f}] {doc.content}")
        print(f"   Metadata: {doc.metadata}")
    
    # Test with metadata filtering
    print("\n=== Filtered Search Results ===")
    filtered_results = vector_store.search(
        query, 
        k=3, 
        filter_metadata={"author": "John Doe"}
    )
    
    print(f"Query: '{query}' (filtered by author: John Doe)")
    for i, (doc, similarity) in enumerate(filtered_results, 1):
        print(f"{i}. [Score: {similarity:.3f}] {doc.content}")
        print(f"   Metadata: {doc.metadata}")
    
    # Test save/load
    print("\n=== Testing Save/Load ===")
    vector_store.save("vector_store.pkl")
    
    # Create new vector store and load
    new_vector_store = VectorStore()
    new_vector_store.load("vector_store.pkl")
    
    print(f"Loaded vector store stats: {new_vector_store.stats()}")
    
    # Test search on loaded store
    loaded_results = new_vector_store.search("artificial intelligence", k=2)
    print("Search results from loaded vector store:")
    for i, (doc, similarity) in enumerate(loaded_results, 1):
        print(f"{i}. [Score: {similarity:.3f}] {doc.content}")