import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import os
from typing import List, Dict, Any, Optional


# =========================
# 1. Load embedding model
# =========================
def load_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Load the sentence transformer model"""
    return SentenceTransformer(model_name)


# =========================
# 2. Create embeddings
# =========================
def create_embeddings(chunks: List[str], model: SentenceTransformer) -> np.ndarray:
    """
    Create embeddings for text chunks
    
    Args:
        chunks: List of text chunks
        model: Sentence transformer model
    
    Returns:
        numpy array of embeddings
    """
    if not chunks:
        raise ValueError("Chunks list cannot be empty")
    
    embeddings = model.encode(chunks, show_progress_bar=True)
    return np.array(embeddings).astype("float32")


# =========================
# 3. Build FAISS index
# =========================
def build_faiss_index(embeddings: np.ndarray, use_cosine: bool = True) -> faiss.Index:
    """
    Build FAISS index for similarity search
    
    Args:
        embeddings: numpy array of embeddings
        use_cosine: If True, use cosine similarity; if False, use L2 distance
    
    Returns:
        FAISS index
    """
    dimension = embeddings.shape[1]
    
    if use_cosine:
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(dimension)  # Inner Product = cosine after normalization
        print(f" Built cosine similarity index (dimension: {dimension})")
    else:
        index = faiss.IndexFlatL2(dimension)
        print(f" Built L2 distance index (dimension: {dimension})")
    
    index.add(embeddings)
    print(f" Added {embeddings.shape[0]} vectors to index")
    
    return index


# =========================
# 4. Search functions
# =========================
def search(query: str, index: faiss.Index, chunks: List[str], model: SentenceTransformer, 
           k: int = 3, use_cosine: bool = True) -> List[str]:
    """
    Search for similar chunks (returns only text)
    
    Args:
        query: Search query
        index: FAISS index
        chunks: Original text chunks
        model: Sentence transformer model
        k: Number of results to return
        use_cosine: Whether cosine similarity was used
    
    Returns:
        List of top-k text chunks
    """
    results = search_with_scores(query, index, chunks, model, k, use_cosine)
    return [r["chunk"] for r in results]


def search_with_scores(query: str, index: faiss.Index, chunks: List[str], model: SentenceTransformer,
                      k: int = 3, use_cosine: bool = True) -> List[Dict[str, Any]]:
    """
    Search for similar chunks with relevance scores
    
    Args:
        query: Search query
        index: FAISS index
        chunks: Original text chunks
        model: Sentence transformer model
        k: Number of results to return
        use_cosine: Whether cosine similarity was used
    
    Returns:
        List of dictionaries with 'chunk', 'score', and 'index'
    """
    if k > len(chunks):
        k = len(chunks)
        print(f"⚠️ Warning: k reduced to {k} (number of available chunks)")
    
    # Encode query
    query_vec = model.encode([query]).astype("float32")
    
    # Normalize query if using cosine similarity
    if use_cosine:
        faiss.normalize_L2(query_vec)
    
    # Search
    distances, indices = index.search(query_vec, k)
    
    # Process results
    results = []
    for i, idx in enumerate(indices[0]):
        if idx != -1:  # FAISS returns -1 for invalid indices
            score = float(distances[0][i])
            
            # Convert score based on metric
            if use_cosine:
                # Cosine similarity: -1 to 1, higher is better
                score = score  # Keep as is (already cosine)
            else:
                # L2 distance: lower is better, convert to similarity
                score = 1.0 / (1.0 + score)  # Convert to 0-1 similarity
            
            results.append({
                "chunk": chunks[idx],
                "score": score,
                "index": int(idx)
            })
    
    return results


# =========================
# 5. Save and load functions
# =========================
def save_index(index: faiss.Index, chunks: List[str], filepath: str = "faiss_index"):
    """
    Save FAISS index and chunks to disk
    
    Args:
        index: FAISS index
        chunks: Original text chunks
        filepath: Base filepath (without extension)
    """
    # Save FAISS index
    faiss.write_index(index, f"{filepath}.bin")
    
    # Save chunks and metadata
    metadata = {
        "chunks": chunks,
        "index_type": type(index).__name__,
        "num_vectors": index.ntotal
    }
    
    with open(f"{filepath}.pkl", "wb") as f:
        pickle.dump(metadata, f)
    
    print(f"✅ Saved index to {filepath}.bin")
    print(f"✅ Saved metadata to {filepath}.pkl")


def load_index(filepath: str = "faiss_index") -> tuple:
    """
    Load FAISS index and chunks from disk
    
    Args:
        filepath: Base filepath (without extension)
    
    Returns:
        Tuple of (index, chunks)
    """
    # Load FAISS index
    if not os.path.exists(f"{filepath}.bin"):
        raise FileNotFoundError(f"Index file {filepath}.bin not found")
    
    index = faiss.read_index(f"{filepath}.bin")
    
    # Load metadata
    if not os.path.exists(f"{filepath}.pkl"):
        raise FileNotFoundError(f"Metadata file {filepath}.pkl not found")
    
    with open(f"{filepath}.pkl", "rb") as f:
        metadata = pickle.load(f)
    
    chunks = metadata["chunks"]
    
    print(f"✅ Loaded index with {index.ntotal} vectors")
    print(f"✅ Loaded {len(chunks)} chunks")
    
    return index, chunks


# =========================
# 6. Vector store class (optional OOP approach)
# =========================
class VectorStore:
    """Complete vector store with FAISS backend"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_cosine: bool = True):
        self.model = load_model(model_name)
        self.use_cosine = use_cosine
        self.index = None
        self.chunks = None
    
    def add_documents(self, chunks: List[str]):
        """Add documents to the vector store"""
        self.chunks = chunks
        embeddings = create_embeddings(chunks, self.model)
        self.index = build_faiss_index(embeddings, self.use_cosine)
        print(f"✅ Added {len(chunks)} documents to vector store")
    
    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        if self.index is None:
            raise ValueError("No documents in vector store. Call add_documents() first.")
        
        return search_with_scores(query, self.index, self.chunks, self.model, k, self.use_cosine)
    
    def save(self, filepath: str = "faiss_index"):
        """Save vector store to disk"""
        if self.index is None:
            raise ValueError("No index to save")
        save_index(self.index, self.chunks, filepath)
    
    def load(self, filepath: str = "faiss_index"):
        """Load vector store from disk"""
        self.index, self.chunks = load_index(filepath)
        print(f"✅ Loaded vector store with {len(self.chunks)} documents")


# =========================
# TEST PIPELINE
# =========================
if __name__ == "__main__":
    
    # Example chunks (replace with your actual data)
    chunks = [
        "Machine learning is about models that learn from data.",
        "Overfitting happens when a model memorizes the training data instead of learning patterns.",
        "Neural networks are computational systems inspired by the human brain.",
        "Deep learning uses multiple layers to learn hierarchical representations.",
        "Cross-validation helps prevent overfitting by validating on unseen data.",
        "Gradient descent is an optimization algorithm used to train neural networks."
    ]
    
    print("=" * 50)
    print(" INITIALIZING VECTOR STORE")
    print("=" * 50)
    
    # Initialize vector store
    vector_store = VectorStore(use_cosine=True)
    
    # Add documents
    print("\n Adding documents...")
    vector_store.add_documents(chunks)
    
    # Search examples
    print("\n" + "=" * 50)
    print(" SEARCH EXAMPLES")
    print("=" * 50)
    
    queries = [
        "What is overfitting?",
        "How do neural networks work?",
        "What optimization algorithm is used for training?"
    ]
    
    for query in queries:
        print(f"\n Query: '{query}'")
        print("-" * 40)
        
        results = vector_store.search(query, k=2)
        
        for i, result in enumerate(results, 1):
            print(f"{i}. [Score: {result['score']:.4f}] {result['chunk']}")
    
    # Save the index
    print("\n" + "=" * 50)
    print(" SAVING VECTOR STORE")
    print("=" * 50)
    vector_store.save("my_faiss_index")
    
    # Demonstrate loading
    print("\n" + "=" * 50)
    print(" LOADING VECTOR STORE FROM DISK")
    print("=" * 50)
    new_vector_store = VectorStore(use_cosine=True)
    new_vector_store.load("my_faiss_index")
    
    # Test loaded store
    print("\n Testing loaded store...")
    results = new_vector_store.search("overfitting", k=1)
    print(f"Result: {results[0]['chunk']}")
    
    print("\n" + "=" * 50)
    print(" VECTOR STORE READY FOR PRODUCTION")
    print("=" * 50)