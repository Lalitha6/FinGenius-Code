import faiss
import numpy as np
from typing import List, Dict, Any
import logging
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)

class FAISSManager:
    def __init__(self, vector_dim: int = 768):
        self.vector_dim = vector_dim
        self.index = faiss.IndexFlatL2(vector_dim)
        self.metadata: List[Dict[str, Any]] = []
        
    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]):
        """Add vectors and metadata to index"""
        try:
            self.index.add(vectors)
            self.metadata.extend(metadata)
            logger.info(f"Added {len(vectors)} vectors to FAISS index")
        except Exception as e:
            logger.error(f"Error adding vectors to FAISS: {str(e)}")
            raise
    
    def similarity_search(self, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors and return metadata"""
        try:
            # Reshape query vector if needed
            if len(query_vector.shape) == 1:
                query_vector = query_vector.reshape(1, -1)
            
            # Perform search
            distances, indices = self.index.search(query_vector, k)
            
            # Format results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.metadata):
                    result = self.metadata[idx].copy()
                    result['distance'] = float(distances[0][i])
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []
    
    def save_index(self, path: Path):
        """Save FAISS index and metadata"""
        try:
            faiss.write_index(self.index, str(path / "vectors.index"))
            with open(path / "metadata.pkl", "wb") as f:
                pickle.dump(self.metadata, f)
            logger.info(f"Saved FAISS index to {path}")
        except Exception as e:
            logger.error(f"Error saving FAISS index: {str(e)}")
            raise
    
    def load_index(self, path: Path):
        """Load FAISS index and metadata"""
        try:
            self.index = faiss.read_index(str(path / "vectors.index"))
            with open(path / "metadata.pkl", "rb") as f:
                self.metadata = pickle.load(f)
            logger.info(f"Loaded FAISS index from {path}")
        except Exception as e:
            logger.error(f"Error loading FAISS index: {str(e)}")
            raise