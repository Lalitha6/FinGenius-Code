from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from src.vector_store import FAISSManager
from src.topic_detector import TopicDetector
from src.utils import RetryHandler
from src.document_processor import DocumentProcessor  # Import DocumentProcessor
from tqdm import tqdm  # Import tqdm for progress bar

logger = logging.getLogger(__name__)

class KnowledgeBase:
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.vector_store = FAISSManager()
        self.topic_detector = TopicDetector(base_path / "models")
        self.document_index: Dict[str, Dict[str, Any]] = {}
        self.document_processor = DocumentProcessor()  # Initialize DocumentProcessor
        
    @RetryHandler(max_retries=3, delay=1, backoff=2)
    def load_documents(self, documents_path: Path):
        """Load and index documents"""
        try:
            # Load document metadata
            metadata_path = documents_path / "metadata.csv"
            if metadata_path.exists():
                metadata_df = pd.read_csv(metadata_path)
                
                # Add progress bar for document processing
                with tqdm(total=len(metadata_df), desc="Processing documents") as pbar:
                    for _, row in metadata_df.iterrows():
                        doc_path = documents_path / row['filename']
                        if doc_path.exists():
                            if doc_path.suffix.lower() == '.pdf':
                                text_chunks = self.document_processor.process_pdf(doc_path)
                                for chunk in text_chunks:
                                    self._index_document(
                                        content=chunk,
                                        metadata={
                                            'id': row['id'],
                                            'title': row['title'],
                                            'source': row['source'],
                                            'date': row['date'],
                                            'filename': row['filename']
                                        }
                                    )
                            else:
                                with open(doc_path, 'r') as f:
                                    content = f.read()
                                self._index_document(
                                    content=content,
                                    metadata={
                                        'id': row['id'],
                                        'title': row['title'],
                                        'source': row['source'],
                                        'date': row['date'],
                                        'filename': row['filename']
                                    }
                                )
                        pbar.update(1)
            
            logger.info(f"Loaded and indexed documents from {documents_path}")
            
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            raise
    
    def _index_document(self, content: str, metadata: Dict[str, Any]):
        """Index a single document"""
        try:
            # Detect topics
            topics = self.topic_detector.detect_topics(content)
            
            # Store document info
            doc_id = metadata['id']
            self.document_index[doc_id] = {
                **metadata,
                'topics': topics,
                'content': content
            }
            
            # Add to vector store
            self.vector_store.add_vectors(
                vectors=np.array([self._generate_embedding(content)]),
                metadata=[{
                    **metadata,
                    'topics': [t[0] for t in topics]
                }]
            )
            
        except Exception as e:
            logger.error(f"Error indexing document {metadata.get('id')}: {str(e)}")
            raise

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search knowledge base for relevant documents"""
        try:
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            
            # Get similar documents
            results = self.vector_store.similarity_search(
                query_vector=query_embedding,
                k=k
            )
            
            # Enhance results with full document content
            enhanced_results = []
            for result in results:
                doc_id = result['id'] if result and 'id' in result else None
                if doc_id and doc_id in self.document_index:
                    enhanced_results.append({
                        **result,
                        'full_content': self.document_index[doc_id]['content'],
                        'topics': self.document_index[doc_id]['topics']
                    })
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Error searching knowledge base: {str(e)}")
            return []

    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text using sentence transformer"""
        # Note: Implementation would depend on the specific embedding model being used
        # This is a placeholder for the actual embedding generation
        raise NotImplementedError("Embedding generation not implemented.")
