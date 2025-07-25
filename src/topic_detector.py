from typing import List, Tuple, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import KeyedVectors
import spacy
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class TopicDetector:
    def __init__(self, model_path: Path):
        """Initialize topic detector with pre-trained models"""
        self.nlp = spacy.load("en_core_web_sm")
        self.tfidf = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=10000,
            stop_words='english'
        )
        
        # Load pre-trained word vectors
        try:
            self.word_vectors = KeyedVectors.load(str(model_path))
            logger.info("Loaded word vectors successfully")
        except Exception as e:
            logger.warning(f"Could not load word vectors: {str(e)}")
            self.word_vectors = None
    
    def detect_topics(self, text: str) -> List[Tuple[str, float]]:
        """Detect financial topics in text using multiple methods"""
        # Extract topics using TF-IDF
        tfidf_topics = self._extract_tfidf_topics(text)
        
        # Extract named entities
        entities = self._extract_entities(text)
        
        # Combine and score topics
        topics = self._combine_topics(tfidf_topics, entities)
        
        return self._filter_financial_topics(topics)
    
    def _extract_tfidf_topics(self, text: str) -> List[Tuple[str, float]]:
        """Extract topics using TF-IDF"""
        # Vectorize text
        vector = self.tfidf.transform([text])
        
        # Get feature names and scores
        feature_names = self.tfidf.get_feature_names_out()
        scores = vector.toarray()[0]
        
        # Sort by score
        topics = [(feature_names[i], float(scores[i]))
                 for i in range(len(feature_names))
                 if scores[i] > 0]
        
        return sorted(topics, key=lambda x: x[1], reverse=True)
    
    def _extract_entities(self, text: str) -> List[Tuple[str, float]]:
        """Extract named entities using spaCy"""
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'MONEY', 'PERCENT']:
                entities.append((ent.text, 1.0))
        
        return entities
    
    def _combine_topics(self, 
                       tfidf_topics: List[Tuple[str, float]],
                       entities: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Combine and deduplicate topics"""
        # Create dictionary to store combined scores
        topic_scores: Dict[str, float] = {}
        
        # Add TF-IDF topics
        for topic, score in tfidf_topics:
            topic_scores[topic.lower()] = score
        
        # Add entities with higher weight
        for entity, score in entities:
            entity_lower = entity.lower()
            if entity_lower in topic_scores:
                topic_scores[entity_lower] *= 1.5
            else:
                topic_scores[entity_lower] = score * 1.5
        
        # Convert back to list and sort
        combined = [(topic, score) 
                   for topic, score in topic_scores.items()]
        return sorted(combined, key=lambda x: x[1], reverse=True)
    
    def _filter_financial_topics(self, 
                               topics: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Filter topics to keep only finance-related ones"""
        financial_topics = []
        
        for topic, score in topics:
            # Skip very short topics
            if len(topic) < 3:
                continue
            
            # Check if topic is finance-related using word vectors
            if self.word_vectors:
                try:
                    similarity = self.word_vectors.similarity(topic, 'finance')
                    if similarity > 0.3:  # Threshold for financial relevance
                        financial_topics.append((topic, score * similarity))
                except KeyError:
                    # Word not in vocabulary
                    continue
            else:
                # Fallback to simple keyword matching
                financial_keywords = {'finance', 'money', 'investment', 'stock',
                                   'bond', 'market', 'fund', 'bank', 'credit',
                                   'debt', 'interest', 'dividend', 'portfolio'}
                if any(keyword in topic.lower() for keyword in financial_keywords):
                    financial_topics.append((topic, score))
        
        return sorted(financial_topics, key=lambda x: x[1], reverse=True)[:5]