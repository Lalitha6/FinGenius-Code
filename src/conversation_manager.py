from typing import Dict, Any, List, Tuple
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import spacy
import re

class ConversationManager:
    def __init__(self):
        self.conversation_history = []
        self.user_profiles = {}
        self.context_window = 5
        
        # Initialize FLAN-T5 model
        self.model_name = "google/flan-t5-small"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Load spaCy for linguistic analysis
        self.nlp = spacy.load("en_core_web_sm")
        
        # Financial patterns and contexts
        self.financial_patterns = {
            'instruments': r'\b(stock|bond|mutual fund|etf|cryptocurrency|forex|option)\b',
            'banking': r'\b(savings|checking|deposit|withdraw|transfer|account|bank|credit|debit)\b',
            'metrics': r'\b(interest rate|apr|yield|dividend|return|profit|loss|cash flow)\b',
            'activities': r'\b(invest|trade|buy|sell|finance|budget|save|spend)\b',
            'insurance': r'\b(insurance|policy|premium|claim|coverage|liability|comprehensive|third-party|renewal)\b'
        }
        
        # Add insurance entities to recognized financial terms
        self.financial_entities = ["bank", "financial", "insurance", "policy", "coverage"]
        
        self.non_financial_contexts = {
            'political': ['government', 'minister', 'president', 'election', 'party', 'policy'],
            'general_news': ['announced', 'said', 'stated', 'declared'],
            'biographical': ['born', 'died', 'married', 'studied', 'graduated']
        }
        
        self.example_pairs = [
            ("What is the current stock price of Apple?", "Financial"),
            ("Who is the finance minister?", "Non-Financial"),
            ("How do I invest in mutual funds?", "Financial"),
            ("The bank of the river is eroding", "Non-Financial"),
            ("What's the interest rate on savings accounts?", "Financial"),
            ("She banks the plane in the air", "Non-Financial")
        ]

    def _analyze_linguistic_structure(self, query: str) -> Dict[str, Any]:
        """Analyze the linguistic structure of the query"""
        doc = self.nlp(query)
        
        subjects = [token for token in doc if "subj" in token.dep_]
        objects = [token for token in doc if "obj" in token.dep_]
        verbs = [token for token in doc if token.pos_ == "VERB"]
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        return {
            "subjects": subjects,
            "objects": objects,
            "verbs": verbs,
            "entities": entities
        }

    def _check_contextual_relevance(self, query: str, linguistic_analysis: Dict[str, Any]) -> float:
        """Determine if financial terms are used in a financial context"""
        score = 0.0
        doc = self.nlp(query.lower())
        
        for category, pattern in self.financial_patterns.items():
            matches = re.finditer(pattern, query.lower())
            for match in matches:
                term = match.group()
                term_idx = -1
                for i, token in enumerate(doc):
                    if token.text == term:
                        term_idx = i
                        break
                
                if term_idx != -1:
                    context_start = max(0, term_idx - 3)
                    context_end = min(len(doc), term_idx + 4)
                    context = doc[context_start:context_end]
                    
                    non_financial = False
                    for token in context:
                        if any(token.text in indicators for indicators in self.non_financial_contexts.values()):
                            non_financial = True
                            break
                    
                    if not non_financial:
                        score += 0.2
        
        # Update entity checking to include insurance terms
        for ent_text, ent_label in linguistic_analysis["entities"]:
            if ent_label in ["ORG", "PERSON", "GPE"] and any(term in ent_text.lower() for term in self.financial_entities):
                score += 0.1
        
        return min(score, 1.0)

    def _generate_few_shot_prompt(self, query: str) -> str:
        """Generate a few-shot learning prompt for FLAN-T5"""
        prompt = "Classify the following queries as either 'Financial' or 'Non-Financial' based on whether they seek financial information or advice:\n\n"
        
        for example_query, classification in self.example_pairs:
            prompt += f"Query: {example_query}\nAnswer: {classification}\n\n"
        
        prompt += f"Query: {query}\nAnswer:"
        return prompt

    def is_financial_query(self, query: str) -> bool:
        """Enhanced method to determine if a query is financial"""
        linguistic_analysis = self._analyze_linguistic_structure(query)
        context_score = self._check_contextual_relevance(query, linguistic_analysis)
        
        prompt = self._generate_few_shot_prompt(query)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=10)
        classification = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        confidence = 0.0
        if "financial" in classification.lower():
            confidence += 0.4
        
        confidence += context_score * 0.6
        
        non_financial_indicators = sum(1 for context_words in self.non_financial_contexts.values()
                                     for word in context_words if word in query.lower())
        
        if non_financial_indicators > 0:
            confidence -= 0.3 * non_financial_indicators
        
        return confidence >= 0.5

    def add_interaction(self, user_id: str, query: str, response: Dict[str, Any]):
        """Add new interaction to conversation history"""
        if not self.is_financial_query(query):
            return {
                'response': "I apologize, but I can only assist with financial questions. "
                           "Please ask me about investments, banking, insurance, retirement, "
                           "taxes, or other financial matters.",
                'valid_query': False
            }
        
        timestamp = datetime.now()
        
        interaction = {
            'timestamp': timestamp,
            'query': query,
            'response': response,
            'topics': self._extract_topics(query),
            'sentiment': self._analyze_sentiment(query)
        }
        
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
        
        self.conversation_history[user_id].append(interaction)
        self._update_user_profile(user_id, interaction)
    
    def get_relevant_context(self, user_id: str, current_query: str) -> Dict[str, Any]:
        """Get relevant context from previous interactions"""
        if user_id not in self.conversation_history:
            return {}
        
        recent_history = self.conversation_history[user_id][-self.context_window:]
        current_topics = self._extract_topics(current_query)
        
        relevant_interactions = []
        for interaction in recent_history:
            if self._calculate_topic_overlap(current_topics, interaction['topics']) > 0.3:
                relevant_interactions.append(interaction)
        
        return {
            'previous_interactions': relevant_interactions,
            'user_profile': self.user_profiles.get(user_id, {}),
            'topic_progression': self._analyze_topic_progression(user_id)
        }
    
    def _update_user_profile(self, user_id: str, interaction: Dict[str, Any]):
        """Update user profile based on new interaction"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                'topics_of_interest': set(),
                'risk_preferences': [],
                'financial_goals': set(),
                'interaction_patterns': []
            }
        
        profile = self.user_profiles[user_id]
        topics = interaction['topics']
        
        # Update topics of interest
        profile['topics_of_interest'].update(topics)
        
        # Update risk preferences if present in interaction
        if 'risk_score' in interaction['response']:
            profile['risk_preferences'].append(interaction['response']['risk_score'])
        
        # Extract and update financial goals
        goals = self._extract_financial_goals(interaction['query'])
        profile['financial_goals'].update(goals)
        
        # Update interaction patterns
        profile['interaction_patterns'].append({
            'time': interaction['timestamp'],
            'topic_category': self._categorize_topic(topics)
        })
    
    def _extract_topics(self, query: str) -> list:
        """Extract topics from query"""
        return []

    def _analyze_sentiment(self, query: str) -> str:
        """Analyze sentiment of query"""
        return "neutral"

    def _calculate_topic_overlap(self, topics1: list, topics2: list) -> float:
        """Calculate topic overlap between two lists of topics"""
        return 0.0

    def _analyze_topic_progression(self, user_id: str) -> list:
        """Analyze topic progression for user"""
        return []

    def _extract_financial_goals(self, query: str) -> list:
        """Extract financial goals from query"""
        return []

    def _categorize_topic(self, topics: list) -> str:
        """Categorize topic"""
        return "general"