from typing import Dict, Any, List, Tuple
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import spacy
import re

class EnhancedFinancialDetector:
    def __init__(self):
        # Initialize FLAN-T5 model
        self.model_name = "google/flan-t5-small"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Load spaCy for linguistic analysis
        self.nlp = spacy.load("en_core_web_sm")
        
        # Expanded financial patterns with more insurance terms
        self.financial_patterns = {
            'instruments': r'\b(stock|bond|mutual fund|etf|cryptocurrency|forex|option)\b',
            'banking': r'\b(savings|checking|deposit|withdraw|transfer|account|bank|credit|debit)\b',
            'metrics': r'\b(interest rate|apr|yield|dividend|return|profit|loss|cash flow)\b',
            'activities': r'\b(invest|trade|buy|sell|finance|budget|save|spend)\b',
            'insurance': r'\b(insurance|policy|premium|claim|coverage|liability|comprehensive|third-party|renewal|insurer|deductible|policyholder)\b',
            'insurance_types': r'\b(car insurance|auto insurance|vehicle insurance|motor insurance|life insurance|health insurance|property insurance)\b',
            'insurance_terms': r'\b(risk|accident|damage|theft|disaster|protection|benefits|cover|covered)\b'
        }
        
        # Updated non-financial contexts
        self.non_financial_contexts = {
            'political': ['government', 'minister', 'president', 'election', 'party'],
            'general_news': ['announced', 'said', 'stated', 'declared'],
            'biographical': ['born', 'died', 'married', 'studied', 'graduated']
        }
        
        # Example pairs for few-shot learning
        self.example_pairs = [
            ("What is the current stock price of Apple?", "Financial"),
            ("Who is the finance minister?", "Non-Financial"),
            ("How do I invest in mutual funds?", "Financial"),
            ("The bank of the river is eroding", "Non-Financial"),
            ("What's the interest rate on savings accounts?", "Financial"),
            ("She banks the plane in the air", "Non-Financial"),
            ("Should I renew my car insurance policy?", "Financial"),
            ("What coverage does home insurance provide?", "Financial"),
            ("The insurance company processed my claim", "Financial")
        ]
        
        # Add more insurance-related example pairs
        self.example_pairs.extend([
            ("Should I switch from comprehensive to third-party car insurance?", "Financial"),
            ("What's the difference between comprehensive and third-party coverage?", "Financial"),
            ("How much can I save by changing my insurance policy?", "Financial"),
            ("Is third party insurance mandatory in Delhi?", "Financial"),
            ("What risks do I face with basic insurance coverage?", "Financial")
        ])

    def _analyze_linguistic_structure(self, query: str) -> Dict[str, Any]:
        """Analyze the linguistic structure of the query"""
        doc = self.nlp(query)
        
        # Extract key linguistic features
        subjects = [token for token in doc if "subj" in token.dep_]
        objects = [token for token in doc if "obj" in token.dep_]
        verbs = [token for token in doc if token.pos_ == "VERB"]
        
        # Analyze named entities
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
        
        # Check for insurance-specific context
        insurance_terms = set()
        for category in ['insurance', 'insurance_types', 'insurance_terms']:
            pattern = self.financial_patterns[category]
            matches = re.finditer(pattern, query.lower())
            for match in matches:
                insurance_terms.add(match.group())
        
        # Boost score for insurance-related queries
        if len(insurance_terms) > 0:
            score += 0.3  # Base score for insurance terms
            score += min(0.3, len(insurance_terms) * 0.1)  # Additional score for multiple terms
        
        # Check for location context with insurance terms
        locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
        if locations and insurance_terms:
            score += 0.2
        
        # Regular financial pattern checking
        for category, pattern in self.financial_patterns.items():
            if category not in ['insurance', 'insurance_types', 'insurance_terms']:
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
                            score += 0.1
        
        # Analyze named entities
        for ent_text, ent_label in linguistic_analysis["entities"]:
            if ent_label in ["ORG", "PERSON", "GPE"] and any(term in ent_text.lower() for term in ["bank", "financial", "insurance"]):
                score += 0.1
        
        return min(score, 1.0)

    def _generate_few_shot_prompt(self, query: str) -> str:
        """Generate a few-shot learning prompt for FLAN-T5"""
        prompt = "Classify the following queries as either 'Financial' or 'Non-Financial' based on whether they seek financial information or advice:\n\n"
        
        # Add example pairs
        for example_query, classification in self.example_pairs:
            prompt += f"Query: {example_query}\nAnswer: {classification}\n\n"
        
        # Add the current query
        prompt += f"Query: {query}\nAnswer:"
        
        return prompt

    def is_financial_query(self, query: str) -> Tuple[bool, float, str]:
        """
        Determine if a query is financial in nature using multiple analysis methods.
        Returns a tuple of (is_financial, confidence_score, reasoning)
        """
        # Step 1: Linguistic Analysis
        linguistic_analysis = self._analyze_linguistic_structure(query)
        
        # Step 2: Context Relevance Score
        context_score = self._check_contextual_relevance(query, linguistic_analysis)
        
        # Step 3: FLAN-T5 Classification
        prompt = self._generate_few_shot_prompt(query)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=10)
        classification = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Step 4: Combine Evidence
        is_financial = False
        confidence = 0.0
        reasoning = []
        
        # Weight the evidence
        if "financial" in classification.lower():
            confidence += 0.4
            reasoning.append("Model classified as financial query")
        
        confidence += context_score * 0.6
        
        if context_score > 0:
            reasoning.append(f"Found financial context with score {context_score:.2f}")
        
        # Check for clear non-financial indicators
        non_financial_indicators = sum(1 for context_words in self.non_financial_contexts.values()
                                     for word in context_words if word in query.lower())
        
        if non_financial_indicators > 0:
            confidence -= 0.3 * non_financial_indicators
            reasoning.append(f"Found {non_financial_indicators} non-financial context indicators")
        
        # Boost confidence for insurance-specific queries
        if any(term in query.lower() for term in ["insurance", "policy", "coverage", "premium"]):
            confidence += 0.2
            reasoning.append("Contains insurance-specific terms")
        
        # Adjust threshold for insurance queries
        is_financial = confidence >= 0.4  # Lower threshold for insurance queries
        
        reasoning_str = " | ".join(reasoning)
        
        return is_financial, confidence, reasoning_str
