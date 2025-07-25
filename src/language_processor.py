from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class LanguageProcessor:
    def __init__(self):
        self.model_name = "google/flan-t5-small"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
        except Exception as e:
            logger.error(f"Error initializing T5 model: {str(e)}")
            raise

    def process_query(self, query: str) -> Dict[str, Any]:
        """Process user query using T5 model"""
        try:
            # Prepare input text
            input_text = f"process financial query: {query}"
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

            # Generate response
            try:
                outputs = self.model.generate(
                    **inputs,
                    max_length=150,
                    num_return_sequences=1,
                    temperature=0.7
                )
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                raise

            # Decode response
            processed_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract intents and entities (basic implementation)
            intents = ["financial_advice"] if "advice" in processed_text else []
            entities = [word for word in query.split() if word.isalpha() and word.lower() not in ['what', 'is', 'the', 'a', 'an']]

            return {
                "processed_text": processed_text,
                "intents": intents,
                "entities": entities
            }

        except Exception as e:
            logger.error(f"Error processing query with T5: {str(e)}")
            raise
