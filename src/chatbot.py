from typing import Dict, Any
import logging
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from src.database import DatabaseManager
from src.language_processor import LanguageProcessor
from src.utils import RetryHandler
from src.config import Config
import asyncio
from src.conversation_manager import ConversationManager  # Import ConversationManager
from src.financial_detector import EnhancedFinancialDetector

logger = logging.getLogger(__name__)

class FinancialChatbot:
    def __init__(self):
        self.config = Config()  # Initialize config first
        self.db_manager = DatabaseManager(db_path=self.config.DB_PATH)  # Pass db_path
        self.language_processor = LanguageProcessor()
        self._initialize_llm()
        self.conversation_manager = ConversationManager()  # Initialize ConversationManager
        self.financial_detector = EnhancedFinancialDetector()

    @RetryHandler(max_retries=3, delay=1, backoff=2)
    def _initialize_llm(self):
        """Initialize LLM with retry mechanism"""
        try:
            self.llm = OllamaLLM(model="deepseek-r1:latest")
            self.system_prompt = ChatPromptTemplate.from_template("""
                You are a financial advisor AI. Provide accurate and helpful advice.
                Consider the following intents and entities: {intents_entities}
                
                User Query: {query}
                
                Provide your response in a clear and structured format.
            """)
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            raise

    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process user query with T5"""
        is_financial, confidence, reasoning = self.financial_detector.is_financial_query(query)
        if not is_financial:
            return {
                'response': "I apologize, but I can only assist with financial questions. "
                           "Please ask me about investments, banking, insurance, retirement, "
                           "taxes, or other financial matters.",
                'thinking': f"Query classified as non-financial (confidence: {confidence:.2f}). {reasoning}",
                'sources': []
            }
        
        try:
            # Process query with T5
            nlp_result = self.language_processor.process_query(query)

            # Generate response
            chain = self.system_prompt | self.llm
            full_response = await asyncio.get_event_loop().run_in_executor(
                None,
                chain.invoke,
                {
                    "query": query,
                    "intents_entities": nlp_result
                }
            )
            
            # Parse thinking and response
            thinking = f"Financial query detected (confidence: {confidence:.2f}). {reasoning}\n\n"
            response = full_response
            
            if "<think>" in full_response and "</think>" in full_response:
                parts = full_response.split("</think>")
                thinking += parts[0].split("<think>")[1].strip()
                response = parts[1].strip()
            
            # Save to database
            self.db_manager.save_chat(
                user_query=query,
                bot_response=response,
                intents=["finance"],
                sources=[],
                metadata={'confidence': confidence, 'reasoning': reasoning, **nlp_result}
            )
            
            return {
                "response": response,
                "thinking": thinking,
                "sources": []
            }
            
        except Exception as e:
            logger.error(f"Error in process_query: {str(e)}")
            return {
                "response": f"I apologize, but I encountered an error: {str(e)}. Please try again.",
                "thinking": "",
                "sources": []
            }
