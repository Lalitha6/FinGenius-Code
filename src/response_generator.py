from typing import Dict, Any, List, Optional
import logging
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from .config import Config

logger = logging.getLogger(__name__)

class ResponseGenerator:
    def __init__(self, config: Config):
        self.config = config
        self.llm = OllamaLLM(model=config.LLM_MODEL)
        
        self.system_prompt = ChatPromptTemplate.from_template("""
            You are a financial advisor AI with access to the following context:
            
            Topics: {topics}
            Similar Documents: {similar_docs}
            Previous Context: {previous_context}
            
            Given this context, provide advice for the following query:
            {query}
            
            Consider:
            1. Relevance of detected topics
            2. Information from similar documents
            3. Previous conversation context
            4. Current market conditions
            
            First, analyze the situation between <think> tags.
            Then provide your advice in a clear, structured format.
        """)
    
    async def generate_response(self,
                              query: str,
                              topics: List[tuple[str, float]],
                              similar_docs: List[Dict[str, Any]],
                              previous_context: Optional[str] = None) -> Dict[str, Any]:
        """Generate response using context and LLM"""
        try:
            # Format context
            context = {
                'query': query,
                'topics': self._format_topics(topics),
                'similar_docs': self._format_similar_docs(similar_docs),
                'previous_context': previous_context or "No previous context"
            }
            
            # Generate response
            chain = self.system_prompt | self.llm
            full_response = await chain.ainvoke(context)
            
            # Parse response
            thinking, response = self._parse_response(full_response)
            
            return {
                'response': response,
                'thinking': thinking,
                'context_used': context
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                'response': f"I apologize, but I encountered an error: {str(e)}",
                'thinking': f"Error occurred during processing: {str(e)}",
                'context_used': {}
            }
    
    def _format_topics(self, topics: List[tuple[str, float]]) -> str:
        """Format topics for prompt"""
        return "\n".join(
            f"- {topic} (confidence: {score:.2f})"
            for topic, score in topics
        )
    
    def _format_similar_docs(self, docs: List[Dict[str, Any]]) -> str:
        """Format similar documents for prompt"""
        formatted_docs = []
        for doc in docs:
            formatted_docs.append(
                f"Document: {doc.get('title', 'Untitled')}\n"
                f"Source: {doc.get('source', 'Unknown')}\n"
                f"Relevance: {1 - doc.get('distance', 0):.2f}\n"
                f"Content: {doc.get('full_content', '')[:500]}..."
            )
        return "\n\n".join(formatted_docs)
    
    def _parse_response(self, full_response: str) -> tuple[str, str]:
        """Parse thinking and response from LLM output"""
        thinking = ""
        response = full_response
        
        if "<think>" in full_response and "</think>" in full_response:
            parts = full_response.split("</think>")
            thinking = parts[0].split("<think>")[1].strip()
            response = parts[1].strip()
            
        return thinking, response
