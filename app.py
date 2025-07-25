import os
import streamlit as st

# Add these lines at the very start of your app.py, before any other imports
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

import logging
from pathlib import Path
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any
import traceback
from src.chatbot import FinancialChatbot
from src.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def initialize_session_state():
    """Initialize Streamlit session state with error handling"""
    try:
        if 'initialized' not in st.session_state:
            config = Config()  # Initialize config first
            st.session_state.update({
                'config': config,
                'chatbot': FinancialChatbot(),  # This will now have proper DB path
                'messages': [],
                'error_count': 0,
                'last_error_time': None,
                'recovery_mode': False,
                'selected_categories': set(),
                'last_query_time': None,
                'query_count': 0,
                'initialized': True
            })
    except Exception as e:
        logger.error(f"Error initializing session state: {str(e)}")
        st.error("Failed to initialize application. Please refresh the page.")
        raise

def render_sidebar():
    """Render sidebar with system status"""
    # System status
    st.sidebar.title("🔧 System Status")
    if st.session_state.error_count > 0:
        st.sidebar.error(f"⚠️ Recent Errors: {st.session_state.error_count}")
        if st.sidebar.button("Reset Error Count"):
            st.session_state.error_count = 0
            st.session_state.last_error_time = None
    
    # Query statistics
    if st.session_state.query_count > 0:
        st.sidebar.info(f"📊 Queries Today: {st.session_state.query_count}")
        
    # Recovery mode indicator
    if st.session_state.recovery_mode:
        st.sidebar.warning("🔄 System in Recovery Mode")

def render_chat_message(message: Dict[str, Any]):
    """Render chat message with NLP insights"""
    try:
        with st.chat_message(message["role"]):
            # Main message content
            st.markdown(message["content"])
            
            # Show NLP insights
            if "nlp_result" in message:
                with st.expander("🧠 NLP Insights"):
                    st.json(message["nlp_result"])

            # Additional information for assistant messages
            if message["role"] == "assistant":
                # Thinking process
                if "thinking" in message and message["thinking"]:
                    with st.expander("💭 Thinking Process"):
                        st.markdown(message["thinking"])
                
                # Retry information
                if "retry_count" in message and message["retry_count"] > 0:
                    st.warning(f"⚠️ Retried {message['retry_count']} times")
                
                # Response time
                if "response_time" in message:
                    st.caption(f"Response time: {message['response_time']:.2f} seconds")
                
                # Categories
                if "categories" in message and message["categories"]:
                    st.caption(f"Categories: {', '.join(message['categories'])}")
    except Exception as e:
        logger.error(f"Error rendering message: {str(e)}")
        st.error("Failed to render message properly")

def handle_rate_limiting() -> bool:
    """Handle rate limiting and return whether request should proceed"""
    now = datetime.now()
    
    # Reset daily query count
    if (st.session_state.last_query_time and 
        st.session_state.last_query_time.date() != now.date()):
        st.session_state.query_count = 0
    
    # Check rate limits
    if st.session_state.query_count >= 100:  # Daily limit
        st.error("Daily query limit reached. Please try again tomorrow.")
        return False
    
    if (st.session_state.last_query_time and 
        (now - st.session_state.last_query_time) < timedelta(seconds=1)):
        st.warning("Please wait a moment before sending another message.")
        return False
    
    return True

async def process_query(prompt: str) -> Dict[str, Any]:
    """Process user query with timing and error handling"""
    start_time = datetime.now()
    
    try:
        # Process the query
        response = await st.session_state.chatbot.process_query(prompt)
        
        # Calculate response time
        response_time = (datetime.now() - start_time).total_seconds()
        
        # Update query statistics
        st.session_state.query_count += 1
        st.session_state.last_query_time = datetime.now()
        
        # Add timing information to response
        response["response_time"] = response_time
        
        return response
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Update error tracking
        st.session_state.error_count += 1
        st.session_state.last_error_time = datetime.now()
        
        # Check if should enter recovery mode
        if st.session_state.error_count >= 3:
            st.session_state.recovery_mode = True
            
        raise

def render_welcome_message():
    """Render welcome message and usage guidelines"""
    st.markdown("""
    # 🏦 Welcome to Financial Advisor AI!
    
    I'm here to help you with:
    - 📈 Investment advice and portfolio management
    - 🏦 Banking and savings strategies
    - 🛡️ Insurance planning
    - 📊 Retirement planning
    - 💰 Debt management
    - 📑 Tax considerations
    
    Ask me anything about your financial questions!
    
    *Note: This is an AI assistant providing general financial information.*
    """)

async def main():
    """Main application function with error handling"""
    try:
        # Page configuration
        st.set_page_config(
            page_title="Financial Advisor AI",
            page_icon="🏦",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Initialize session state
        initialize_session_state()
        
        # Render sidebar
        render_sidebar()
        
        # Show welcome message for new sessions
        if not st.session_state.messages:
            render_welcome_message()
        
        # Display chat history
        for message in st.session_state.messages:
            render_chat_message(message)
        
        # Chat input
        if prompt := st.chat_input("Ask me anything about finance..."):
            # Check rate limits
            if not handle_rate_limiting():
                return
            
            # Add user message
            user_message = {
                "role": "user",
                "content": prompt,
                "timestamp": datetime.now().isoformat()
            }
            st.session_state.messages.append(user_message)
            render_chat_message(user_message)
            
            try:
                # Process query
                with st.spinner("Thinking..."):
                    response = await process_query(prompt)
                
                # Create bot message
                bot_message = {
                    "role": "assistant",
                    "content": response["response"],
                    "thinking": response.get("thinking", ""),
                    "retry_count": response.get("retry_count", 0),
                    "response_time": response.get("response_time", 0),
                    "categories": list(st.session_state.selected_categories),
                    "timestamp": datetime.now().isoformat()
                }
                
                # Add and render bot message
                st.session_state.messages.append(bot_message)
                render_chat_message(bot_message)
                
                # Reset recovery mode if successful
                if st.session_state.recovery_mode:
                    st.session_state.recovery_mode = False
                    st.session_state.error_count = 0
                    st.success("System recovered successfully!")
                
            except Exception as e:
                error_message = {
                    "role": "assistant",
                    "content": f"I apologize, but I encountered an error: {str(e)}. Please try again.",
                    "timestamp": datetime.now().isoformat()
                }
                st.session_state.messages.append(error_message)
                render_chat_message(error_message)
    
    except Exception as e:
        logger.error(f"Critical application error: {str(e)}")
        logger.error(traceback.format_exc())
        st.error("A critical error occurred. Please refresh the page.")

if __name__ == "__main__":
    asyncio.run(main())