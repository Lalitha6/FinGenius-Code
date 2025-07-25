import streamlit as st
from pathlib import Path
from .chatbot import FinancialChatbot
from .document_processor import DocumentManager
from . import config
import logging
import traceback

logger = logging.getLogger(__name__)

def init_ui():
    """Initialize the Streamlit UI with theme switching"""
    st.set_page_config(page_title="Financial Advisor AI", layout="wide")
    
    # Theme state management
    if 'theme' not in st.session_state:
        st.session_state.theme = 'dark'
    
    # Theme switcher in sidebar
    theme = st.sidebar.radio(
        "Choose Theme",
        options=['light', 'dark'],
        index=0 if st.session_state.theme == 'light' else 1,
        key='theme_selector'
    )
    
    # Update theme state
    st.session_state.theme = theme
    
    # Custom styling based on theme
    if theme == 'dark':
        st.markdown("""
            <style>
            .stApp {
                background-color: #1E1E1E;
                color: #FFFFFF;
            }
            .chat-message {
                background-color: #2E2E2E;
                padding: 1rem;
                border-radius: 0.5rem;
                margin: 1rem 0;
            }
            .source-citation {
                font-size: 0.8em;
                color: #888;
                margin-top: 5px;
            }
            .status-box {
                background-color: #2E2E2E;
                padding: 10px;
                border-radius: 5px;
                margin: 10px 0;
            }
            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
            .stApp {
                background-color: #FFFFFF;
                color: #000000;
            }
            .chat-message {
                background-color: #F0F2F6;
                padding: 1rem;
                border-radius: 0.5rem;
                margin: 1rem 0;
            }
            .source-citation {
                font-size: 0.8em;
                color: #666;
                margin-top: 5px;
            }
            .status-box {
                background-color: #F0F2F6;
                padding: 10px;
                border-radius: 5px;
                margin: 10px 0;
            }
            </style>
        """, unsafe_allow_html=True)

def update_stats(chat_history):
    """Update and display system statistics"""
    st.sidebar.markdown(f"""
        <div class="status-box">
        📝 Total Conversations: {len(chat_history)}<br>
        🕒 Last Interaction: {chat_history[-1]['timestamp'] if chat_history else 'None'}<br>
        </div>
    """, unsafe_allow_html=True)

def handle_chat_response(chatbot, prompt: str) -> dict:
    """
    Handle the chatbot response and return formatted data
    
    Args:
        chatbot: FinancialChatbot instance
        prompt (str): User's question
        
    Returns:
        dict: Contains formatted response with the following keys:
            - response (str): Main response from the chatbot
            - thinking (str): Reasoning/thought process
    """
    try:
        # Get response from chatbot
        response_data = chatbot.process_query(prompt)
        
        return {
            "response": response_data.get("response", "No response available"),
            "thinking": response_data.get("thinking", "No reasoning provided"),
        }
    except Exception as e:
        logger.error(f"Error in handle_chat_response: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"Error processing request: {str(e)}")
        return {
            "response": "I apologize, but I encountered an error processing your request.",
            "thinking": f"Error occurred: {str(e)}",
        }

def main():
    """Main application function"""
    init_ui()
    
    st.title("Financial Advisor AI")
    
    # Initialize chat history if not exists
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.container():
            st.markdown(f"""
                <div class="chat-message">
                    <strong>{'Assistant' if message['role'] == 'assistant' else 'You'}:</strong><br>
                    {message['content']}
                </div>
            """, unsafe_allow_html=True)
    
    # Chat input
    prompt = st.text_input("Ask a question:", key="user_input")
    
    if prompt:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get chatbot response
        response = handle_chat_response(None, prompt)  # Replace None with your chatbot instance
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response["response"]})
        
        # Force refresh
        st.experimental_rerun()
    
    # Update statistics
    update_stats(st.session_state.messages)

if __name__ == "__main__":
    main()