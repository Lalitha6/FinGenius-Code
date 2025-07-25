# Financial Advisor Chatbot System

## Core Components

### Main Application Files

#### /src/chatbot.py
- Main chatbot implementation class FinancialChatbot
- Handles query processing, LLM integration, and response generation
- Essential as the core engine for handling financial queries and providing responses


#### /src/config.py
- Configuration management class Config
- Manages paths, model settings, retry settings, and error tracking
- Necessary for centralizing application configuration and maintaining consistent settings


#### /src/conversation_manager.py
- Manages conversation state and history
- Handles context tracking and financial query classification
- Important for maintaining conversation flow and ensuring queries are finance-related


### Data Management

#### /src/database.py
- DatabaseManager class for SQLite database operations
- Stores chat history and error logs
- Critical for persistence and analytics of chat interactions


#### /src/document_processor.py
- Handles PDF document processing and text chunking
- Uses PyMuPDF (fitz) for PDF extraction
- Essential for processing financial documents and knowledge base creation


### Financial Processing

#### /src/financial_calculator.py
- Contains financial calculation logic
- Handles loan metrics, EMI calculations, and insurance needs
- Necessary for providing accurate financial computations


#### /src/financial_detector.py
- Enhanced detection of financial queries
- Uses NLP and pattern matching to identify financial context
- Critical for filtering non-financial queries


### Knowledge Management

#### /src/knowledge_base.py
- Manages the financial knowledge base
- Integrates with FAISS for vector storage
- Important for retrieving relevant financial information


#### /src/language_processor.py
- Handles natural language processing tasks
- Uses transformers for query understanding
- Essential for understanding user intent and context


### Response Generation

#### /src/response_generator.py
- Generates structured responses
- Uses templates and context for coherent answers
- Necessary for providing well-formatted, contextual responses


#### /src/topic_detector.py
- Detects financial topics in text
- Uses TF-IDF and named entity recognition
- Important for understanding query topics and context


### User Interface

#### /src/ui.py
- Streamlit UI components
- Handles theme switching and chat interface
- Essential for user interaction


### Utility Components

#### /src/utils.py
- Utility functions for retry handling and error tracking
- Contains helper classes for application robustness
- Necessary for error handling and reliability


#### /src/vector_store.py
- FAISS vector store management
- Handles similarity search and metadata
- Critical for efficient information retrieval


#### /utils/data_checker.py
- Data validation and checking functionality
- Ensures data quality and integrity
- Important for preventing issues with invalid data


## Application Entry Points

#### app.py
- Main Streamlit application
- Integrates all components
- Essential as the application entry point


#### main.py
- Document processing pipeline
- Sets up vector store and embeddings
- Necessary for initial system setup


## Project Configuration

#### requirements.txt
- Lists all Python dependencies
- Ensures consistent environment setup
- Critical for deployment and reproducibility


#### run.py
- Simple script to launch the Streamlit app
- Sets environment variables
- Convenient for application startup