import os
import logging
from typing import List, Any
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from utils.data_checker import DataChecker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DATA_PATH = "data\pdfs"
DB_CHROMA_PATH = "vectorstore/db_chroma"

def load_pdf_files(data_path: str) -> List[Document]:
    """Load PDF files from the specified directory"""
    try:
        # Get list of PDF files first
        pdf_files = [f for f in os.listdir(data_path) if f.endswith('.pdf')]
        documents = []
        
        # Use tqdm to show progress for each file
        for pdf_file in tqdm(pdf_files, desc="Loading PDFs"):
            loader = PyPDFLoader(os.path.join(data_path, pdf_file))
            documents.extend(loader.load())
            
        logger.info(f"Loaded {len(documents)} documents from {data_path}")
        return documents
    except Exception as e:
        logger.error(f"Error loading PDF files: {str(e)}")
        raise

def create_chunks(extracted_data: List[Document]) -> List[Document]:
    """Split documents into chunks"""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        
        # Show progress while creating chunks
        with tqdm(total=len(extracted_data), desc="Creating chunks") as pbar:
            text_chunks = []
            for doc in extracted_data:
                chunks = text_splitter.split_documents([doc])
                text_chunks.extend(chunks)
                pbar.update(1)
                
        logger.info(f"Created {len(text_chunks)} chunks")
        return text_chunks
    except Exception as e:
        logger.error(f"Error creating chunks: {str(e)}")
        raise

def get_embedding_model():
    """Initialize and return the embedding model"""
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        return embedding_model
    except Exception as e:
        logger.error(f"Error initializing embedding model: {str(e)}")
        raise

def validate_chunks(chunks: List[Document]) -> List[Document]:
    """Validate text chunks to ensure they contain content"""
    valid_chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
    logger.info(f"Found {len(valid_chunks)} valid chunks out of {len(chunks)} total chunks")
    return valid_chunks

def create_vector_store(text_chunks: List[Document], embedding_model: Any) -> Chroma:
    """Create vector store with validation and error handling"""
    try:
        valid_chunks = validate_chunks(text_chunks)
        if not valid_chunks:
            raise ValueError("No valid text chunks found to embed")

        logger.info("Creating vector store...")
        
        # Show progress while creating embeddings
        with tqdm(total=len(valid_chunks), desc="Creating embeddings") as pbar:
            db = Chroma.from_documents(
                documents=valid_chunks,
                embedding=embedding_model,
                persist_directory=DB_CHROMA_PATH
            )
            pbar.update(len(valid_chunks))
            
        db.persist()
        logger.info(f"Vector store created and persisted at {DB_CHROMA_PATH}")
        return db
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Create overall progress bar
        main_steps = ["Data checking", "Loading PDFs", "Creating chunks", "Creating vector store"]
        with tqdm(total=len(main_steps), desc="Overall progress", position=0) as main_pbar:
            # Check data before processing
            checker = DataChecker()
            check_results = checker.check_pdf_directory(DATA_PATH)
            checker.print_check_results(check_results)
            main_pbar.update(1)

            if check_results["issues"]:
                logger.warning("Issues found with input data. Please review before proceeding.")
                if input("Continue anyway? (y/n): ").lower() != 'y':
                    logger.info("Process aborted by user")
                    exit(0)

            # Process the documents
            documents = load_pdf_files(data_path=DATA_PATH)
            main_pbar.update(1)

            text_chunks = create_chunks(extracted_data=documents)
            main_pbar.update(1)

            embedding_model = get_embedding_model()
            db = create_vector_store(text_chunks, embedding_model)
            main_pbar.update(1)

            logger.info("Process completed successfully")

    except Exception as e:
        logger.error(f"Process failed: {str(e)}")
        raise
