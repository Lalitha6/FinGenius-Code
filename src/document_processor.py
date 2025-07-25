import fitz  # pymupdf
import logging
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        pass

    def process_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Process a PDF document and return a list of text chunks"""
        try:
            doc = fitz.open(pdf_path)
            text_chunks = []

            for page_num in tqdm(range(doc.page_count), desc=f"Processing {pdf_path.name}"):
                page = doc.load_page(page_num)
                text = page.get_text("text")
                text_chunks.extend(self._chunk_text(text))

            doc.close()
            return text_chunks

        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            return []

    def _chunk_text(self, text: str, chunk_size: int = 512, chunk_overlap: int = 64) -> List[str]:
        """Split text into smaller chunks with overlap"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += (chunk_size - chunk_overlap)
        return chunks
