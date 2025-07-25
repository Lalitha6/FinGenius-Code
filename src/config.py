from pathlib import Path
from typing import Dict, Any
import yaml
import os

class Config:
    def __init__(self):
        # Use absolute path based on the project root
        self.PROJECT_ROOT = Path(__file__).parent.parent
        self.BASE_DIR = self.PROJECT_ROOT / "financial_data"
        self.PDF_DIR = self.BASE_DIR / "pdfs"
        self.PROCESSED_DIR = self.BASE_DIR / "processed"
        self.LOG_DIR = self.BASE_DIR / "logs"
        self.DB_PATH = self.BASE_DIR / "finance_chat.db"
        
        # Model settings
        self.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        self.LLM_MODEL = "deepseek-r1:latest"
        
        # Retry settings
        self.MAX_RETRIES = 3
        self.RETRY_DELAY = 1  # seconds
        self.RETRY_BACKOFF = 2
        
        # Error tracking
        self.ERROR_THRESHOLD = 5
        self.ERROR_WINDOW = 300  # 5 minutes
        
        # Create directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories"""
        for dir_path in [self.BASE_DIR, self.PDF_DIR, self.PROCESSED_DIR, self.LOG_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
    def save_error_state(self, state: Dict[str, Any]):
        """Save error state to file"""
        with (self.LOG_DIR / "error_state.yaml").open("w") as f:
            yaml.dump(state, f)
    
    def load_error_state(self) -> Dict[str, Any]:
        """Load error state from file"""
        state_file = self.LOG_DIR / "error_state.yaml"
        if state_file.exists():
            with state_file.open("r") as f:
                return yaml.safe_load(f)
        return {}
