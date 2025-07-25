# financial_advisor/src/database.py
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import json
from src.utils import RetryHandler, ErrorTracker  # Updated import path

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path: Path):
        # Ensure db_path is a Path object and parent directory exists
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.error_tracker = ErrorTracker()
        self.init_database()
    
    @RetryHandler(max_retries=3, delay=1, backoff=2)
    def init_database(self):
        """Initialize database with retry mechanism"""
        logger.info("Initializing database...")
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Create chat history table with all required columns
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS chat_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        user_query TEXT,
                        bot_response TEXT,
                        error_context TEXT,
                        retry_count INTEGER DEFAULT 0,
                        success BOOLEAN DEFAULT TRUE,
                        intents TEXT,
                        sources TEXT,
                        metadata TEXT
                    )
                """)
                
                # Create error logs table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS error_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        error_type TEXT,
                        error_message TEXT,
                        stack_trace TEXT,
                        context TEXT
                    )
                """)

                # Add any missing columns to existing chat_history table
                try:
                    columns_to_add = [
                        ("error_context", "TEXT"),
                        ("retry_count", "INTEGER DEFAULT 0"),
                        ("success", "BOOLEAN DEFAULT TRUE"),
                        ("intents", "TEXT"),
                        ("sources", "TEXT"),
                        ("metadata", "TEXT")
                    ]
                    
                    for column_name, column_type in columns_to_add:
                        try:
                            conn.execute(f"""
                                ALTER TABLE chat_history 
                                ADD COLUMN {column_name} {column_type}
                            """)
                        except sqlite3.OperationalError as e:
                            if "duplicate column name" not in str(e).lower():
                                raise
                        
                except sqlite3.OperationalError as e:
                    logger.warning(f"Error adding columns: {str(e)}")
                    # Table might not exist yet, which is fine as it was created above
                    pass
                
            logger.info("Database initialized successfully")
        except Exception as e:
            self.error_tracker.add_error(e, {'operation': 'init_database'})
            raise
    
    @RetryHandler(max_retries=3, delay=1, backoff=2)
    def save_chat(self, user_query: str, bot_response: str, 
                  error_context: Dict[str, Any] = None,
                  retry_count: int = 0, success: bool = True,
                  intents: List[str] = None,
                  sources: List[str] = None,
                  metadata: Dict[str, Any] = None):
        """Save chat interaction with error handling"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """INSERT INTO chat_history 
                       (timestamp, user_query, bot_response, error_context,
                        retry_count, success, intents, sources, metadata) 
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        datetime.now().isoformat(),
                        user_query,
                        bot_response,
                        json.dumps(error_context or {}),
                        retry_count,
                        success,
                        json.dumps(intents or []),
                        json.dumps(sources or []),
                        json.dumps(metadata or {})
                    )
                )
            logger.info(f"Saved chat interaction: {user_query[:50]}...")
        except Exception as e:
            self.error_tracker.add_error(e, {
                'operation': 'save_chat',
                'user_query': user_query[:100]
            })
            raise