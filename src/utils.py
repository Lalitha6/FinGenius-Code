import time
from functools import wraps
from typing import Callable, Any, Dict, List
import logging
from datetime import datetime, timedelta
import threading
from queue import Queue
import traceback

logger = logging.getLogger(__name__)

class RetryHandler:
    def __init__(self, max_retries: int = 3, delay: int = 1, backoff: int = 2):
        self.max_retries = max_retries
        self.delay = delay
        self.backoff = backoff
        
    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            retries = 0
            current_delay = self.delay
            
            while retries < self.max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries == self.max_retries:
                        logger.error(f"Max retries ({self.max_retries}) reached for {func.__name__}")
                        raise
                    
                    logger.warning(
                        f"Attempt {retries} failed for {func.__name__}. "
                        f"Retrying in {current_delay} seconds. Error: {str(e)}"
                    )
                    
                    time.sleep(current_delay)
                    current_delay *= self.backoff
            
            return None
        return wrapper

class ErrorTracker:
    def __init__(self, threshold: int = 5, window_seconds: int = 300):
        self.threshold = threshold
        self.window = timedelta(seconds=window_seconds)
        self.errors: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
        
    def add_error(self, error: Exception, context: Dict[str, Any] = None):
        """Add an error to the tracker"""
        with self.lock:
            now = datetime.now()
            self.errors.append({
                'timestamp': now,
                'error': str(error),
                'traceback': traceback.format_exc(),
                'context': context or {}
            })
            
            # Clean old errors
            self.errors = [
                e for e in self.errors 
                if now - e['timestamp'] <= self.window
            ]
    
    def should_alert(self) -> bool:
        """Check if error threshold is exceeded"""
        with self.lock:
            return len(self.errors) >= self.threshold
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of recent errors"""
        with self.lock:
            return {
                'total_errors': len(self.errors),
                'unique_errors': len(set(e['error'] for e in self.errors)),
                'latest_error': self.errors[-1] if self.errors else None,
                'errors': self.errors
            }