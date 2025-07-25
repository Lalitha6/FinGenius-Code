from queue import Queue
from threading import Thread, Event
from typing import Callable, Any, Dict, List
import logging
import time
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class WorkItem:
    id: str
    func: Callable
    args: tuple
    kwargs: Dict[str, Any]
    timestamp: datetime = datetime.now()
    retries: int = 0
    max_retries: int = 3
    
    def execute(self) -> Any:
        """Execute work item with retry logic"""
        last_error = None
        while self.retries <= self.max_retries:
            try:
                return self.func(*self.args, **self.kwargs)
            except Exception as e:
                last_error = e
                self.retries += 1
                if self.retries <= self.max_retries:
                    time.sleep(2 ** self.retries)  # Exponential backoff
        raise last_error

class WorkQueue:
    def __init__(self, num_workers: int = 3):
        self.queue: Queue = Queue()
        self.workers: List[Thread] = []
        self.stop_event = Event()
        self.num_workers = num_workers
        
    def start(self):
        """Start worker threads"""
        for _ in range(self.num_workers):
            worker = Thread(target=self._worker_loop)
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
        logger.info(f"Started {self.num_workers} worker threads")
    
    def stop(self):
        """Stop worker threads"""
        self.stop_event.set()
        for worker in self.workers:
            worker.join()
        logger.info("Stopped all worker threads")
    
    def _worker_loop(self):
        """Main worker loop"""
        while not self.stop_event.is_set():
            try:
                work_item: WorkItem = self.queue.get(timeout=1)
                try:
                    result = work_item.execute()
                    logger.debug(f"Completed work item {work_item.id}")
                except Exception as e:
                    logger.error(f"Failed to process work item {work_item.id}: {str(e)}")
                finally:
                    self.queue.task_done()
            except Queue.Empty:
                continue
    
    def add_work(self, work_item: WorkItem):
        """Add work item to queue"""
        self.queue.put(work_item)
        logger.debug(f"Added work item {work_item.id} to queue")
    
    def wait_completion(self):
        """Wait for all queued work to complete"""
        self.queue.join()
