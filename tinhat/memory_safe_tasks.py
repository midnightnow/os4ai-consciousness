
"""
Memory-Safe Background Task Manager
Prevents memory leaks with proper cleanup
"""

import asyncio
import weakref
import gc
from typing import Dict, Set, Optional, Any
from datetime import datetime, timezone
import psutil
import logging

logger = logging.getLogger(__name__)

class MemorySafeTaskManager:
    """Background task manager with memory leak prevention"""
    
    def __init__(self, max_tasks: int = 10, memory_limit_mb: int = 512):
        self.max_tasks = max_tasks
        self.memory_limit_mb = memory_limit_mb
        self._tasks: Dict[str, asyncio.Task] = {}
        self._task_refs: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
        self._shutdown = False
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start task manager with cleanup monitoring"""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Memory-safe task manager started")
    
    async def stop(self):
        """Stop all tasks and cleanup"""
        self._shutdown = True
        
        # Cancel all tasks
        for task_id, task in list(self._tasks.items()):
            if not task.done():
                task.cancel()
        
        # Wait for cleanup
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Force garbage collection
        self._tasks.clear()
        self._task_refs.clear()
        gc.collect()
        
        logger.info("Memory-safe task manager stopped")
    
    async def create_task(
        self,
        task_id: str,
        coro,
        cleanup_callback=None
    ) -> Optional[asyncio.Task]:
        """Create managed task with cleanup"""
        
        # Check task limit
        if len(self._tasks) >= self.max_tasks:
            logger.warning(f"Task limit reached ({self.max_tasks})")
            return None
        
        # Check memory usage
        if not self._check_memory():
            logger.warning("Memory limit exceeded, refusing new task")
            return None
        
        # Cancel existing task with same ID
        if task_id in self._tasks:
            self._tasks[task_id].cancel()
        
        # Create wrapper with cleanup
        async def task_wrapper():
            try:
                result = await coro
                return result
            except asyncio.CancelledError:
                logger.debug(f"Task {task_id} cancelled")
                raise
            except Exception as e:
                logger.error(f"Task {task_id} error: {e}")
                raise
            finally:
                # Cleanup
                self._tasks.pop(task_id, None)
                if cleanup_callback:
                    try:
                        await cleanup_callback()
                    except Exception as e:
                        logger.error(f"Cleanup error for {task_id}: {e}")
        
        # Create and track task
        task = asyncio.create_task(task_wrapper())
        self._tasks[task_id] = task
        self._task_refs[task_id] = task
        
        return task
    
    async def _cleanup_loop(self):
        """Periodic cleanup of completed tasks"""
        while not self._shutdown:
            try:
                # Remove completed tasks
                completed = [
                    task_id for task_id, task in self._tasks.items()
                    if task.done()
                ]
                
                for task_id in completed:
                    self._tasks.pop(task_id, None)
                
                # Check memory usage
                if not self._check_memory():
                    logger.warning("Memory pressure detected, forcing cleanup")
                    gc.collect()
                
                # Log statistics
                if len(self._tasks) > 0:
                    memory_info = psutil.Process().memory_info()
                    logger.debug(
                        f"Active tasks: {len(self._tasks)}, "
                        f"Memory: {memory_info.rss / 1024 / 1024:.1f}MB"
                    )
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(60)
    
    def _check_memory(self) -> bool:
        """Check if memory usage is within limits"""
        try:
            memory_info = psutil.Process().memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            return memory_mb < self.memory_limit_mb
        except Exception:
            return True  # Assume OK if can't check
    
    def get_stats(self) -> Dict[str, Any]:
        """Get task manager statistics"""
        memory_info = psutil.Process().memory_info()
        
        return {
            "active_tasks": len(self._tasks),
            "max_tasks": self.max_tasks,
            "memory_mb": memory_info.rss / 1024 / 1024,
            "memory_limit_mb": self.memory_limit_mb,
            "task_ids": list(self._tasks.keys())
        }

# Global instance
memory_safe_tasks = MemorySafeTaskManager()
