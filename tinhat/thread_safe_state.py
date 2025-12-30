
"""
Thread-Safe State Machine
Prevents race conditions and invalid transitions
"""

import asyncio
from enum import Enum
from typing import Dict, List, Optional, Set
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

class SystemState(Enum):
    """System states with clear semantics"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"
    SHUTDOWN = "shutdown"

class ThreadSafeStateManager:
    """Thread-safe state management with transition validation"""
    
    # Define valid state transitions
    VALID_TRANSITIONS: Dict[SystemState, Set[SystemState]] = {
        SystemState.IDLE: {SystemState.INITIALIZING, SystemState.SHUTDOWN},
        SystemState.INITIALIZING: {SystemState.READY, SystemState.ERROR},
        SystemState.READY: {SystemState.PROCESSING, SystemState.IDLE, SystemState.ERROR},
        SystemState.PROCESSING: {SystemState.READY, SystemState.ERROR},
        SystemState.ERROR: {SystemState.INITIALIZING, SystemState.SHUTDOWN},
        SystemState.SHUTDOWN: set()  # Terminal state
    }
    
    def __init__(self):
        self._state = SystemState.IDLE
        self._lock = asyncio.Lock()
        self._transition_history = []
        self._state_listeners = []
    
    async def get_state(self) -> SystemState:
        """Get current state (thread-safe)"""
        async with self._lock:
            return self._state
    
    async def transition(self, new_state: SystemState) -> bool:
        """Attempt state transition with validation"""
        async with self._lock:
            # Validate transition
            if new_state not in self.VALID_TRANSITIONS[self._state]:
                logger.error(
                    f"Invalid transition attempted: {self._state.value} → {new_state.value}"
                )
                return False
            
            # Record transition
            old_state = self._state
            self._state = new_state
            
            transition_record = {
                "from": old_state.value,
                "to": new_state.value,
                "timestamp": datetime.now(timezone.utc),
                "thread_id": asyncio.current_task().get_name()
            }
            
            self._transition_history.append(transition_record)
            
            # Log transition
            logger.info(
                f"State transition: {old_state.value} → {new_state.value} "
                f"(Task: {transition_record['thread_id']})"
            )
            
            # Notify listeners
            for listener in self._state_listeners:
                asyncio.create_task(listener(old_state, new_state))
            
            return True
    
    async def force_shutdown(self):
        """Force system to shutdown state (emergency)"""
        async with self._lock:
            logger.warning(f"Forcing shutdown from {self._state.value}")
            self._state = SystemState.SHUTDOWN
    
    def add_listener(self, callback):
        """Add state change listener"""
        self._state_listeners.append(callback)
    
    def get_transition_history(self, limit: int = 100) -> List[Dict]:
        """Get recent transition history"""
        return self._transition_history[-limit:]
    
    def is_operational(self) -> bool:
        """Check if system is in operational state"""
        return self._state in {SystemState.READY, SystemState.PROCESSING}

# Global instance
state_manager = ThreadSafeStateManager()
