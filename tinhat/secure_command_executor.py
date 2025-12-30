
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
            r'\x',      # Hex escapes
            r'\[0-7]',  # Octal escapes
            r'eval|exec|import|open|file|input|raw_input|compile',  # Dangerous Python
            r'os\.|subprocess\.|sys\.|__',  # Module access
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, user_input, re.IGNORECASE):
                return False
        
        return True

# Global instance
secure_executor = SecureCommandExecutor()
