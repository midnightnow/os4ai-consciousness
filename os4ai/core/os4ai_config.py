"""
OS4AI Perfect Consciousness System - Secure Configuration
Centralized configuration for OS4AI with security best practices
"""

from typing import List, Optional, Dict
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from functools import lru_cache
import secrets
import os
import re


class OS4AISettings(BaseSettings):
    """OS4AI-specific settings with validation"""
    
    # Security - No hardcoded secrets!
    jwt_secret: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    
    # Redis Configuration with Connection Pooling
    redis_url: str = "redis://localhost:6379/0"
    redis_password: Optional[str] = None
    redis_pool_size: int = 10
    redis_max_connections: int = 50
    redis_socket_keepalive: bool = True
    redis_socket_keepalive_options: Dict = Field(default_factory=dict)
    
    # OS4AI Mode Configuration
    os4ai_docker_mode: bool = False
    os4ai_use_mock_sensors: bool = False
    disable_background_tasks: bool = False

    # VetSorcery Testing Mode
    vetsorcery_testing_mode: bool = False
    
    # Sensor Configuration with Validation
    thermal_poll_interval: int = Field(10, ge=1, le=300)  # 1-300 seconds
    thermal_warning_threshold: float = Field(70.0, ge=0, le=150)
    thermal_critical_threshold: float = Field(85.0, ge=0, le=150)
    
    # Privacy Settings
    acoustic_privacy_mode: bool = True
    media_privacy_mode: bool = True
    wifi_privacy_mode: bool = True
    
    # Performance Tuning
    sensor_timeout_seconds: int = Field(5, ge=1, le=30)
    max_pattern_memory_size: int = Field(1000, ge=100, le=10000)
    state_history_size: int = Field(100, ge=10, le=1000)
    
    # Background Task Control
    thermal_monitor_enabled: bool = True
    acoustic_monitor_enabled: bool = True
    media_monitor_enabled: bool = True
    wifi_monitor_enabled: bool = True
    
    # WebSocket Configuration
    websocket_heartbeat_interval: int = Field(30, ge=10, le=300)
    websocket_max_connections: int = Field(100, ge=1, le=1000)
    websocket_ping_timeout: int = Field(20, ge=5, le=60)
    
    # Rate Limiting
    rate_limit_requests_per_minute: int = Field(60, ge=10, le=1000)
    rate_limit_burst_size: int = Field(10, ge=1, le=100)
    
    # Command Sanitization Patterns
    dangerous_command_patterns: List[str] = [
        r"[;&|`$(){}[\]<>\\'\"]",  # Shell metacharacters
        r"\.\./",  # Directory traversal
        r"(sudo|su|chmod|chown|rm|dd|mkfs|format)",  # Dangerous commands
    ]
    
    # File Upload Security
    allowed_upload_extensions: List[str] = [".jpg", ".jpeg", ".png", ".wav", ".mp3"]
    max_upload_size_mb: int = Field(10, ge=1, le=100)
    upload_directory: str = "/tmp/os4ai_uploads"
    
    @field_validator("thermal_critical_threshold")
    @classmethod
    def validate_thermal_thresholds(cls, v, info):
        """Ensure critical > warning threshold"""
        warning = info.data.get("thermal_warning_threshold", 70) if info.data else 70
        if v <= warning:
            raise ValueError(f"Critical threshold ({v}) must be > warning ({warning})")
        return v
    
    @field_validator("upload_directory")
    @classmethod
    def validate_upload_directory(cls, v):
        """Ensure upload directory is safe"""
        if not v.startswith(("/tmp", "/var/tmp")):
            raise ValueError("Upload directory must be in /tmp or /var/tmp")
        return v
    
    @field_validator("redis_url")
    @classmethod
    def validate_redis_url(cls, v):
        """Validate Redis URL format"""
        if not re.match(r"^rediss?://[\w\-\.]+:\d+(/\d+)?$", v):
            raise ValueError("Invalid Redis URL format")
        return v
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore"  # Ignore extra fields from .env
    }
        
    def get_redis_pool_config(self) -> dict:
        """Get Redis connection pool configuration"""
        return {
            "max_connections": self.redis_max_connections,
            "socket_keepalive": self.redis_socket_keepalive,
            "socket_keepalive_options": self.redis_socket_keepalive_options,
            "health_check_interval": 30,
            "retry_on_timeout": True,
            "socket_connect_timeout": 5,
        }
    
    def should_monitor_sensor(self, sensor_type: str) -> bool:
        """Check if a sensor should be monitored"""
        if self.disable_background_tasks:
            return False
            
        sensor_map = {
            "thermal": self.thermal_monitor_enabled,
            "acoustic": self.acoustic_monitor_enabled,
            "media": self.media_monitor_enabled,
            "wifi": self.wifi_monitor_enabled,
        }
        return sensor_map.get(sensor_type, False)


@lru_cache()
def get_os4ai_settings() -> OS4AISettings:
    """Get cached OS4AI settings instance"""
    return OS4AISettings()


# Security utilities for OS4AI
class CommandSanitizer:
    """Sanitize shell commands to prevent injection"""
    
    def __init__(self, settings: OS4AISettings):
        self.settings = settings
        self.compiled_patterns = [
            re.compile(pattern) for pattern in settings.dangerous_command_patterns
        ]
    
    def sanitize(self, command: str) -> str:
        """Remove dangerous patterns from command"""
        # Limit length
        command = command[:256]
        
        # Remove dangerous patterns
        for pattern in self.compiled_patterns:
            command = pattern.sub("", command)
        
        # Whitelist allowed characters
        command = re.sub(r"[^a-zA-Z0-9\s\-_./]", "", command)
        
        return command.strip()
    
    def is_safe_command(self, command: str) -> bool:
        """Check if command is safe to execute"""
        # Check against dangerous patterns
        for pattern in self.compiled_patterns:
            if pattern.search(command):
                return False
        
        # Check for suspicious keywords
        suspicious = ["eval", "exec", "compile", "__import__", "globals", "locals"]
        if any(word in command.lower() for word in suspicious):
            return False
            
        return True


class PathValidator:
    """Validate file paths for security"""
    
    @staticmethod
    def is_safe_path(path: str, base_path: str) -> bool:
        """Check if path is within base directory (no traversal)"""
        try:
            # Resolve both paths
            resolved_path = os.path.abspath(os.path.normpath(path))
            resolved_base = os.path.abspath(os.path.normpath(base_path))
            
            # Check if resolved path starts with base
            return resolved_path.startswith(resolved_base)
        except Exception:
            return False
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for safe storage"""
        # Remove path components
        filename = os.path.basename(filename)
        
        # Remove dangerous characters
        filename = re.sub(r"[^a-zA-Z0-9\-_.]", "", filename)
        
        # Limit length
        name, ext = os.path.splitext(filename)
        if len(name) > 100:
            name = name[:100]
        
        return name + ext