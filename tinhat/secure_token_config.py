
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
