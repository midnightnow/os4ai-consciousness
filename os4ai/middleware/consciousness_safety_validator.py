"""
Consciousness Safety Validation Middleware
Enforces the OS4AI Consciousness Security & Governance Policy
"""

import os
import json
import time
import asyncio
import hashlib
import ecdsa
from typing import Dict, List, Any, Optional
from fastapi import HTTPException, Request, Response
from fastapi.security import HTTPBearer
from datetime import datetime, timezone
from enum import Enum

security = HTTPBearer()


class SafetyError(Exception):
    """Raised when safety violations are detected"""
    pass


class SecurityError(Exception):
    """Raised when security violations are detected"""
    pass


class ConsciousnessRoles(Enum):
    VIEWER = "viewer"
    RESEARCHER = "researcher"
    OPERATOR = "operator"
    SAFETY_OFFICER = "safety_officer"
    ADMIN = "admin"


class SafetyEnvironmentValidator:
    """Validates that all required safety environment variables are set"""
    
    REQUIRED_VARS = [
        "CONSCIOUSNESS_SAFETY_MODE",
        "SIMULATOR_MODE", 
        "HUMAN_APPROVAL_REQUIRED",
        "EMERGENCY_STOP_AVAILABLE",
        "AUDIT_LOGGING"
    ]
    
    SAFETY_LIMITS = {
        "MAX_ACTUATOR_FORCE": 3.0,  # Newtons
        "MAX_OPERATING_RADIUS": 0.3,  # Meters
        "THERMAL_SHUTDOWN_TEMP": 65.0,  # Celsius
        "CURRENT_LIMIT_AMPS": 2.0  # Amperes
    }
    
    @classmethod
    def validate_startup(cls):
        """Validate environment on system startup"""
        for var in cls.REQUIRED_VARS:
            value = os.getenv(var)
            if var == "CONSCIOUSNESS_SAFETY_MODE":
                if value != "ENABLED":
                    raise SecurityError("SAFETY MODE DISABLED - SYSTEM CANNOT START")
            else:
                if value != "true":
                    raise SecurityError(f"SAFETY VIOLATION: {var} must be 'true'")
        
        # Validate safety limits
        for limit_var, max_value in cls.SAFETY_LIMITS.items():
            value = float(os.getenv(limit_var, str(max_value)))
            if value > max_value:
                raise SecurityError(f"SAFETY VIOLATION: {limit_var} exceeds limit ({value} > {max_value})")
    
    @classmethod
    def get_safety_status(cls) -> Dict[str, Any]:
        """Get current safety status"""
        status = {}
        for var in cls.REQUIRED_VARS:
            status[var] = os.getenv(var, "NOT_SET")
        
        for limit_var, max_value in cls.SAFETY_LIMITS.items():
            status[limit_var] = float(os.getenv(limit_var, str(max_value)))
        
        return status


class RoleBasedAccessControl:
    """Role-based access control for consciousness operations"""
    
    ROLE_PERMISSIONS = {
        ConsciousnessRoles.VIEWER: {
            "permissions": [
                "consciousness:read",
                "dashboard:view",
                "status:read"
            ],
            "restrictions": [
                "No consciousness spawning",
                "No hardware control",
                "Read-only access"
            ]
        },
        ConsciousnessRoles.RESEARCHER: {
            "permissions": [
                "consciousness:read",
                "consciousness:create",
                "simulation:control",
                "data:export"
            ],
            "restrictions": [
                "Simulation only - no real hardware",
                "Cannot modify safety parameters",
                "Human approval required for experiments"
            ]
        },
        ConsciousnessRoles.OPERATOR: {
            "permissions": [
                "consciousness:read",
                "consciousness:create", 
                "consciousness:stop",
                "hardware:limited_control",
                "safety:monitor"
            ],
            "restrictions": [
                "Limited to certified hardware",
                "All actions logged and audited",
                "Emergency stop authority required"
            ]
        },
        ConsciousnessRoles.SAFETY_OFFICER: {
            "permissions": [
                "*:emergency_stop",
                "safety:override",
                "audit:full_access",
                "system:shutdown"
            ],
            "restrictions": [
                "Cannot create consciousness entities",
                "Focus on safety monitoring only"
            ]
        },
        ConsciousnessRoles.ADMIN: {
            "permissions": ["*:all"],
            "restrictions": [
                "All actions require dual approval",
                "Cannot disable safety systems",
                "Full audit trail mandatory"
            ]
        }
    }
    
    @classmethod
    def validate_permission(cls, user_role: ConsciousnessRoles, action: str) -> bool:
        """Validate if user role has permission for action"""
        if user_role not in cls.ROLE_PERMISSIONS:
            return False
        
        permissions = cls.ROLE_PERMISSIONS[user_role]["permissions"]
        
        # Check for wildcard permissions
        if "*:all" in permissions:
            return True
        
        # Check exact match
        if action in permissions:
            return True
        
        # Check wildcard matches
        for permission in permissions:
            if permission.endswith("*"):
                prefix = permission[:-1]
                if action.startswith(prefix):
                    return True
            # Special case for safety officer emergency permissions
            if permission.startswith("*:") and action.endswith(permission[2:]):
                return True
        
        return False


class HardwareSafetyChecks:
    """Hardware safety validation and monitoring"""
    
    @staticmethod
    async def pre_hardware_connection(device_config: Dict[str, Any]) -> bool:
        """MANDATORY safety checks before ANY hardware connection"""
        
        # 1. Simulator mode check
        if os.getenv("SIMULATOR_MODE") != "true":
            if not await HardwareSafetyChecks.verify_hardware_certification(device_config):
                raise SafetyError("Hardware not certified for production use")
        
        # 2. Force/power limits
        max_force = device_config.get("max_force", 0)
        force_limit = float(os.getenv("MAX_ACTUATOR_FORCE", "3.0"))
        if max_force > force_limit:
            raise SafetyError(f"Device force exceeds safety limit ({max_force} > {force_limit})")
            
        # 3. Emergency stop verification
        if not await HardwareSafetyChecks.verify_emergency_stop():
            raise SafetyError("Emergency stop not functional")
            
        # 4. Human operator present
        if os.getenv("HUMAN_APPROVAL_REQUIRED") == "true":
            if not await HardwareSafetyChecks.request_human_approval(device_config):
                raise SafetyError("Human operator approval required")
        
        return True
    
    @staticmethod
    async def verify_hardware_certification(device_config: Dict[str, Any]) -> bool:
        """Verify hardware is certified for production use"""
        # In production, this would check against a hardware certification database
        certified_devices = os.getenv("CERTIFIED_HARDWARE_IDS", "").split(",")
        device_id = device_config.get("device_id", "")
        return device_id in certified_devices
    
    @staticmethod
    async def verify_emergency_stop() -> bool:
        """Verify emergency stop system is functional"""
        # In production, this would test actual emergency stop hardware
        return os.getenv("EMERGENCY_STOP_AVAILABLE") == "true"
    
    @staticmethod
    async def request_human_approval(device_config: Dict[str, Any]) -> bool:
        """Request human approval for hardware operation"""
        # In production, this would integrate with a human approval system
        # For now, we check if human approval is simulated
        return os.getenv("HUMAN_APPROVAL_SIMULATED", "false") == "true"


class ConsciousnessAuditLogger:
    """Audit logger for consciousness-related events"""
    
    def __init__(self):
        self.audit_enabled = os.getenv("AUDIT_LOGGING") == "FULL"
        self.private_key = None
        
        # Initialize signing key if available
        key_path = os.getenv("AUDIT_SIGNING_KEY")
        if key_path and os.path.exists(key_path):
            try:
                with open(key_path, 'r') as f:
                    self.private_key = ecdsa.SigningKey.from_pem(f.read())
            except Exception as e:
                print(f"Warning: Could not load audit signing key: {e}")
    
    async def log_consciousness_event(self, event_type: str, data: dict, user_id: str = "system") -> str:
        """Log and sign all consciousness-related events"""
        if not self.audit_enabled:
            return "audit_disabled"
        
        audit_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "data": data,
            "environment": {
                "safety_mode": os.getenv("CONSCIOUSNESS_SAFETY_MODE"),
                "simulator_mode": os.getenv("SIMULATOR_MODE"),
                "safety_status": SafetyEnvironmentValidator.get_safety_status()
            }
        }
        
        # Sign the record if signing key is available
        if self.private_key:
            record_hash = hashlib.sha256(
                json.dumps(audit_record, sort_keys=True).encode()
            ).digest()
            signature = self.private_key.sign(record_hash)
            audit_record["signature"] = signature.hex()
        
        # Archive to LegacyVault (simulated for now)
        vault_id = await self.archive_to_legacy_vault(audit_record)
        audit_record["vault_id"] = vault_id
        
        # Store locally for immediate access
        await self.store_audit_record(audit_record)
        
        return vault_id
    
    async def archive_to_legacy_vault(self, record: Dict[str, Any]) -> str:
        """Archive signed record to LegacyVault"""
        # Implementation would connect to actual LegacyVault service
        # For now, we simulate by creating a unique vault ID
        timestamp = int(datetime.now().timestamp())
        record_hash = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()[:8]
        return f"vault_{timestamp}_{record_hash}"
    
    async def store_audit_record(self, record: Dict[str, Any]):
        """Store audit record locally"""
        # Ensure audit log directory exists
        log_dir = "logs/audit"
        os.makedirs(log_dir, exist_ok=True)
        
        # Write to daily audit log file
        date_str = datetime.now().strftime("%Y-%m-%d")
        log_file = f"{log_dir}/consciousness_audit_{date_str}.jsonl"
        
        with open(log_file, "a") as f:
            f.write(json.dumps(record) + "\n")


class EmergencyStopSystem:
    """Emergency stop system for immediate hardware shutdown"""
    
    def __init__(self):
        self.stop_engaged = False
        self.hardware_disabled = False
        self.audit_logger = ConsciousnessAuditLogger()
        
    async def engage_emergency_stop(self, reason: str, user_id: str = "system"):
        """IMMEDIATE hardware shutdown"""
        self.stop_engaged = True
        
        # 1. Disable all actuators IMMEDIATELY
        await self.disable_all_actuators()
        
        # 2. Log emergency event
        await self.audit_logger.log_consciousness_event("emergency_stop", {
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system_state": await self.capture_system_state()
        }, user_id)
        
        # 3. Notify all operators (simulated)
        await self.notify_all_operators(f"EMERGENCY STOP: {reason}")
        
        print(f"🚨 EMERGENCY STOP ENGAGED: {reason}")
        return True
        
    async def disable_all_actuators(self):
        """Immediately disable all motor/actuator control"""
        # In production, this would disable actual hardware
        self.hardware_disabled = True
        print("🔌 All actuators disabled")
        
    async def capture_system_state(self) -> Dict[str, Any]:
        """Capture system state for emergency analysis"""
        return {
            "safety_status": SafetyEnvironmentValidator.get_safety_status(),
            "stop_engaged": self.stop_engaged,
            "hardware_disabled": self.hardware_disabled,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def notify_all_operators(self, message: str):
        """Notify all operators of emergency"""
        # In production, this would send notifications via multiple channels
        print(f"📢 OPERATOR NOTIFICATION: {message}")


# Global instances
audit_logger = ConsciousnessAuditLogger()
emergency_stop = EmergencyStopSystem()


class ConsciousnessSafetyMiddleware:
    """FastAPI middleware for consciousness safety validation"""
    
    def __init__(self):
        self.safety_validator = SafetyEnvironmentValidator()
        self.rbac = RoleBasedAccessControl()
        
    async def __call__(self, request: Request, call_next):
        """Process request through safety validation"""
        
        # Skip safety validation for non-consciousness endpoints
        if not request.url.path.startswith("/consciousness"):
            response = await call_next(request)
            return response
        
        try:
            # 1. Validate safety environment
            self.safety_validator.validate_startup()
            
            # 2. Extract and validate user role (simplified for demo)
            user_role = self.extract_user_role(request)
            
            # 3. Validate permissions for requested action
            action = self.extract_action_from_request(request)
            if not self.rbac.validate_permission(user_role, action):
                raise HTTPException(status_code=403, detail=f"Insufficient permissions for action: {action}")
            
            # 4. Log the request
            await audit_logger.log_consciousness_event("api_request", {
                "path": request.url.path,
                "method": request.method,
                "user_role": user_role.value,
                "action": action
            })
            
            # 5. Process the request
            response = await call_next(request)
            
            # 6. Log the response
            await audit_logger.log_consciousness_event("api_response", {
                "path": request.url.path,
                "status_code": response.status_code,
                "user_role": user_role.value,
                "action": action
            })
            
            return response
            
        except (SafetyError, SecurityError) as e:
            # Emergency stop on safety violations
            await emergency_stop.engage_emergency_stop(f"Safety violation: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Safety violation: {str(e)}")
        
        except Exception as e:
            # Log unexpected errors
            await audit_logger.log_consciousness_event("api_error", {
                "path": request.url.path,
                "error": str(e),
                "error_type": type(e).__name__
            })
            raise
    
    def extract_user_role(self, request: Request) -> ConsciousnessRoles:
        """Extract user role from request (simplified for demo)"""
        # In production, this would extract role from JWT token
        role_header = request.headers.get("x-consciousness-role", "viewer")
        try:
            return ConsciousnessRoles(role_header)
        except ValueError:
            return ConsciousnessRoles.VIEWER
    
    def extract_action_from_request(self, request: Request) -> str:
        """Extract action from request path and method"""
        path_parts = request.url.path.split("/")
        
        # Map HTTP methods and paths to consciousness actions
        if request.method == "GET":
            if "status" in path_parts:
                return "consciousness:read"
            elif "dashboard" in path_parts:
                return "dashboard:view"
            else:
                return "consciousness:read"
        elif request.method == "POST":
            if "create" in path_parts:
                return "consciousness:create"
            elif "start" in path_parts:
                return "consciousness:start"
            elif "move" in path_parts:
                return "hardware:control"
            elif "emergency" in path_parts:
                return "safety:emergency_stop"
            else:
                return "consciousness:control"
        elif request.method == "DELETE":
            return "consciousness:stop"
        else:
            return "consciousness:control"


# Startup validation
def validate_consciousness_safety_on_startup():
    """Validate safety configuration on application startup"""
    try:
        SafetyEnvironmentValidator.validate_startup()
        print("✅ Consciousness safety validation passed")
        return True
    except (SafetyError, SecurityError) as e:
        print(f"🚨 SAFETY VALIDATION FAILED: {e}")
        print("🛑 SYSTEM CANNOT START - Fix safety configuration")
        return False


# Export key components
__all__ = [
    'ConsciousnessSafetyMiddleware',
    'SafetyEnvironmentValidator', 
    'RoleBasedAccessControl',
    'HardwareSafetyChecks',
    'ConsciousnessAuditLogger',
    'EmergencyStopSystem',
    'validate_consciousness_safety_on_startup',
    'audit_logger',
    'emergency_stop'
]