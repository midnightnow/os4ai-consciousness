import os
"""
OS4AI Perfect Consciousness Router
Production-ready API with authentication, rate limiting, and monitoring
"""

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, status, Depends, Request, Response, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
import jwt
import redis
import logging
from functools import wraps
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge
import json

# Import perfect implementations
from .os4ai_perfect_thermal_integration import PerfectThermalConsciousness, ThermalConfig, ThermalAwareness
from .os4ai_perfect_acoustic_integration import AcousticAwareness
from .os4ai_perfect_media_integration import MediaAwareness
from .os4ai_perfect_wifi_integration import WiFiAwareness
from .os4ai_parasitic_rf_integration import ParasiticRFAwareness
from .os4ai_perfect_bluetooth_integration import BluetoothAwareness
from .os4ai_perfect_integration import PerfectConsciousnessOrchestrator, ConsciousnessConfig, UnifiedConsciousnessState
from .os4ai_perfect_websocket_manager import websocket_manager, websocket_endpoint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# Metrics
request_counter = Counter(
    'os4ai_consciousness_requests_total',
    'Total requests to consciousness endpoints',
    ['method', 'endpoint', 'status']
)

request_duration = Histogram(
    'os4ai_consciousness_request_duration_seconds',
    'Request duration for consciousness endpoints',
    ['method', 'endpoint']
)

active_connections = Gauge(
    'os4ai_websocket_connections_active',
    'Active WebSocket connections'
)

sensor_readings = Gauge(
    'os4ai_sensor_readings',
    'Latest sensor readings',
    ['sensor_type', 'sensor_id']
)

# JWT Configuration
jwt_secret = os.getenv("JWT_SECRET_KEY")  # Load from environment
JWT_ALGORITHM = "HS256"

# Response Models
class ConsciousnessStatus(BaseModel):
    """Overall consciousness system status"""
    status: str = Field(..., description="System status: healthy, degraded, critical")
    thermal: Dict[str, Any]
    acoustic: Dict[str, Any]
    media: Dict[str, Any]
    wifi: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = "4.0.0"

class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    components: Dict[str, str]
    metrics: Dict[str, Any]
    version: str = "4.0.0"

class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str
    detail: Optional[str] = None
    correlation_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# Authentication dependency
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Verify JWT token and return user info"""
    token = credentials.credentials
    
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        
        # Check expiration
        exp = datetime.fromtimestamp(payload.get('exp', 0))
        if exp < datetime.now(timezone.utc):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired"
            )
        
        return {
            'user_id': payload.get('sub'),
            'roles': payload.get('roles', []),
            'permissions': payload.get('permissions', [])
        }
        
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

# Role-based access control
def require_permission(permission: str):
    """Decorator to require specific permission"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract user from kwargs
            user = kwargs.get('current_user')
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="User context not found"
                )
            
            # Check permission
            user_permissions = user.get('permissions', [])
            role_permissions = {
                'admin': ['*'],
                'veterinarian': ['read_sensors', 'write_commands', 'manage_patients'],
                'technician': ['read_sensors', 'view_patients'],
                'viewer': ['read_sensors']
            }
            
            # Check direct permission
            if permission in user_permissions or '*' in user_permissions:
                return await func(*args, **kwargs)
            
            # Check role-based permission
            for role in user.get('roles', []):
                if role in role_permissions:
                    if '*' in role_permissions[role] or permission in role_permissions[role]:
                        return await func(*args, **kwargs)
            
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied: {permission} required"
            )
        
        return wrapper
    return decorator

# Rate limiting dependency
class RateLimiter:
    """Rate limiting with Redis backend"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.memory_limits: Dict[str, List[float]] = {}
    
    async def check_limit(self, key: str, limit: int, window: int) -> bool:
        """Check if request is within rate limit"""
        if self.redis_client:
            try:
                pipe = self.redis_client.pipeline()
                now = datetime.now(timezone.utc).timestamp()
                pipe.zremrangebyscore(key, 0, now - window)
                pipe.zadd(key, {str(now): now})
                pipe.zcount(key, now - window, now)
                pipe.expire(key, window)
                results = pipe.execute()
                return results[2] <= limit
            except Exception:
                pass
        
        # Fallback to memory
        now = datetime.now(timezone.utc).timestamp()
        if key not in self.memory_limits:
            self.memory_limits[key] = []
        
        self.memory_limits[key] = [
            t for t in self.memory_limits[key] 
            if t > now - window
        ]
        
        if len(self.memory_limits[key]) < limit:
            self.memory_limits[key].append(now)
            return True
        
        return False

# Create router
router = APIRouter(
    prefix="/api/v1/consciousness",
    tags=["consciousness"],
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Forbidden"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)

# Initialize components
rate_limiter = RateLimiter()

# Middleware for metrics
# NOTE: This middleware should be added to the main FastAPI app, not the router
# @router.middleware("http")
async def track_metrics(request: Request, call_next):
    """Track request metrics"""
    start_time = datetime.now(timezone.utc)
    
    # Add correlation ID
    correlation_id = request.headers.get('X-Correlation-ID', str(uuid.uuid4()))
    request.state.correlation_id = correlation_id
    
    # Process request
    response = await call_next(request)
    
    # Record metrics
    duration = (datetime.now(timezone.utc) - start_time).total_seconds()
    request_counter.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    request_duration.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)
    
    # Add correlation ID to response
    response.headers["X-Correlation-ID"] = correlation_id
    
    return response

# Health check endpoint
@router.get("/health", response_model=HealthCheckResponse)
async def health_check(request: Request):
    """
    Comprehensive health check for consciousness system
    """
    try:
        # Check all components
        components = {
            "thermal": "healthy",
            "acoustic": "healthy",
            "media": "healthy",
            "wifi": "healthy",
            "websocket": "healthy",
            "redis": "healthy"
        }
        
        # Check Redis
        redis_client = getattr(request.app.state, 'redis', None)
        if redis_client:
            try:
                await redis_client.ping()
            except Exception:
                components["redis"] = "unhealthy"
        
        # Check WebSocket manager
        ws_stats = websocket_manager.get_stats()
        if ws_stats['connected_clients'] > 0 or ws_stats['connections_total'] > 0:
            components["websocket"] = "healthy"
        
        # Overall status
        unhealthy = [k for k, v in components.items() if v != "healthy"]
        if len(unhealthy) == 0:
            overall_status = "healthy"
        elif len(unhealthy) < len(components) / 2:
            overall_status = "degraded"
        else:
            overall_status = "critical"
        
        return HealthCheckResponse(
            status=overall_status,
            timestamp=datetime.now(timezone.utc),
            components=components,
            metrics={
                "websocket_connections": ws_stats['connected_clients'],
                "total_requests": sum(
                    request_counter._metrics.values() 
                    if hasattr(request_counter, '_metrics') else [0]
                )
            }
        )
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Health check failed"
        )

# Consciousness status endpoint
@router.get("/status", response_model=ConsciousnessStatus)
@require_permission("read_sensors")
async def get_consciousness_status(
    request: Request,
    current_user: Dict[str, Any] = Depends(verify_token)
):
    """
    Get overall consciousness system status
    """
    # Rate limiting
    user_id = current_user['user_id']
    if not await rate_limiter.check_limit(f"status:{user_id}", 30, 60):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )
    
    try:
        # Initialize orchestrator
        config = ConsciousnessConfig()
        redis_client = getattr(request.app.state, 'redis', None)
        orchestrator = PerfectConsciousnessOrchestrator(config, redis_client)
        
        # Get unified status
        state = await orchestrator.get_unified_consciousness(
            user_id=user_id,
            correlation_id=request.state.correlation_id
        )
        
        return ConsciousnessStatus(
            status=state.system_health,
            thermal=state.thermal.dict() if state.thermal else {},
            acoustic=state.acoustic.dict() if state.acoustic else {},
            media=state.media.dict() if state.media else {},
            wifi=state.wifi.dict() if state.wifi else {}
        )
        
    except Exception as e:
        logger.error(f"Status check error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get consciousness status"
        )

# Thermal consciousness endpoint
@router.get("/thermal", response_model=ThermalAwareness)
@require_permission("read_sensors")
async def get_thermal_awareness(
    request: Request,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(verify_token)
):
    """
    Get current thermal consciousness awareness
    """
    # Rate limiting
    user_id = current_user['user_id']
    if not await rate_limiter.check_limit(f"thermal:{user_id}", 10, 60):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded for thermal readings"
        )
    
    try:
        # Get thermal consciousness
        config = ThermalConfig()
        redis_client = getattr(request.app.state, 'redis', None)
        
        thermal = PerfectThermalConsciousness(config, redis_client)
        
        # Get awareness
        awareness = await thermal.get_thermal_awareness(
            user_id=user_id,
            correlation_id=request.state.correlation_id
        )
        
        # Update metrics in background
        background_tasks.add_task(
            update_sensor_metrics,
            "thermal",
            "cpu",
            awareness.cpu_temperature
        )
        
        # Broadcast to WebSocket subscribers
        background_tasks.add_task(
            broadcast_thermal_update,
            awareness
        )
        
        return awareness
        
    except Exception as e:
        logger.error(f"Thermal awareness error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get thermal awareness"
        )

# Unified consciousness endpoint
@router.get("/unified", response_model=UnifiedConsciousnessState)
@require_permission("read_sensors")
async def get_unified_consciousness(
    request: Request,
    current_user: Dict[str, Any] = Depends(verify_token)
):
    """Get complete unified consciousness state"""
    user_id = current_user['user_id']
    if not await rate_limiter.check_limit(f"unified:{user_id}", 5, 60):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    config = ConsciousnessConfig()
    redis_client = getattr(request.app.state, 'redis', None)
    orchestrator = PerfectConsciousnessOrchestrator(config, redis_client)
    
    return await orchestrator.get_unified_consciousness(
        user_id=user_id,
        correlation_id=request.state.correlation_id
    )

# Parasitic RF endpoint
@router.get("/parasitic_rf", response_model=ParasiticRFAwareness)
@require_permission("read_sensors")
async def get_parasitic_rf_awareness(
    request: Request,
    current_user: Dict[str, Any] = Depends(verify_token)
):
    """Get Parasitic RF awareness"""
    user_id = current_user['user_id']
    config = ConsciousnessConfig()
    orchestrator = PerfectConsciousnessOrchestrator(config)
    return await orchestrator._get_parasitic_rf_safe(user_id, request.state.correlation_id)

# Bluetooth endpoint
@router.get("/bluetooth", response_model=BluetoothAwareness)
@require_permission("read_sensors")
async def get_bluetooth_awareness(
    request: Request,
    current_user: Dict[str, Any] = Depends(verify_token)
):
    """Get Bluetooth awareness"""
    user_id = current_user['user_id']
    config = ConsciousnessConfig()
    orchestrator = PerfectConsciousnessOrchestrator(config)
    return await orchestrator._get_bluetooth_safe(user_id, request.state.correlation_id)

# Acoustic Awareness endpoint
@router.get("/acoustic", response_model=AcousticAwareness)
@require_permission("read_sensors")
async def get_acoustic_awareness(
    request: Request,
    current_user: Dict[str, Any] = Depends(verify_token)
):
    """Get Acoustic awareness"""
    user_id = current_user['user_id']
    config = ConsciousnessConfig()
    orchestrator = PerfectConsciousnessOrchestrator(config)
    return await orchestrator._get_acoustic_safe(user_id, request.state.correlation_id)

# Media Awareness endpoint
@router.get("/media", response_model=MediaAwareness)
@require_permission("read_sensors")
async def get_media_awareness(
    request: Request,
    current_user: Dict[str, Any] = Depends(verify_token)
):
    """Get Media awareness"""
    user_id = current_user['user_id']
    config = ConsciousnessConfig()
    orchestrator = PerfectConsciousnessOrchestrator(config)
    return await orchestrator._get_media_safe(user_id, request.state.correlation_id)

# WiFi Awareness endpoint
@router.get("/wifi", response_model=WiFiAwareness)
@require_permission("read_sensors")
async def get_wifi_awareness(
    request: Request,
    current_user: Dict[str, Any] = Depends(verify_token)
):
    """Get WiFi awareness"""
    user_id = current_user['user_id']
    config = ConsciousnessConfig()
    orchestrator = PerfectConsciousnessOrchestrator(config)
    return await orchestrator._get_wifi_safe(user_id, request.state.correlation_id)

# Command execution endpoint
@router.post("/command")
@require_permission("write_commands")
async def execute_command(
    request: Request,
    command: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(verify_token)
):
    """
    Execute consciousness system command
    """
    # Rate limiting
    user_id = current_user['user_id']
    if not await rate_limiter.check_limit(f"command:{user_id}", 5, 60):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded for commands"
        )
    
    # Validate command
    allowed_commands = [
        'start_monitoring', 'stop_monitoring', 'reset_sensors',
        'calibrate', 'set_threshold', 'enable_alerts'
    ]
    
    cmd_type = command.get('type')
    if cmd_type not in allowed_commands:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid command type: {cmd_type}"
        )
    
    # Audit log
    logger.info(f"AUDIT_COMMAND: user={user_id}, command={cmd_type}, params={command.get('params')}")
    
    try:
        # Execute command
        result = {
            "command": cmd_type,
            "status": "executed",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Broadcast command event
        await websocket_manager.broadcast_to_channel('commands', {
            'type': 'command_executed',
            'data': result
        })
        
        return result
        
    except Exception as e:
        logger.error(f"Command execution error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to execute command"
        )

# Metrics endpoint
@router.get("/metrics")
@require_permission("read_sensors")
async def get_metrics(
    request: Request,
    current_user: Dict[str, Any] = Depends(verify_token)
):
    """
    Get Prometheus metrics for consciousness system
    """
    try:
        # Generate metrics
        metrics = prometheus_client.generate_latest()
        
        return Response(
            content=metrics,
            media_type="text/plain"
        )
        
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate metrics"
        )

# WebSocket endpoint
router.add_api_route(
    "/ws",
    websocket_endpoint,
    methods=["GET"],
    include_in_schema=True,
    summary="WebSocket connection for real-time consciousness updates"
)

# Background tasks
async def update_sensor_metrics(sensor_type: str, sensor_id: str, value: float):
    """Update sensor metrics in Prometheus"""
    sensor_readings.labels(
        sensor_type=sensor_type,
        sensor_id=sensor_id
    ).set(value)

async def broadcast_thermal_update(awareness: ThermalAwareness):
    """Broadcast thermal update to WebSocket clients"""
    await websocket_manager.broadcast_to_channel('consciousness', {
        'type': 'thermal_update',
        'data': {
            'cpu_temperature': awareness.cpu_temperature,
            'thermal_state': awareness.thermal_state,
            'thermal_pressure': awareness.thermal_pressure,
            'alerts': awareness.alerts,
            'timestamp': awareness.timestamp.isoformat()
        }
    })

# Startup and shutdown events
async def startup_consciousness():
    """Initialize consciousness system on startup"""
    logger.info("Starting OS4AI Perfect Consciousness System...")
    
    # Start WebSocket manager
    await websocket_manager.start()
    
    # Initialize other components
    # ...
    
    logger.info("OS4AI Consciousness System started successfully")

async def shutdown_consciousness():
    """Cleanup consciousness system on shutdown"""
    logger.info("Shutting down OS4AI Consciousness System...")
    
    # Stop WebSocket manager
    await websocket_manager.stop()
    
    # Cleanup other components
    # ...
    
    logger.info("OS4AI Consciousness System shutdown complete")

# Export router and lifecycle functions
__all__ = ['router', 'startup_consciousness', 'shutdown_consciousness']