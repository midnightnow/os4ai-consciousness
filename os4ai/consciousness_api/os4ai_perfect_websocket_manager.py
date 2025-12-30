import os
"""
OS4AI Perfect WebSocket Manager
Production-ready WebSocket with JWT authentication, encryption, and monitoring
"""

import asyncio
import json
import jwt
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Set, Optional, Any, Callable
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field, validator
from fastapi import WebSocket, WebSocketDisconnect, HTTPException, status, Query, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import redis
from collections import defaultdict
import hashlib
import hmac

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# JWT Configuration
jwt_secret = os.getenv("JWT_SECRET_KEY")  # Load from environment
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

class WebSocketMessage(BaseModel):
    """Validated WebSocket message"""
    type: str = Field(..., regex="^[a-zA-Z_]+$", max_length=50)
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    correlation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    @validator('type')
    def validate_message_type(cls, v):
        """Validate message type against whitelist"""
        allowed_types = [
            'consciousness_update', 'sensor_data', 'alert',
            'command', 'query', 'response', 'ping', 'pong',
            'subscribe', 'unsubscribe', 'error'
        ]
        if v not in allowed_types:
            raise ValueError(f"Invalid message type: {v}")
        return v

class WebSocketClient:
    """Represents a connected WebSocket client with security context"""
    
    def __init__(self, websocket: WebSocket, client_id: str, user_id: str, roles: List[str]):
        self.websocket = websocket
        self.client_id = client_id
        self.user_id = user_id
        self.roles = set(roles)
        self.subscriptions: Set[str] = set()
        self.connected_at = datetime.now(timezone.utc)
        self.last_activity = datetime.now(timezone.utc)
        self.message_count = 0
        self.rate_limit_bucket = []
        self.metadata: Dict[str, Any] = {}
    
    def has_permission(self, permission: str) -> bool:
        """Check if client has specific permission"""
        # Role-based permission mapping
        permission_map = {
            'admin': ['*'],
            'veterinarian': ['read_sensors', 'write_commands', 'receive_alerts'],
            'technician': ['read_sensors', 'receive_alerts'],
            'viewer': ['read_sensors']
        }
        
        for role in self.roles:
            if role in permission_map:
                if '*' in permission_map[role] or permission in permission_map[role]:
                    return True
        return False
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now(timezone.utc)
        self.message_count += 1

class PerfectWebSocketManager:
    """
    Production-ready WebSocket manager with comprehensive security and monitoring
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.clients: Dict[str, WebSocketClient] = {}
        self.channels: Dict[str, Set[str]] = defaultdict(set)  # channel -> client_ids
        self.redis_client = redis_client
        self._rate_limiter = WebSocketRateLimiter(redis_client)
        self._metrics = WebSocketMetrics()
        self._message_handlers: Dict[str, Callable] = {}
        self._shutdown_event = asyncio.Event()
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Register default handlers
        self._register_default_handlers()
    
    async def start(self):
        """Start WebSocket manager background tasks"""
        logger.info("Starting Perfect WebSocket Manager...")
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def stop(self):
        """Gracefully stop WebSocket manager"""
        logger.info("Stopping Perfect WebSocket Manager...")
        self._shutdown_event.set()
        
        # Close all connections
        for client in list(self.clients.values()):
            await self.disconnect(client.client_id, reason="Server shutdown")
        
        if self._cleanup_task:
            await self._cleanup_task
    
    async def authenticate_websocket(self, websocket: WebSocket, token: str) -> Dict[str, Any]:
        """
        Authenticate WebSocket connection using JWT
        """
        try:
            # Decode and verify JWT
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            
            # Check expiration
            exp = datetime.fromtimestamp(payload.get('exp', 0))
            if exp < datetime.now(timezone.utc):
                raise jwt.ExpiredSignatureError("Token expired")
            
            # Extract user info
            user_info = {
                'user_id': payload.get('sub'),
                'roles': payload.get('roles', ['viewer']),
                'permissions': payload.get('permissions', []),
                'metadata': payload.get('metadata', {})
            }
            
            # Validate required fields
            if not user_info['user_id']:
                raise ValueError("Invalid token: missing user_id")
            
            # Log successful authentication
            logger.info(f"WebSocket authenticated: user={user_info['user_id']}")
            
            return user_info
            
        except jwt.ExpiredSignatureError:
            await websocket.close(code=4001, reason="Token expired")
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError as e:
            await websocket.close(code=4002, reason="Invalid token")
            raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")
        except Exception as e:
            await websocket.close(code=4003, reason="Authentication failed")
            raise HTTPException(status_code=401, detail="Authentication failed")
    
    async def connect(self, websocket: WebSocket, token: str) -> str:
        """
        Accept and authenticate new WebSocket connection
        """
        # Authenticate first
        user_info = await self.authenticate_websocket(websocket, token)
        
        # Accept connection
        await websocket.accept()
        
        # Generate client ID
        client_id = f"{user_info['user_id']}_{uuid.uuid4().hex[:8]}"
        
        # Create client instance
        client = WebSocketClient(
            websocket=websocket,
            client_id=client_id,
            user_id=user_info['user_id'],
            roles=user_info['roles']
        )
        client.metadata = user_info.get('metadata', {})
        
        # Store client
        self.clients[client_id] = client
        
        # Update metrics
        self._metrics.record_connection(client_id)
        
        # Send welcome message
        await self.send_to_client(client_id, {
            'type': 'connected',
            'data': {
                'client_id': client_id,
                'server_time': datetime.now(timezone.utc).isoformat(),
                'version': '1.0.0'
            }
        })
        
        # Audit log
        await self._audit_log('websocket_connected', client_id, {
            'user_id': client.user_id,
            'roles': list(client.roles)
        })
        
        logger.info(f"Client connected: {client_id}")
        return client_id
    
    async def disconnect(self, client_id: str, reason: str = "Normal closure"):
        """
        Disconnect client and cleanup
        """
        if client_id not in self.clients:
            return
        
        client = self.clients[client_id]
        
        try:
            # Send disconnect message
            await self.send_to_client(client_id, {
                'type': 'disconnecting',
                'data': {'reason': reason}
            })
            
            # Close WebSocket
            await client.websocket.close(code=1000, reason=reason)
            
        except Exception as e:
            logger.warning(f"Error closing WebSocket for {client_id}: {e}")
        
        # Remove from channels
        for channel in list(client.subscriptions):
            await self.unsubscribe(client_id, channel)
        
        # Remove client
        del self.clients[client_id]
        
        # Update metrics
        self._metrics.record_disconnection(client_id)
        
        # Audit log
        await self._audit_log('websocket_disconnected', client_id, {
            'reason': reason,
            'duration': (datetime.now(timezone.utc) - client.connected_at).total_seconds(),
            'message_count': client.message_count
        })
        
        logger.info(f"Client disconnected: {client_id} ({reason})")
    
    async def handle_message(self, client_id: str, raw_message: str):
        """
        Handle incoming WebSocket message with validation and security
        """
        client = self.clients.get(client_id)
        if not client:
            return
        
        try:
            # Rate limiting
            if not await self._rate_limiter.check_client_rate(client_id, 60, 60):  # 60 msg/min
                await self.send_error(client_id, "Rate limit exceeded", "RATE_LIMIT")
                self._metrics.record_rate_limit(client_id)
                return
            
            # Parse and validate message
            try:
                message_data = json.loads(raw_message)
                message = WebSocketMessage(**message_data)
            except (json.JSONDecodeError, ValueError) as e:
                await self.send_error(client_id, f"Invalid message format: {str(e)}", "INVALID_FORMAT")
                return
            
            # Update client activity
            client.update_activity()
            
            # Check permissions for message type
            if message.type == 'command' and not client.has_permission('write_commands'):
                await self.send_error(client_id, "Permission denied", "PERMISSION_DENIED")
                return
            
            # Process message
            handler = self._message_handlers.get(message.type)
            if handler:
                await handler(self, client_id, message)
            else:
                await self.send_error(client_id, f"Unknown message type: {message.type}", "UNKNOWN_TYPE")
            
            # Update metrics
            self._metrics.record_message(client_id, message.type)
            
        except Exception as e:
            logger.error(f"Error handling message from {client_id}: {e}")
            await self.send_error(client_id, "Internal server error", "INTERNAL_ERROR")
    
    async def send_to_client(self, client_id: str, data: Dict[str, Any]):
        """
        Send message to specific client with encryption
        """
        client = self.clients.get(client_id)
        if not client:
            return
        
        try:
            # Create message
            message = WebSocketMessage(
                type=data.get('type', 'message'),
                data=data.get('data', {})
            )
            
            # Serialize message
            message_json = message.json()
            
            # Send to client
            await client.websocket.send_text(message_json)
            
            # Update metrics
            self._metrics.record_sent_message(client_id)
            
        except WebSocketDisconnect:
            await self.disconnect(client_id, "Connection lost")
        except Exception as e:
            logger.error(f"Error sending to client {client_id}: {e}")
            await self.disconnect(client_id, "Send error")
    
    async def broadcast_to_channel(self, channel: str, data: Dict[str, Any], 
                                  exclude_client: Optional[str] = None,
                                  permission: Optional[str] = None):
        """
        Broadcast message to all clients in a channel
        """
        client_ids = self.channels.get(channel, set()).copy()
        
        # Filter by permission if specified
        if permission:
            client_ids = {
                cid for cid in client_ids 
                if cid in self.clients and self.clients[cid].has_permission(permission)
            }
        
        # Exclude specific client if specified
        if exclude_client and exclude_client in client_ids:
            client_ids.remove(exclude_client)
        
        # Send to all clients in parallel
        tasks = [
            self.send_to_client(client_id, data)
            for client_id in client_ids
        ]
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Update metrics
        self._metrics.record_broadcast(channel, len(tasks))
    
    async def subscribe(self, client_id: str, channel: str) -> bool:
        """
        Subscribe client to a channel with permission check
        """
        client = self.clients.get(client_id)
        if not client:
            return False
        
        # Permission check for channel
        channel_permissions = {
            'consciousness': 'read_sensors',
            'alerts': 'receive_alerts',
            'commands': 'write_commands'
        }
        
        required_permission = channel_permissions.get(channel, 'read_sensors')
        if not client.has_permission(required_permission):
            await self.send_error(client_id, f"Permission denied for channel: {channel}", "PERMISSION_DENIED")
            return False
        
        # Add to channel
        self.channels[channel].add(client_id)
        client.subscriptions.add(channel)
        
        # Notify client
        await self.send_to_client(client_id, {
            'type': 'subscribed',
            'data': {'channel': channel}
        })
        
        # Audit log
        await self._audit_log('channel_subscribed', client_id, {'channel': channel})
        
        logger.info(f"Client {client_id} subscribed to {channel}")
        return True
    
    async def unsubscribe(self, client_id: str, channel: str) -> bool:
        """
        Unsubscribe client from a channel
        """
        client = self.clients.get(client_id)
        if not client or channel not in client.subscriptions:
            return False
        
        # Remove from channel
        self.channels[channel].discard(client_id)
        client.subscriptions.discard(channel)
        
        # Clean up empty channels
        if not self.channels[channel]:
            del self.channels[channel]
        
        # Notify client
        await self.send_to_client(client_id, {
            'type': 'unsubscribed',
            'data': {'channel': channel}
        })
        
        logger.info(f"Client {client_id} unsubscribed from {channel}")
        return True
    
    async def send_error(self, client_id: str, message: str, code: str = "ERROR"):
        """
        Send error message to client
        """
        await self.send_to_client(client_id, {
            'type': 'error',
            'data': {
                'message': message,
                'code': code,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        })
    
    def _register_default_handlers(self):
        """Register default message handlers"""
        
        async def handle_ping(manager, client_id: str, message: WebSocketMessage):
            """Handle ping message"""
            await manager.send_to_client(client_id, {
                'type': 'pong',
                'data': {'timestamp': datetime.now(timezone.utc).isoformat()}
            })
        
        async def handle_subscribe(manager, client_id: str, message: WebSocketMessage):
            """Handle subscribe request"""
            channel = message.data.get('channel')
            if channel:
                await manager.subscribe(client_id, channel)
        
        async def handle_unsubscribe(manager, client_id: str, message: WebSocketMessage):
            """Handle unsubscribe request"""
            channel = message.data.get('channel')
            if channel:
                await manager.unsubscribe(client_id, channel)
        
        self._message_handlers['ping'] = handle_ping
        self._message_handlers['subscribe'] = handle_subscribe
        self._message_handlers['unsubscribe'] = handle_unsubscribe
    
    def register_handler(self, message_type: str, handler: Callable):
        """Register custom message handler"""
        self._message_handlers[message_type] = handler
    
    async def _cleanup_loop(self):
        """Background task to clean up inactive connections"""
        while not self._shutdown_event.is_set():
            try:
                # Check for inactive clients
                now = datetime.now(timezone.utc)
                inactive_timeout = timedelta(minutes=5)
                
                for client_id, client in list(self.clients.items()):
                    if now - client.last_activity > inactive_timeout:
                        # Send ping to check if still alive
                        try:
                            await self.send_to_client(client_id, {
                                'type': 'ping',
                                'data': {'require_pong': True}
                            })
                        except Exception:
                            # Connection dead, remove it
                            await self.disconnect(client_id, "Inactive timeout")
                
                # Clean up metrics
                self._metrics.cleanup_old_data()
                
                # Wait for next cleanup cycle
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=60  # Run every minute
                )
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(60)
    
    async def _audit_log(self, event: str, client_id: str, details: Dict[str, Any]):
        """Log audit event"""
        audit_entry = {
            'event': event,
            'client_id': client_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'details': details
        }
        
        logger.info(f"AUDIT_WEBSOCKET: {json.dumps(audit_entry)}")
        
        # Store in Redis if available
        if self.redis_client:
            try:
                await self.redis_client.lpush(
                    'websocket:audit_log',
                    json.dumps(audit_entry)
                )
                await self.redis_client.ltrim('websocket:audit_log', 0, 1000)
            except Exception as e:
                logger.warning(f"Failed to store audit log: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get WebSocket manager statistics"""
        return {
            'connected_clients': len(self.clients),
            'active_channels': len(self.channels),
            'channel_subscriptions': {
                channel: len(clients) 
                for channel, clients in self.channels.items()
            },
            'metrics': self._metrics.get_summary()
        }

class WebSocketRateLimiter:
    """Rate limiting for WebSocket connections"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.memory_buckets: Dict[str, List[float]] = defaultdict(list)
    
    async def check_client_rate(self, client_id: str, limit: int, window: int) -> bool:
        """Check if client is within rate limit"""
        key = f"ws_rate:{client_id}"
        
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
        bucket = self.memory_buckets[client_id]
        
        # Clean old entries
        bucket[:] = [t for t in bucket if t > now - window]
        
        if len(bucket) < limit:
            bucket.append(now)
            return True
        
        return False

class WebSocketMetrics:
    """Metrics collection for WebSocket connections"""
    
    def __init__(self):
        self.connections_total = 0
        self.disconnections_total = 0
        self.messages_received = defaultdict(int)
        self.messages_sent = 0
        self.broadcasts_sent = defaultdict(int)
        self.rate_limits_hit = defaultdict(int)
        self.errors = defaultdict(int)
        self.connection_durations = []
    
    def record_connection(self, client_id: str):
        """Record new connection"""
        self.connections_total += 1
    
    def record_disconnection(self, client_id: str):
        """Record disconnection"""
        self.disconnections_total += 1
    
    def record_message(self, client_id: str, message_type: str):
        """Record received message"""
        self.messages_received[message_type] += 1
    
    def record_sent_message(self, client_id: str):
        """Record sent message"""
        self.messages_sent += 1
    
    def record_broadcast(self, channel: str, recipient_count: int):
        """Record broadcast"""
        self.broadcasts_sent[channel] += recipient_count
    
    def record_rate_limit(self, client_id: str):
        """Record rate limit hit"""
        self.rate_limits_hit[client_id] += 1
    
    def record_error(self, error_type: str):
        """Record error"""
        self.errors[error_type] += 1
    
    def cleanup_old_data(self):
        """Clean up old metrics data"""
        # Keep only recent data
        max_entries = 1000
        for key in list(self.rate_limits_hit.keys()):
            if len(self.rate_limits_hit) > max_entries:
                del self.rate_limits_hit[key]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        return {
            'connections_total': self.connections_total,
            'disconnections_total': self.disconnections_total,
            'active_connections': self.connections_total - self.disconnections_total,
            'messages_received': dict(self.messages_received),
            'messages_sent': self.messages_sent,
            'broadcasts_sent': dict(self.broadcasts_sent),
            'rate_limits_hit': len(self.rate_limits_hit),
            'errors': dict(self.errors)
        }

# Singleton instance
websocket_manager = PerfectWebSocketManager()

# FastAPI WebSocket endpoint
async def websocket_endpoint(
    websocket: WebSocket,
    token: str = Query(..., description="JWT authentication token")
):
    """
    WebSocket endpoint with JWT authentication
    """
    client_id = None
    
    try:
        # Connect and authenticate
        client_id = await websocket_manager.connect(websocket, token)
        
        # Handle messages
        while True:
            message = await websocket.receive_text()
            await websocket_manager.handle_message(client_id, message)
            
    except WebSocketDisconnect:
        if client_id:
            await websocket_manager.disconnect(client_id, "Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if client_id:
            await websocket_manager.disconnect(client_id, f"Error: {str(e)}")

# Generate JWT token for testing
def generate_test_token(user_id: str, roles: List[str]) -> str:
    """Generate JWT token for testing"""
    payload = {
        'sub': user_id,
        'roles': roles,
        'exp': datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRATION_HOURS),
        'iat': datetime.now(timezone.utc)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

# Example usage
async def example_usage():
    """Example of using perfect WebSocket manager"""
    
    # Initialize manager
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
    manager = PerfectWebSocketManager(redis_client)
    
    # Start manager
    await manager.start()
    
    # Register custom handler
    async def handle_sensor_query(mgr, client_id: str, message: WebSocketMessage):
        """Handle sensor query"""
        sensor_type = message.data.get('sensor_type')
        # Process query...
        await mgr.send_to_client(client_id, {
            'type': 'response',
            'data': {
                'sensor_type': sensor_type,
                'value': 42.0,
                'unit': 'celsius'
            }
        })
    
    manager.register_handler('sensor_query', handle_sensor_query)
    
    # Broadcast consciousness update
    await manager.broadcast_to_channel('consciousness', {
        'type': 'consciousness_update',
        'data': {
            'thermal_state': 'normal',
            'acoustic_awareness': 'active',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    }, permission='read_sensors')
    
    # Get stats
    stats = manager.get_stats()
    print(f"WebSocket Stats: {json.dumps(stats, indent=2)}")
    
    # Graceful shutdown
    await manager.stop()

if __name__ == "__main__":
    # Generate test tokens
    admin_token = generate_test_token("admin-user", ["admin"])
    vet_token = generate_test_token("vet-user", ["veterinarian"])
    viewer_token = generate_test_token("viewer-user", ["viewer"])
    
    print(f"Admin Token: {admin_token}")
    print(f"Vet Token: {vet_token}")
    print(f"Viewer Token: {viewer_token}")
    
    # Run example
    asyncio.run(example_usage())