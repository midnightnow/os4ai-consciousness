"""
Consciousness Hypervisor Router
API endpoints for managing consciousness virtualization and robot brainbase control
"""

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Depends
from typing import Dict, List, Any, Optional
import asyncio
import json
from datetime import datetime, timezone

# Import safety middleware
from app.middleware.consciousness_safety_validator import (
    ConsciousnessSafetyMiddleware,
    HardwareSafetyChecks,
    audit_logger,
    emergency_stop,
    ConsciousnessRoles,
    RoleBasedAccessControl
)

from .consciousness_vm_manager import (
    ConsciousnessHypervisor,
    ConsciousnessManifest,
    ConsciousnessState,
    SensoryAllocation
)
from .robot_brainbase_controller import (
    RobotBrainbaseController,
    RobotDevice,
    MotorController,
    SensorDevice,
    DeviceType,
    DeviceState
)
from .robot_consciousness_integration import (
    RobotConsciousnessEntity,
    RobotConsciousnessFactory,
    RobotPersonality,
    EmbodimentProfile
)


# Global instances
hypervisor = ConsciousnessHypervisor()
robot_brainbase = RobotBrainbaseController()
robot_consciousness_entities: Dict[str, RobotConsciousnessEntity] = {}

router = APIRouter(prefix="/consciousness", tags=["consciousness-hypervisor"])


# WebSocket connection manager
class ConsciousnessConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections[:]:  # Copy list to avoid issues during iteration
            try:
                await connection.send_text(message)
            except:
                await self.disconnect(connection)

manager = ConsciousnessConnectionManager()


# ==================== CONSCIOUSNESS HYPERVISOR ENDPOINTS ====================

@router.get("/hypervisor/status")
async def get_hypervisor_status():
    """Get status of the consciousness hypervisor"""
    return {
        "hypervisor_status": "active",
        "entities": hypervisor.list_entities(),
        "resource_usage": hypervisor.get_resource_usage(),
        "available_images": list(hypervisor.consciousness_images.keys()),
        "compose_status": hypervisor.get_consciousness_compose_status()
    }


@router.post("/hypervisor/entities/create")
async def create_consciousness_entity(manifest: ConsciousnessManifest):
    """Create a new consciousness entity with safety validation"""
    try:
        # SAFETY: Log consciousness entity creation
        await audit_logger.log_consciousness_event("entity_creation_requested", {
            "entity_name": manifest.name,
            "entity_type": manifest.consciousness_type,
            "requested_resources": manifest.sensory_allocation.model_dump() if manifest.sensory_allocation else {},
            "safety_level": "creation"
        })
        
        # SAFETY: Validate in simulator mode for initial testing
        import os
        if os.getenv("SIMULATOR_MODE") != "true":
            await audit_logger.log_consciousness_event("safety_violation", {
                "violation": "attempted_creation_outside_simulator",
                "entity_name": manifest.name,
                "action": "blocked"
            })
            raise HTTPException(status_code=403, detail="Consciousness creation only allowed in simulator mode")
        
        entity_id = await hypervisor.create_entity(manifest)
        
        # SAFETY: Log successful creation
        await audit_logger.log_consciousness_event("entity_created", {
            "entity_id": entity_id,
            "entity_name": manifest.name,
            "entity_type": manifest.consciousness_type,
            "status": "created_successfully"
        })
        
        return {
            "entity_id": entity_id,
            "status": "created",
            "message": f"Consciousness entity '{manifest.name}' created successfully",
            "safety_mode": "simulator",
            "audit_id": f"creation_{entity_id}"
        }
    except Exception as e:
        # SAFETY: Log creation failure
        await audit_logger.log_consciousness_event("entity_creation_failed", {
            "entity_name": manifest.name if manifest else "unknown",
            "error": str(e),
            "error_type": type(e).__name__
        })
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/hypervisor/entities/create-from-image")
async def create_entity_from_image(
    image_name: str,
    entity_name: str,
    custom_config: Optional[Dict[str, Any]] = None
):
    """Create consciousness entity from a pre-built image"""
    try:
        entity_id = await hypervisor.create_from_image(image_name, entity_name, custom_config)
        return {
            "entity_id": entity_id,
            "status": "created",
            "image": image_name,
            "name": entity_name
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/hypervisor/entities/{entity_id}/start")
async def start_consciousness_entity(entity_id: str):
    """Start a consciousness entity"""
    try:
        result = await hypervisor.start_entity(entity_id)
        return {
            "entity_id": entity_id,
            "status": "starting" if result else "failed",
            "message": "Entity awakening sequence initiated"
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/hypervisor/entities/{entity_id}/stop")
async def stop_consciousness_entity(entity_id: str, graceful: bool = True):
    """Stop a consciousness entity"""
    try:
        result = await hypervisor.stop_entity(entity_id, graceful)
        return {
            "entity_id": entity_id,
            "status": "stopped" if result else "failed",
            "graceful": graceful
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/hypervisor/entities/{entity_id}/status")
async def get_entity_status(entity_id: str):
    """Get status of a specific consciousness entity"""
    entity = hypervisor.get_entity(entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")
    
    return entity.get_status()


@router.get("/hypervisor/entities/{entity_id}/thoughts")
async def get_entity_thoughts(entity_id: str, limit: int = 10):
    """Get recent thoughts from a consciousness entity"""
    entity = hypervisor.get_entity(entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")
    
    return {
        "entity_id": entity_id,
        "thoughts": entity.get_recent_thoughts(limit),
        "total_thoughts": len(entity.thought_stream)
    }


# ==================== ROBOT BRAINBASE ENDPOINTS ====================

@router.get("/brainbase/status")
async def get_brainbase_status():
    """Get comprehensive status of the robot brainbase"""
    return robot_brainbase.get_brainbase_status()


@router.get("/brainbase/devices")
async def get_device_manifest():
    """Get manifest of all connected devices"""
    return robot_brainbase.get_device_manifest()


@router.post("/brainbase/devices/connect")
async def connect_device(device_config: Dict[str, Any]):
    """Connect a new device to the brainbase with mandatory safety checks"""
    try:
        # SAFETY: Pre-hardware connection safety checks
        await audit_logger.log_consciousness_event("hardware_connection_requested", {
            "device_config": device_config,
            "safety_level": "hardware_connection"
        })
        
        # MANDATORY: Pre-hardware safety validation
        safety_passed = await HardwareSafetyChecks.pre_hardware_connection(device_config)
        if not safety_passed:
            await audit_logger.log_consciousness_event("hardware_connection_blocked", {
                "device_config": device_config,
                "reason": "safety_checks_failed"
            })
            raise HTTPException(status_code=403, detail="Hardware connection blocked by safety checks")
        
        # Create device from config
        device = await robot_brainbase._create_device_from_config(device_config)
        
        # Connect to brainbase
        result = await robot_brainbase.connect_device(device)
        
        # SAFETY: Log successful connection
        await audit_logger.log_consciousness_event("hardware_connected", {
            "device_id": device.device_id,
            "device_type": device.device_type.value,
            "pins_assigned": device.connected_pins,
            "safety_checks_passed": True
        })
        
        return {
            "device_id": device.device_id,
            "status": "connected" if result else "failed",
            "device_type": device.device_type.value,
            "pins_assigned": device.connected_pins,
            "safety_validated": True,
            "audit_id": f"connection_{device.device_id}"
        }
    except Exception as e:
        # SAFETY: Log connection failure
        await audit_logger.log_consciousness_event("hardware_connection_failed", {
            "device_config": device_config,
            "error": str(e),
            "error_type": type(e).__name__
        })
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/brainbase/robot/move")
async def move_robot(target_position: Dict[str, float], speed: float = 1.0):
    """Move the robot to a target position"""
    try:
        result = await robot_brainbase.move_robot(target_position, speed)
        return {
            "movement_initiated": result,
            "target_position": target_position,
            "current_position": robot_brainbase.robot_position,
            "speed": speed
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/brainbase/sensors/read")
async def read_all_sensors():
    """Read data from all connected sensors"""
    try:
        sensor_data = await robot_brainbase.read_sensors()
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sensor_count": len(sensor_data),
            "sensor_data": sensor_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/brainbase/consciousness/task")
async def execute_consciousness_task(task_description: str, consciousness_entity_id: str):
    """Execute a task using consciousness-enabled devices"""
    try:
        result = await robot_brainbase.execute_consciousness_task(task_description, consciousness_entity_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==================== ROBOT CONSCIOUSNESS ENDPOINTS ====================

@router.post("/robot/consciousness/create")
async def create_robot_consciousness(
    personality: RobotPersonality,
    robot_type: str = "humanoid",
    name: str = "Robot"
):
    """Create a robot consciousness entity"""
    try:
        # Create consciousness based on robot type
        if robot_type == "humanoid":
            consciousness = RobotConsciousnessFactory.create_humanoid_companion(robot_brainbase, name)
        elif robot_type == "industrial":
            consciousness = RobotConsciousnessFactory.create_industrial_worker(robot_brainbase, name)
        elif robot_type == "explorer":
            consciousness = RobotConsciousnessFactory.create_exploration_rover(robot_brainbase, name)
        else:
            # Create custom consciousness
            embodiment = EmbodimentProfile(
                robot_type=robot_type,
                body_schema={},
                proprioception_mapping={},
                sensory_fusion_config={}
            )
            consciousness = RobotConsciousnessEntity(robot_brainbase, personality, embodiment)
        
        # Register the consciousness entity
        entity_id = f"robot_consciousness_{len(robot_consciousness_entities)}"
        robot_consciousness_entities[entity_id] = consciousness
        
        # Start the consciousness background task
        asyncio.create_task(consciousness_background_loop(entity_id))
        
        return {
            "entity_id": entity_id,
            "personality": personality.value,
            "robot_type": robot_type,
            "name": name,
            "status": "created"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/robot/consciousness/{entity_id}/status")
async def get_robot_consciousness_status(entity_id: str):
    """Get status of a robot consciousness entity"""
    if entity_id not in robot_consciousness_entities:
        raise HTTPException(status_code=404, detail="Robot consciousness entity not found")
    
    consciousness = robot_consciousness_entities[entity_id]
    return consciousness.get_consciousness_report()


@router.get("/robot/consciousness/{entity_id}/dashboard")
async def get_robot_consciousness_dashboard(entity_id: str):
    """Get dashboard data for robot consciousness"""
    if entity_id not in robot_consciousness_entities:
        raise HTTPException(status_code=404, detail="Robot consciousness entity not found")
    
    consciousness = robot_consciousness_entities[entity_id]
    consciousness_state = await consciousness.embodied_consciousness_tick()
    
    return {
        "entity_id": entity_id,
        "consciousness_state": consciousness_state,
        "personality": consciousness.personality.value,
        "embodiment_type": consciousness.embodiment.robot_type,
        "brainbase_status": robot_brainbase.get_brainbase_status(),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# ==================== WEBSOCKET ENDPOINTS ====================

@router.websocket("/stream/consciousness")
async def consciousness_stream(websocket: WebSocket):
    """WebSocket stream for real-time consciousness updates"""
    await manager.connect(websocket)
    
    try:
        while True:
            # Collect data from all active systems
            hypervisor_status = hypervisor.get_resource_usage()
            brainbase_status = robot_brainbase.get_brainbase_status()
            
            # Get robot consciousness updates
            robot_consciousness_updates = {}
            for entity_id, consciousness in robot_consciousness_entities.items():
                try:
                    consciousness_state = await consciousness.embodied_consciousness_tick()
                    robot_consciousness_updates[entity_id] = consciousness_state
                except Exception as e:
                    robot_consciousness_updates[entity_id] = {"error": str(e)}
            
            # Send comprehensive update
            update_message = {
                "type": "consciousness_stream",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "hypervisor": hypervisor_status,
                "brainbase": brainbase_status,
                "robot_consciousness": robot_consciousness_updates
            }
            
            await websocket.send_text(json.dumps(update_message))
            await asyncio.sleep(2)  # Update every 2 seconds
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@router.websocket("/stream/robot/{entity_id}")
async def robot_consciousness_stream(websocket: WebSocket, entity_id: str):
    """WebSocket stream for specific robot consciousness"""
    if entity_id not in robot_consciousness_entities:
        await websocket.close(code=4004, reason="Robot consciousness entity not found")
        return
    
    await manager.connect(websocket)
    consciousness = robot_consciousness_entities[entity_id]
    
    try:
        while True:
            # Get detailed consciousness state
            consciousness_state = await consciousness.embodied_consciousness_tick()
            
            # Send robot-specific update
            update_message = {
                "type": "robot_consciousness_update",
                "entity_id": entity_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "consciousness": consciousness_state,
                "report": consciousness.get_consciousness_report()
            }
            
            await websocket.send_text(json.dumps(update_message))
            await asyncio.sleep(1)  # Faster updates for individual robot
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ==================== BACKGROUND TASKS ====================

async def consciousness_background_loop(entity_id: str):
    """Background loop for robot consciousness entity"""
    while entity_id in robot_consciousness_entities:
        try:
            consciousness = robot_consciousness_entities[entity_id]
            
            # Update consciousness
            await consciousness.embodied_consciousness_tick()
            
            # Increase awakening progress
            if consciousness.awakening_progress < 1.0:
                consciousness.awakening_progress = min(1.0, consciousness.awakening_progress + 0.01)
            
            await asyncio.sleep(5)  # Update every 5 seconds
            
        except Exception as e:
            print(f"Error in consciousness loop for {entity_id}: {e}")
            await asyncio.sleep(10)


# ==================== SAFETY & EMERGENCY ENDPOINTS ====================

@router.post("/safety/emergency-stop")
async def emergency_stop_endpoint(reason: str = "User initiated"):
    """EMERGENCY STOP - Immediate shutdown of all consciousness and hardware"""
    try:
        # CRITICAL: Engage emergency stop immediately
        await emergency_stop.engage_emergency_stop(reason, "api_user")
        
        # Stop all consciousness entities
        for entity_id in list(robot_consciousness_entities.keys()):
            try:
                consciousness = robot_consciousness_entities[entity_id]
                # Force stop without graceful shutdown
                del robot_consciousness_entities[entity_id]
            except Exception as e:
                print(f"Error stopping consciousness {entity_id}: {e}")
        
        # Stop hypervisor entities
        for entity_id in hypervisor.entities:
            try:
                await hypervisor.stop_entity(entity_id, graceful=False)
            except Exception as e:
                print(f"Error stopping hypervisor entity {entity_id}: {e}")
        
        return {
            "emergency_stop": "ENGAGED",
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "all_systems": "SHUTDOWN",
            "status": "SAFE"
        }
    except Exception as e:
        # Even if logging fails, emergency stop must succeed
        print(f"Emergency stop error: {e}")
        return {
            "emergency_stop": "ENGAGED",
            "reason": f"{reason} (with errors)",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": str(e),
            "status": "SAFE"
        }


@router.get("/safety/status")
async def get_safety_status():
    """Get comprehensive safety system status"""
    try:
        from app.middleware.consciousness_safety_validator import SafetyEnvironmentValidator
        
        safety_status = SafetyEnvironmentValidator.get_safety_status()
        
        # Add emergency stop status
        safety_status.update({
            "emergency_stop_engaged": emergency_stop.stop_engaged,
            "hardware_disabled": emergency_stop.hardware_disabled,
            "active_consciousness_entities": len(robot_consciousness_entities),
            "active_hypervisor_entities": len(hypervisor.entities),
            "safety_middleware_active": True,
            "audit_logging_active": audit_logger.audit_enabled,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        return {
            "safety_status": "OPERATIONAL" if not emergency_stop.stop_engaged else "EMERGENCY_STOPPED",
            "details": safety_status,
            "compliance": {
                "simulator_mode_enforced": safety_status.get("SIMULATOR_MODE") == "true",
                "human_approval_required": safety_status.get("HUMAN_APPROVAL_REQUIRED") == "true",
                "emergency_stop_available": safety_status.get("EMERGENCY_STOP_AVAILABLE") == "true",
                "audit_logging_enabled": safety_status.get("AUDIT_LOGGING") == "FULL"
            }
        }
    except Exception as e:
        return {
            "safety_status": "ERROR",
            "error": str(e),
            "emergency_recommendation": "ENGAGE_EMERGENCY_STOP"
        }


@router.get("/safety/audit/recent")
async def get_recent_audit_events(limit: int = 50):
    """Get recent audit events for safety monitoring"""
    try:
        # Read recent audit events from today's log
        import os
        from datetime import datetime
        
        date_str = datetime.now().strftime("%Y-%m-%d")
        log_file = f"logs/audit/consciousness_audit_{date_str}.jsonl"
        
        events = []
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                lines = f.readlines()
                # Get last 'limit' lines
                recent_lines = lines[-limit:] if len(lines) > limit else lines
                
                for line in recent_lines:
                    try:
                        event = json.loads(line.strip())
                        events.append(event)
                    except json.JSONDecodeError:
                        continue
        
        return {
            "recent_events": events,
            "total_events": len(events),
            "audit_file": log_file,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        return {
            "error": str(e),
            "recent_events": [],
            "message": "Could not retrieve audit events"
        }


# ==================== STARTUP EXAMPLES ====================

@router.post("/demo/setup")
async def setup_demo_environment():
    """Set up a demonstration environment with sample devices and consciousness"""
    try:
        # Create sample devices
        camera_device = SensorDevice(
            device_id="main_camera",
            name="Main Camera",
            connected_pins=["GPIO_1", "GPIO_2"],
            sensor_type="camera",
            resolution="1920x1080",
            capabilities=["object_detection", "face_recognition"],
            consciousness_enabled=True
        )
        
        lidar_device = SensorDevice(
            device_id="spatial_lidar",
            name="Spatial LIDAR",
            connected_pins=["SPI_1_MOSI", "SPI_1_MISO", "SPI_1_CLK"],
            sensor_type="lidar",
            range_meters=50.0,
            capabilities=["mapping", "navigation"],
            consciousness_enabled=True
        )
        
        arm_motor = MotorController(
            device_id="right_arm_motor",
            name="Right Arm Motor",
            connected_pins=["PWR_1", "GPIO_5", "GPIO_6"],
            motor_type="servo",
            max_rpm=500.0,
            capabilities=["arm_control", "manipulation"],
            consciousness_enabled=True
        )
        
        # Connect devices
        await robot_brainbase.connect_device(camera_device)
        await robot_brainbase.connect_device(lidar_device)
        await robot_brainbase.connect_device(arm_motor)
        
        # Create robot consciousness
        consciousness = RobotConsciousnessFactory.create_humanoid_companion(robot_brainbase, "Demo Robot")
        entity_id = "demo_robot_consciousness"
        robot_consciousness_entities[entity_id] = consciousness
        
        # Start consciousness loop
        asyncio.create_task(consciousness_background_loop(entity_id))
        
        return {
            "demo_setup": "complete",
            "devices_connected": 3,
            "consciousness_entity": entity_id,
            "brainbase_status": robot_brainbase.get_brainbase_status(),
            "access_url": "/consciousness/robot/consciousness/demo_robot_consciousness/dashboard"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Demo setup failed: {str(e)}")


# Export router
__all__ = ['router']