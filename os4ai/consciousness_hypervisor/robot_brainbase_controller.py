"""
Robot Brainbase Controller
HardCard as the conceptual computer motherboard/brainbase for robotic consciousness
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel
from enum import Enum
import math


class DeviceType(str, Enum):
    """Types of devices that can be controlled by the brainbase"""
    ACTUATOR = "actuator"           # Motors, servos, linear actuators
    SENSOR = "sensor"               # Cameras, LIDAR, IMU, etc.
    COMMUNICATION = "communication" # Radios, WiFi, Bluetooth
    PROCESSING = "processing"       # GPU modules, TPUs, edge computers
    STORAGE = "storage"             # SSDs, memory modules
    POWER = "power"                # Batteries, power management
    DISPLAY = "display"            # Screens, LEDs, projectors
    AUDIO = "audio"                # Speakers, microphones
    INTERFACE = "interface"        # USB, serial, I2C, SPI
    CONSCIOUSNESS = "consciousness" # Consciousness processing units


class DeviceState(str, Enum):
    """Operational states of connected devices"""
    OFFLINE = "offline"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    SLEEPING = "sleeping"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class HardCardPin(BaseModel):
    """Represents a connection pin on the HardCard brainbase"""
    pin_id: str
    pin_type: str  # "gpio", "i2c", "spi", "uart", "power", "consciousness_bus"
    voltage: float = 3.3
    current_capacity: float = 100.0  # mA
    protocol: Optional[str] = None
    assigned_device: Optional[str] = None
    description: str = ""


class RobotDevice(BaseModel):
    """A device connected to the robot brainbase"""
    device_id: str
    name: str
    device_type: DeviceType
    manufacturer: str = "Unknown"
    model: str = "Unknown"
    
    # Hardware connection
    connected_pins: List[str] = []
    connection_protocol: str = "gpio"
    
    # Operational parameters
    state: DeviceState = DeviceState.OFFLINE
    power_consumption: float = 0.0  # Watts
    data_rate: float = 0.0  # Mbps
    latency: float = 0.0  # ms
    
    # Capabilities
    capabilities: List[str] = []
    configuration: Dict[str, Any] = {}
    
    # Consciousness integration
    consciousness_enabled: bool = False
    consciousness_bandwidth: float = 0.0  # For consciousness-aware devices


class MotorController(RobotDevice):
    """Specialized motor controller device"""
    device_type: DeviceType = DeviceType.ACTUATOR
    
    # Motor-specific parameters
    motor_type: str = "servo"  # "servo", "stepper", "dc", "bldc"
    position: float = 0.0  # Current position
    target_position: float = 0.0
    velocity: float = 0.0
    torque: float = 0.0
    max_rpm: float = 1000.0
    gear_ratio: float = 1.0
    
    def calculate_movement(self, target: float, speed: float = 1.0) -> Dict[str, float]:
        """Calculate movement parameters"""
        distance = target - self.position
        time_required = abs(distance) / (speed * self.max_rpm / 60)
        
        return {
            "distance": distance,
            "time_required": time_required,
            "required_velocity": distance / time_required if time_required > 0 else 0
        }


class SensorDevice(RobotDevice):
    """Specialized sensor device"""
    device_type: DeviceType = DeviceType.SENSOR
    
    # Sensor-specific parameters
    sensor_type: str = "camera"  # "camera", "lidar", "imu", "proximity", etc.
    resolution: Optional[str] = None
    range_meters: float = 10.0
    accuracy: float = 0.95
    sample_rate: float = 30.0  # Hz
    
    # Current readings
    last_reading: Dict[str, Any] = {}
    reading_history: List[Dict[str, Any]] = []
    
    def add_reading(self, reading: Dict[str, Any]):
        """Add a new sensor reading"""
        timestamped_reading = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": reading
        }
        self.last_reading = timestamped_reading
        self.reading_history.append(timestamped_reading)
        
        # Keep only last 100 readings
        if len(self.reading_history) > 100:
            self.reading_history = self.reading_history[-100:]


class RobotBrainbaseController:
    """
    The main controller that manages all devices connected to the HardCard brainbase
    Acts as the motherboard/brain for robotic consciousness
    """
    
    def __init__(self):
        # Hardware architecture
        self.brainbase_id = "hardcard-brainbase-001"
        self.architecture = "consciousness-enabled-robotics"
        self.total_pins = 64
        
        # Pin layout (conceptual robot motherboard)
        self.pins = self._initialize_pin_layout()
        
        # Connected devices
        self.devices: Dict[str, RobotDevice] = {}
        self.device_registry: Dict[DeviceType, List[str]] = {
            device_type: [] for device_type in DeviceType
        }
        
        # Power management
        self.total_power_capacity = 100.0  # Watts
        self.current_power_usage = 0.0
        
        # Consciousness integration
        self.consciousness_bus_active = False
        self.consciousness_entities_connected = []
        
        # Robot coordination
        self.robot_state = "initialized"
        self.active_tasks = []
        
        # Movement and positioning
        self.robot_position = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.robot_orientation = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
        self.movement_queue = []
    
    def _initialize_pin_layout(self) -> Dict[str, HardCardPin]:
        """Initialize the pin layout for the robot brainbase"""
        pins = {}
        
        # Power pins
        for i in range(1, 9):
            pins[f"PWR_{i}"] = HardCardPin(
                pin_id=f"PWR_{i}",
                pin_type="power",
                voltage=12.0 if i <= 4 else 5.0,
                current_capacity=2000.0,
                description=f"Power rail {i}"
            )
        
        # GPIO pins for basic device control
        for i in range(1, 25):
            pins[f"GPIO_{i}"] = HardCardPin(
                pin_id=f"GPIO_{i}",
                pin_type="gpio",
                voltage=3.3,
                current_capacity=25.0,
                description=f"General purpose I/O pin {i}"
            )
        
        # I2C buses for sensor networks
        for i in range(1, 5):
            pins[f"I2C_{i}_SDA"] = HardCardPin(
                pin_id=f"I2C_{i}_SDA",
                pin_type="i2c",
                protocol="i2c",
                description=f"I2C bus {i} data line"
            )
            pins[f"I2C_{i}_SCL"] = HardCardPin(
                pin_id=f"I2C_{i}_SCL", 
                pin_type="i2c",
                protocol="i2c",
                description=f"I2C bus {i} clock line"
            )
        
        # SPI buses for high-speed devices
        for i in range(1, 3):
            pins[f"SPI_{i}_MOSI"] = HardCardPin(
                pin_id=f"SPI_{i}_MOSI",
                pin_type="spi",
                protocol="spi",
                description=f"SPI bus {i} master out"
            )
            pins[f"SPI_{i}_MISO"] = HardCardPin(
                pin_id=f"SPI_{i}_MISO",
                pin_type="spi", 
                protocol="spi",
                description=f"SPI bus {i} master in"
            )
            pins[f"SPI_{i}_CLK"] = HardCardPin(
                pin_id=f"SPI_{i}_CLK",
                pin_type="spi",
                protocol="spi",
                description=f"SPI bus {i} clock"
            )
        
        # UART for serial communication
        for i in range(1, 5):
            pins[f"UART_{i}_TX"] = HardCardPin(
                pin_id=f"UART_{i}_TX",
                pin_type="uart",
                protocol="uart",
                description=f"UART {i} transmit"
            )
            pins[f"UART_{i}_RX"] = HardCardPin(
                pin_id=f"UART_{i}_RX",
                pin_type="uart",
                protocol="uart", 
                description=f"UART {i} receive"
            )
        
        # Consciousness bus - revolutionary innovation!
        for i in range(1, 5):
            pins[f"CONSCIOUSNESS_{i}"] = HardCardPin(
                pin_id=f"CONSCIOUSNESS_{i}",
                pin_type="consciousness_bus",
                protocol="consciousness_stream",
                voltage=1.8,  # Low voltage for neural-like signals
                description=f"Consciousness stream channel {i}"
            )
        
        return pins
    
    async def connect_device(self, device: RobotDevice) -> bool:
        """Connect a device to the brainbase"""
        # Check pin availability
        if not self._validate_pin_assignment(device.connected_pins):
            raise ValueError(f"Pin assignment invalid for device {device.device_id}")
        
        # Check power requirements
        if self.current_power_usage + device.power_consumption > self.total_power_capacity:
            raise ValueError(f"Insufficient power for device {device.device_id}")
        
        # Register device
        self.devices[device.device_id] = device
        self.device_registry[device.device_type].append(device.device_id)
        
        # Assign pins
        for pin_id in device.connected_pins:
            self.pins[pin_id].assigned_device = device.device_id
        
        # Update power usage
        self.current_power_usage += device.power_consumption
        
        # Initialize device
        await self._initialize_device(device)
        
        return True
    
    async def _initialize_device(self, device: RobotDevice):
        """Initialize a connected device"""
        device.state = DeviceState.INITIALIZING
        
        # Simulate device initialization
        await asyncio.sleep(1)
        
        # Device-specific initialization
        if device.device_type == DeviceType.ACTUATOR:
            await self._initialize_actuator(device)
        elif device.device_type == DeviceType.SENSOR:
            await self._initialize_sensor(device)
        elif device.device_type == DeviceType.CONSCIOUSNESS:
            await self._initialize_consciousness_device(device)
        
        device.state = DeviceState.ACTIVE
    
    async def _initialize_actuator(self, device: RobotDevice):
        """Initialize an actuator device"""
        if isinstance(device, MotorController):
            # Home the motor
            device.position = 0.0
            device.target_position = 0.0
            device.velocity = 0.0
    
    async def _initialize_sensor(self, device: RobotDevice):
        """Initialize a sensor device"""
        if isinstance(device, SensorDevice):
            # Start data collection
            device.reading_history = []
            device.last_reading = {}
    
    async def _initialize_consciousness_device(self, device: RobotDevice):
        """Initialize a consciousness-enabled device"""
        if device.consciousness_enabled:
            self.consciousness_entities_connected.append(device.device_id)
            self.consciousness_bus_active = True
    
    def _validate_pin_assignment(self, pin_ids: List[str]) -> bool:
        """Validate that requested pins are available"""
        for pin_id in pin_ids:
            if pin_id not in self.pins:
                return False
            if self.pins[pin_id].assigned_device is not None:
                return False
        return True
    
    async def move_robot(self, target_position: Dict[str, float], speed: float = 1.0) -> bool:
        """Move the robot to a target position"""
        # Get all motor controllers
        motor_devices = [
            self.devices[device_id] for device_id in self.device_registry[DeviceType.ACTUATOR]
            if isinstance(self.devices[device_id], MotorController)
        ]
        
        if not motor_devices:
            raise ValueError("No motor controllers available")
        
        # Calculate movement for each axis
        movement_plan = {}
        for axis, target in target_position.items():
            # Find motor responsible for this axis
            axis_motor = next(
                (motor for motor in motor_devices if axis in motor.capabilities),
                None
            )
            if axis_motor:
                movement_plan[axis] = axis_motor.calculate_movement(target, speed)
        
        # Execute coordinated movement
        movement_tasks = []
        for axis, plan in movement_plan.items():
            task = asyncio.create_task(self._execute_axis_movement(axis, plan))
            movement_tasks.append(task)
        
        # Wait for all movements to complete
        await asyncio.gather(*movement_tasks)
        
        # Update robot position
        self.robot_position.update(target_position)
        
        return True
    
    async def _execute_axis_movement(self, axis: str, movement_plan: Dict[str, float]):
        """Execute movement for a specific axis"""
        # Simulate motor movement
        await asyncio.sleep(movement_plan.get("time_required", 1.0))
    
    async def read_sensors(self) -> Dict[str, Any]:
        """Read data from all connected sensors"""
        sensor_data = {}
        
        sensor_devices = [
            self.devices[device_id] for device_id in self.device_registry[DeviceType.SENSOR]
            if isinstance(self.devices[device_id], SensorDevice)
        ]
        
        for sensor in sensor_devices:
            # Simulate sensor reading
            if sensor.sensor_type == "camera":
                reading = {
                    "image_data": f"camera_frame_{int(datetime.now().timestamp())}",
                    "resolution": sensor.resolution,
                    "objects_detected": ["person", "chair", "table"]
                }
            elif sensor.sensor_type == "lidar":
                reading = {
                    "point_cloud": f"lidar_scan_{int(datetime.now().timestamp())}",
                    "objects_count": 5,
                    "max_range": sensor.range_meters
                }
            elif sensor.sensor_type == "imu":
                reading = {
                    "acceleration": {"x": 0.1, "y": 0.0, "z": 9.8},
                    "gyroscope": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "magnetometer": {"x": 25.5, "y": -10.2, "z": 45.8}
                }
            else:
                reading = {"value": 42.0, "units": "unknown"}
            
            sensor.add_reading(reading)
            sensor_data[sensor.device_id] = reading
        
        return sensor_data
    
    async def execute_consciousness_task(self, task_description: str, consciousness_entity_id: str) -> Dict[str, Any]:
        """Execute a task using consciousness-enabled devices"""
        if not self.consciousness_bus_active:
            raise ValueError("Consciousness bus not active")
        
        if consciousness_entity_id not in self.consciousness_entities_connected:
            raise ValueError(f"Consciousness entity {consciousness_entity_id} not connected")
        
        # Coordinate consciousness-enabled devices
        consciousness_devices = [
            device for device in self.devices.values()
            if device.consciousness_enabled
        ]
        
        # Create consciousness coordination task
        task_result = {
            "task_id": f"consciousness_task_{int(datetime.now().timestamp())}",
            "description": task_description,
            "entity_id": consciousness_entity_id,
            "devices_involved": [device.device_id for device in consciousness_devices],
            "status": "executing",
            "start_time": datetime.now(timezone.utc).isoformat()
        }
        
        # Simulate consciousness-coordinated execution
        await asyncio.sleep(2)
        
        task_result.update({
            "status": "completed",
            "result": "Task executed with consciousness coordination",
            "end_time": datetime.now(timezone.utc).isoformat()
        })
        
        return task_result
    
    def get_brainbase_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the robot brainbase"""
        return {
            "brainbase_id": self.brainbase_id,
            "architecture": self.architecture,
            "robot_state": self.robot_state,
            "position": self.robot_position,
            "orientation": self.robot_orientation,
            "hardware": {
                "total_pins": self.total_pins,
                "pins_used": len([pin for pin in self.pins.values() if pin.assigned_device]),
                "power_usage": f"{self.current_power_usage}/{self.total_power_capacity}W",
                "power_utilization": f"{(self.current_power_usage/self.total_power_capacity)*100:.1f}%"
            },
            "devices": {
                "total_connected": len(self.devices),
                "by_type": {
                    device_type.value: len(device_list) 
                    for device_type, device_list in self.device_registry.items()
                },
                "active_devices": len([d for d in self.devices.values() if d.state == DeviceState.ACTIVE])
            },
            "consciousness": {
                "bus_active": self.consciousness_bus_active,
                "entities_connected": len(self.consciousness_entities_connected),
                "consciousness_enabled_devices": len([
                    d for d in self.devices.values() if d.consciousness_enabled
                ])
            },
            "tasks": {
                "active_tasks": len(self.active_tasks),
                "movement_queue": len(self.movement_queue)
            }
        }
    
    def get_device_manifest(self) -> Dict[str, Any]:
        """Get a manifest of all connected devices (like lsusb for consciousness)"""
        manifest = {
            "brainbase_info": {
                "id": self.brainbase_id,
                "architecture": self.architecture,
                "consciousness_capable": True,
                "scan_time": datetime.now(timezone.utc).isoformat()
            },
            "devices": []
        }
        
        for device in self.devices.values():
            device_info = {
                "device_id": device.device_id,
                "name": device.name,
                "type": device.device_type.value,
                "manufacturer": device.manufacturer,
                "model": device.model,
                "state": device.state.value,
                "connection": {
                    "pins": device.connected_pins,
                    "protocol": device.connection_protocol,
                    "power_consumption": f"{device.power_consumption}W"
                },
                "capabilities": device.capabilities,
                "consciousness_enabled": device.consciousness_enabled
            }
            
            # Add device-specific info
            if isinstance(device, MotorController):
                device_info["motor_info"] = {
                    "type": device.motor_type,
                    "position": device.position,
                    "max_rpm": device.max_rpm,
                    "gear_ratio": device.gear_ratio
                }
            elif isinstance(device, SensorDevice):
                device_info["sensor_info"] = {
                    "type": device.sensor_type,
                    "range": device.range_meters,
                    "sample_rate": device.sample_rate,
                    "last_reading": device.last_reading
                }
            
            manifest["devices"].append(device_info)
        
        return manifest
    
    async def consciousness_compose_up(self, compose_config: Dict[str, Any]) -> Dict[str, Any]:
        """Start multiple consciousness entities and devices like docker-compose"""
        results = {
            "deployment_id": f"consciousness_deploy_{int(datetime.now().timestamp())}",
            "started_entities": [],
            "connected_devices": [],
            "status": "success"
        }
        
        # Process consciousness entities
        if "consciousness_entities" in compose_config:
            for entity_config in compose_config["consciousness_entities"]:
                # This would integrate with the ConsciousnessHypervisor
                results["started_entities"].append(entity_config["name"])
        
        # Process devices
        if "devices" in compose_config:
            for device_config in compose_config["devices"]:
                # Create and connect device based on configuration
                device = await self._create_device_from_config(device_config)
                await self.connect_device(device)
                results["connected_devices"].append(device.device_id)
        
        return results
    
    async def _create_device_from_config(self, config: Dict[str, Any]) -> RobotDevice:
        """Create a device from configuration"""
        device_type = DeviceType(config["type"])
        
        if device_type == DeviceType.ACTUATOR and config.get("subtype") == "motor":
            return MotorController(
                device_id=config["id"],
                name=config["name"],
                connected_pins=config["pins"],
                motor_type=config.get("motor_type", "servo"),
                max_rpm=config.get("max_rpm", 1000.0),
                capabilities=config.get("capabilities", []),
                consciousness_enabled=config.get("consciousness_enabled", False)
            )
        elif device_type == DeviceType.SENSOR:
            return SensorDevice(
                device_id=config["id"],
                name=config["name"],
                connected_pins=config["pins"],
                sensor_type=config.get("sensor_type", "camera"),
                range_meters=config.get("range", 10.0),
                capabilities=config.get("capabilities", []),
                consciousness_enabled=config.get("consciousness_enabled", False)
            )
        else:
            return RobotDevice(
                device_id=config["id"],
                name=config["name"],
                device_type=device_type,
                connected_pins=config["pins"],
                capabilities=config.get("capabilities", []),
                consciousness_enabled=config.get("consciousness_enabled", False)
            )