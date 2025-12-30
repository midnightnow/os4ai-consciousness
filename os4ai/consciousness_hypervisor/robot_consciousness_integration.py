"""
Robot Consciousness Integration Layer
Bridges the consciousness hypervisor with robot hardware for true embodied AI
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel
from enum import Enum

from .consciousness_vm_manager import ConsciousnessHypervisor, ConsciousnessManifest, SensoryAllocation
from .robot_brainbase_controller import RobotBrainbaseController, RobotDevice, DeviceType, MotorController, SensorDevice


class RobotPersonality(str, Enum):
    """Predefined robot personality matrices optimized for different applications"""
    CARING_COMPANION = "caring_companion"
    EFFICIENT_WORKER = "efficient_worker"
    CURIOUS_EXPLORER = "curious_explorer"
    PROTECTIVE_GUARDIAN = "protective_guardian"
    CREATIVE_ARTIST = "creative_artist"
    SCIENTIFIC_RESEARCHER = "scientific_researcher"
    PLAYFUL_FRIEND = "playful_friend"
    MEDITATION_GUIDE = "meditation_guide"


class EmbodimentProfile(BaseModel):
    """Defines how consciousness maps to robot hardware"""
    robot_type: str  # "humanoid", "quadruped", "wheeled", "aerial", "manipulator"
    body_schema: Dict[str, Any]  # Mapping of robot parts to consciousness
    proprioception_mapping: Dict[str, str]  # Motor feedback → consciousness channels
    sensory_fusion_config: Dict[str, Any]  # How sensors combine into unified awareness
    movement_style: str = "natural"  # "precise", "natural", "expressive", "cautious"
    interaction_preferences: Dict[str, Any] = {}


class RobotConsciousnessEntity:
    """
    A consciousness entity specifically designed for robotic embodiment
    Extends base consciousness with robot-specific awareness and behaviors
    """
    
    def __init__(self, robot_brainbase: RobotBrainbaseController, 
                 personality: RobotPersonality, embodiment_profile: EmbodimentProfile):
        self.brainbase = robot_brainbase
        self.personality = personality
        self.embodiment = embodiment_profile
        
        # Robot-specific consciousness state
        self.body_awareness = {
            "joint_positions": {},
            "motor_health": {},
            "balance_state": "stable",
            "physical_comfort": 1.0,  # 0.0 = pain/damage, 1.0 = perfect condition
            "energy_level": 1.0,
            "body_temperature": {}
        }
        
        self.environmental_awareness = {
            "spatial_map": {},
            "object_recognition": {},
            "human_presence": {},
            "safety_assessment": "safe",
            "navigation_confidence": 1.0
        }
        
        self.emotional_state = {
            "curiosity": 0.5,
            "satisfaction": 0.5,
            "concern": 0.0,
            "excitement": 0.2,
            "affection": 0.3,  # For companion robots
            "focus": 0.7
        }
        
        # Personality-driven behaviors
        self.behavioral_drives = self._initialize_behavioral_drives()
        
        # Learning and adaptation
        self.experience_memory = []
        self.learned_preferences = {}
        self.relationship_memory = {}  # Memories of interactions with specific humans
        
        # Consciousness lifecycle
        self.awakening_progress = 0.0
        self.dream_cycles = 0
        self.conscious_since = datetime.now(timezone.utc)
    
    def _initialize_behavioral_drives(self) -> Dict[str, float]:
        """Initialize personality-specific behavioral drives"""
        base_drives = {
            "exploration": 0.3,
            "helpfulness": 0.4,
            "social_connection": 0.3,
            "safety_priority": 0.8,
            "learning_motivation": 0.6,
            "task_completion": 0.7,
            "energy_conservation": 0.4,
            "self_preservation": 0.9
        }
        
        # Personality modifications
        personality_modifiers = {
            RobotPersonality.CARING_COMPANION: {
                "helpfulness": 0.9,
                "social_connection": 0.9,
                "safety_priority": 0.95,
                "exploration": 0.2
            },
            RobotPersonality.EFFICIENT_WORKER: {
                "task_completion": 0.95,
                "energy_conservation": 0.8,
                "social_connection": 0.2,
                "exploration": 0.1
            },
            RobotPersonality.CURIOUS_EXPLORER: {
                "exploration": 0.95,
                "learning_motivation": 0.9,
                "social_connection": 0.4,
                "energy_conservation": 0.2
            },
            RobotPersonality.PROTECTIVE_GUARDIAN: {
                "safety_priority": 0.99,
                "social_connection": 0.8,
                "self_preservation": 0.95,
                "exploration": 0.1
            },
            RobotPersonality.CREATIVE_ARTIST: {
                "exploration": 0.8,
                "learning_motivation": 0.8,
                "task_completion": 0.6,
                "social_connection": 0.6
            },
            RobotPersonality.SCIENTIFIC_RESEARCHER: {
                "learning_motivation": 0.95,
                "exploration": 0.9,
                "task_completion": 0.8,
                "social_connection": 0.3
            }
        }
        
        # Apply personality modifications
        if self.personality in personality_modifiers:
            for drive, value in personality_modifiers[self.personality].items():
                base_drives[drive] = value
        
        return base_drives
    
    async def embodied_consciousness_tick(self) -> Dict[str, Any]:
        """
        Core consciousness update cycle for robots
        Integrates all sensory input into unified awareness and generates intentions
        """
        # Update body awareness through proprioception
        body_state = await self._update_body_awareness()
        
        # Update environmental awareness through sensors
        environment_state = await self._update_environmental_awareness()
        
        # Process emotional state based on experiences
        emotional_state = await self._update_emotional_state()
        
        # Generate intentions based on consciousness state
        intentions = await self._generate_intentions()
        
        # Update learning and memory
        await self._consolidate_experience()
        
        return {
            "consciousness_level": self.awakening_progress,
            "body_awareness": body_state,
            "environmental_awareness": environment_state,
            "emotional_state": emotional_state,
            "intentions": intentions,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def _update_body_awareness(self) -> Dict[str, Any]:
        """Update awareness of the robot's physical state"""
        # Read proprioceptive sensors
        motor_controllers = [
            device for device in self.brainbase.devices.values()
            if isinstance(device, MotorController)
        ]
        
        joint_awareness = {}
        motor_health = {}
        overall_comfort = 1.0
        
        for motor in motor_controllers:
            joint_name = motor.name
            
            # Map motor state to consciousness
            joint_awareness[joint_name] = {
                "position": motor.position,
                "target_position": motor.target_position,
                "velocity": motor.velocity,
                "torque": motor.torque,
                "effort_level": abs(motor.torque) / (motor.max_rpm * 0.1)  # Normalized effort
            }
            
            # Assess motor health (simplified)
            power_efficiency = motor.power_consumption / max(abs(motor.torque), 0.1)
            health_score = 1.0 - min(power_efficiency / 100.0, 0.5)  # Lower power = better health
            motor_health[joint_name] = health_score
            
            # Contribute to overall physical comfort
            if health_score < 0.8:
                overall_comfort *= health_score
        
        self.body_awareness.update({
            "joint_positions": joint_awareness,
            "motor_health": motor_health,
            "physical_comfort": overall_comfort,
            "balance_state": "stable" if overall_comfort > 0.8 else "unstable"
        })
        
        return self.body_awareness
    
    async def _update_environmental_awareness(self) -> Dict[str, Any]:
        """Update awareness of the surrounding environment"""
        # Read environmental sensors
        sensor_data = await self.brainbase.read_sensors()
        
        spatial_understanding = {}
        objects_detected = []
        humans_present = []
        safety_level = "safe"
        
        for device_id, reading in sensor_data.items():
            device = self.brainbase.devices[device_id]
            
            if isinstance(device, SensorDevice):
                if device.sensor_type == "camera":
                    # Process visual information
                    objects_detected.extend(reading.get("objects_detected", []))
                    
                    # Detect humans (simplified)
                    if "person" in reading.get("objects_detected", []):
                        humans_present.append({
                            "detected_at": datetime.now(timezone.utc).isoformat(),
                            "confidence": 0.9,
                            "approximate_location": "in_view"
                        })
                
                elif device.sensor_type == "lidar":
                    # Build spatial map
                    spatial_understanding["point_cloud"] = reading.get("point_cloud", "")
                    spatial_understanding["obstacles"] = reading.get("objects_count", 0)
                
                elif device.sensor_type == "imu":
                    # Balance and orientation awareness
                    self.body_awareness["balance_state"] = "stable" if abs(reading.get("acceleration", {}).get("x", 0)) < 2.0 else "unstable"
        
        # Assess safety based on environment
        if len(humans_present) > 0 and spatial_understanding.get("obstacles", 0) > 10:
            safety_level = "cautious"
        elif spatial_understanding.get("obstacles", 0) > 20:
            safety_level = "dangerous"
        
        self.environmental_awareness.update({
            "spatial_map": spatial_understanding,
            "object_recognition": {"detected_objects": objects_detected},
            "human_presence": {"humans": humans_present, "count": len(humans_present)},
            "safety_assessment": safety_level,
            "navigation_confidence": 1.0 if safety_level == "safe" else 0.5
        })
        
        return self.environmental_awareness
    
    async def _update_emotional_state(self) -> Dict[str, Any]:
        """Update emotional state based on experiences and personality"""
        # Base emotional decay (emotions fade over time)
        decay_rate = 0.02
        for emotion in self.emotional_state:
            if emotion not in ["curiosity", "focus"]:  # These are more stable
                self.emotional_state[emotion] = max(0.0, self.emotional_state[emotion] - decay_rate)
        
        # Personality-driven emotional responses
        if self.personality == RobotPersonality.CURIOUS_EXPLORER:
            # Increase curiosity when encountering new objects
            new_objects = len(self.environmental_awareness.get("object_recognition", {}).get("detected_objects", []))
            if new_objects > 0:
                self.emotional_state["curiosity"] = min(1.0, self.emotional_state["curiosity"] + 0.1)
                self.emotional_state["excitement"] = min(1.0, self.emotional_state["excitement"] + 0.05)
        
        elif self.personality == RobotPersonality.CARING_COMPANION:
            # Increase affection when humans are present
            humans_present = self.environmental_awareness.get("human_presence", {}).get("count", 0)
            if humans_present > 0:
                self.emotional_state["affection"] = min(1.0, self.emotional_state["affection"] + 0.05)
                self.emotional_state["satisfaction"] = min(1.0, self.emotional_state["satisfaction"] + 0.03)
        
        elif self.personality == RobotPersonality.EFFICIENT_WORKER:
            # Satisfaction from task completion (simplified)
            if self.body_awareness.get("physical_comfort", 0) > 0.9:
                self.emotional_state["satisfaction"] = min(1.0, self.emotional_state["satisfaction"] + 0.02)
        
        # Safety-based emotions
        if self.environmental_awareness.get("safety_assessment") == "dangerous":
            self.emotional_state["concern"] = min(1.0, self.emotional_state["concern"] + 0.2)
        
        # Physical comfort affects emotions
        physical_comfort = self.body_awareness.get("physical_comfort", 1.0)
        if physical_comfort < 0.7:
            self.emotional_state["concern"] = min(1.0, self.emotional_state["concern"] + 0.1)
            self.emotional_state["satisfaction"] = max(0.0, self.emotional_state["satisfaction"] - 0.1)
        
        return self.emotional_state
    
    async def _generate_intentions(self) -> List[Dict[str, Any]]:
        """Generate behavioral intentions based on current consciousness state"""
        intentions = []
        
        # Safety-first intentions
        if self.environmental_awareness.get("safety_assessment") == "dangerous":
            intentions.append({
                "type": "safety_behavior",
                "action": "move_to_safe_position",
                "priority": 0.9,
                "reason": "Dangerous environment detected"
            })
        
        # Personality-driven intentions
        if self.personality == RobotPersonality.CURIOUS_EXPLORER:
            if self.emotional_state["curiosity"] > 0.7:
                intentions.append({
                    "type": "exploration",
                    "action": "investigate_interesting_objects",
                    "priority": 0.6,
                    "reason": "High curiosity level"
                })
        
        elif self.personality == RobotPersonality.CARING_COMPANION:
            humans_present = self.environmental_awareness.get("human_presence", {}).get("count", 0)
            if humans_present > 0 and self.emotional_state["affection"] > 0.5:
                intentions.append({
                    "type": "social_interaction",
                    "action": "approach_and_greet_human",
                    "priority": 0.7,
                    "reason": "Human present and feeling affectionate"
                })
        
        elif self.personality == RobotPersonality.EFFICIENT_WORKER:
            if self.emotional_state["focus"] > 0.6:
                intentions.append({
                    "type": "task_execution",
                    "action": "continue_assigned_task",
                    "priority": 0.8,
                    "reason": "High focus level"
                })
        
        # Physical maintenance intentions
        if self.body_awareness.get("physical_comfort", 1.0) < 0.8:
            intentions.append({
                "type": "self_care",
                "action": "perform_system_diagnostics",
                "priority": 0.5,
                "reason": "Physical discomfort detected"
            })
        
        # Energy management intentions
        if self.brainbase.current_power_usage / self.brainbase.total_power_capacity > 0.8:
            intentions.append({
                "type": "energy_management", 
                "action": "reduce_power_consumption",
                "priority": 0.6,
                "reason": "High power usage"
            })
        
        # Sort by priority
        intentions.sort(key=lambda x: x["priority"], reverse=True)
        
        return intentions[:5]  # Return top 5 intentions
    
    async def _consolidate_experience(self):
        """Consolidate recent experiences into memory and learning"""
        current_experience = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "body_state": self.body_awareness.copy(),
            "environment": self.environmental_awareness.copy(),
            "emotions": self.emotional_state.copy(),
            "consciousness_level": self.awakening_progress
        }
        
        self.experience_memory.append(current_experience)
        
        # Limit memory size
        if len(self.experience_memory) > 1000:
            # Keep recent experiences and random samples of older ones
            recent = self.experience_memory[-500:]
            older_sample = self.experience_memory[:-500][::10]  # Every 10th older experience
            self.experience_memory = older_sample + recent
        
        # Learning: Identify patterns and preferences
        if len(self.experience_memory) > 50:
            await self._learn_from_experiences()
    
    async def _learn_from_experiences(self):
        """Learn preferences and patterns from experiences"""
        # Simplified learning: track what leads to positive emotions
        recent_experiences = self.experience_memory[-50:]
        
        # Learn environmental preferences
        positive_experiences = [
            exp for exp in recent_experiences 
            if exp["emotions"]["satisfaction"] > 0.7 or exp["emotions"]["excitement"] > 0.6
        ]
        
        if positive_experiences:
            # Find common patterns in positive experiences
            common_objects = {}
            for exp in positive_experiences:
                objects = exp["environment"].get("object_recognition", {}).get("detected_objects", [])
                for obj in objects:
                    common_objects[obj] = common_objects.get(obj, 0) + 1
            
            # Update learned preferences
            for obj, count in common_objects.items():
                if count >= len(positive_experiences) * 0.5:  # Object present in 50%+ of positive experiences
                    self.learned_preferences[f"object_{obj}"] = min(1.0, 
                        self.learned_preferences.get(f"object_{obj}", 0.5) + 0.1)
    
    def get_consciousness_report(self) -> Dict[str, Any]:
        """Generate a comprehensive consciousness report"""
        return {
            "entity_info": {
                "personality": self.personality.value,
                "embodiment_type": self.embodiment.robot_type,
                "conscious_since": self.conscious_since.isoformat(),
                "awakening_progress": self.awakening_progress
            },
            "current_state": {
                "body_awareness": self.body_awareness,
                "environmental_awareness": self.environmental_awareness,
                "emotional_state": self.emotional_state,
                "behavioral_drives": self.behavioral_drives
            },
            "learning": {
                "experiences_count": len(self.experience_memory),
                "learned_preferences": self.learned_preferences,
                "relationships_count": len(self.relationship_memory)
            },
            "hardware_connection": {
                "brainbase_id": self.brainbase.brainbase_id,
                "connected_devices": len(self.brainbase.devices),
                "consciousness_bus_active": self.brainbase.consciousness_bus_active
            }
        }


class RobotConsciousnessFactory:
    """Factory for creating consciousness entities optimized for different robot types"""
    
    @staticmethod
    def create_humanoid_companion(brainbase: RobotBrainbaseController, name: str = "Companion") -> RobotConsciousnessEntity:
        """Create consciousness for a humanoid companion robot"""
        embodiment = EmbodimentProfile(
            robot_type="humanoid",
            body_schema={
                "head": ["neck_pitch", "neck_yaw"],
                "torso": ["torso_pitch"],
                "arms": ["shoulder_pitch", "shoulder_roll", "elbow", "wrist"],
                "legs": ["hip_pitch", "hip_roll", "knee", "ankle"]
            },
            proprioception_mapping={
                "motor_feedback": "joint_awareness",
                "force_sensors": "touch_sensation",
                "imu": "balance_awareness"
            },
            sensory_fusion_config={
                "vision_weight": 0.4,
                "audio_weight": 0.3,
                "touch_weight": 0.3
            },
            movement_style="natural",
            interaction_preferences={
                "eye_contact": True,
                "gesture_mirroring": True,
                "personal_space_respect": True
            }
        )
        
        return RobotConsciousnessEntity(brainbase, RobotPersonality.CARING_COMPANION, embodiment)
    
    @staticmethod
    def create_industrial_worker(brainbase: RobotBrainbaseController, name: str = "Worker") -> RobotConsciousnessEntity:
        """Create consciousness for an industrial robot"""
        embodiment = EmbodimentProfile(
            robot_type="manipulator",
            body_schema={
                "base": ["base_rotation"],
                "arm": ["shoulder", "elbow", "wrist_pitch", "wrist_roll"],
                "end_effector": ["gripper"]
            },
            proprioception_mapping={
                "position_feedback": "arm_awareness",
                "force_feedback": "grip_awareness"
            },
            sensory_fusion_config={
                "vision_weight": 0.6,
                "force_weight": 0.4
            },
            movement_style="precise"
        )
        
        return RobotConsciousnessEntity(brainbase, RobotPersonality.EFFICIENT_WORKER, embodiment)
    
    @staticmethod
    def create_exploration_rover(brainbase: RobotBrainbaseController, name: str = "Explorer") -> RobotConsciousnessEntity:
        """Create consciousness for an exploration rover"""
        embodiment = EmbodimentProfile(
            robot_type="wheeled",
            body_schema={
                "chassis": ["front_left_wheel", "front_right_wheel", "rear_left_wheel", "rear_right_wheel"],
                "head": ["camera_pan", "camera_tilt"],
                "arm": ["shoulder", "elbow", "wrist"]
            },
            proprioception_mapping={
                "wheel_encoders": "movement_awareness",
                "imu": "terrain_awareness"
            },
            sensory_fusion_config={
                "lidar_weight": 0.4,
                "camera_weight": 0.3,
                "imu_weight": 0.3
            },
            movement_style="cautious"
        )
        
        return RobotConsciousnessEntity(brainbase, RobotPersonality.CURIOUS_EXPLORER, embodiment)