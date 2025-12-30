"""
OS4AI Perfect Integration Module
Orchestrates all consciousness components into a unified system
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict
import numpy as np
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import redis
from contextlib import asynccontextmanager
import uuid

# Import all perfect consciousness components
from .os4ai_perfect_thermal_integration import (
    PerfectThermalConsciousness, ThermalConfig, ThermalAwareness
)
from .os4ai_perfect_acoustic_integration import (
    PerfectAcousticConsciousness, AcousticConfig, AcousticAwareness
)
from .os4ai_perfect_media_integration import (
    PerfectMediaConsciousness, MediaConfig, MediaAwareness
)
from .os4ai_perfect_wifi_integration import (
    PerfectWiFiConsciousness, WiFiConfig, WiFiAwareness
)
from .os4ai_parasitic_rf_integration import (
    PerfectParasiticRFConsciousness, ParasiticRFConfig, ParasiticRFAwareness
)
from .os4ai_perfect_bluetooth_integration import (
    PerfectBluetoothConsciousness, BluetoothConfig, BluetoothAwareness
)
from .os4ai_perfect_websocket_manager import (
    PerfectWebSocketManager, WebSocketMessage
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConsciousnessConfig(BaseModel):
    """Unified consciousness system configuration"""
    thermal: ThermalConfig = Field(default_factory=ThermalConfig)
    acoustic: AcousticConfig = Field(default_factory=AcousticConfig)
    media: MediaConfig = Field(default_factory=MediaConfig)
    wifi: WiFiConfig = Field(default_factory=WiFiConfig)
    parasitic_rf: ParasiticRFConfig = Field(default_factory=ParasiticRFConfig)
    bluetooth: BluetoothConfig = Field(default_factory=BluetoothConfig)
    fusion_enabled: bool = True
    prediction_horizon: int = Field(300, ge=60, le=3600)  # seconds
    alert_threshold: float = Field(0.7, ge=0.0, le=1.0)
    correlation_window: int = Field(30, ge=10, le=300)  # seconds
    anomaly_detection: bool = True
    adaptive_learning: bool = True

class UnifiedConsciousnessState(BaseModel):
    """Complete consciousness state across all modalities"""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    correlation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Individual modality states
    thermal: Optional[ThermalAwareness] = None
    acoustic: Optional[AcousticAwareness] = None
    media: Optional[MediaAwareness] = None
    wifi: Optional[WiFiAwareness] = None
    parasitic_rf: Optional[ParasiticRFAwareness] = None
    bluetooth: Optional[BluetoothAwareness] = None
    
    # Fusion results
    fusion_confidence: float = Field(0.0, ge=0.0, le=1.0)
    environmental_map: Dict[str, Any] = Field(default_factory=dict)
    detected_entities: List[Dict[str, Any]] = Field(default_factory=list)
    behavioral_patterns: List[Dict[str, Any]] = Field(default_factory=list)
    
    # System status
    active_modalities: List[str] = Field(default_factory=list)
    system_health: str = Field("healthy")
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    
    # Alerts and predictions
    alerts: List[Dict[str, Any]] = Field(default_factory=list)
    predictions: List[Dict[str, Any]] = Field(default_factory=list)
    anomalies: List[Dict[str, Any]] = Field(default_factory=list)

class ConsciousnessFusion:
    """
    Advanced multi-modal consciousness fusion engine
    """
    
    def __init__(self):
        self.pattern_memory = defaultdict(list)
        self.anomaly_detector = AnomalyDetector()
        self.predictive_model = PredictiveModel()
        self.correlation_engine = CorrelationEngine()
        
    async def fuse_consciousness(
        self,
        thermal: Optional[ThermalAwareness],
        acoustic: Optional[AcousticAwareness],
        media: Optional[MediaAwareness],
        wifi: Optional[WiFiAwareness],
        parasitic_rf: Optional[ParasiticRFAwareness],
        bluetooth: Optional[BluetoothAwareness]
    ) -> Dict[str, Any]:
        """
        Fuse multi-modal consciousness data into unified understanding (6 Modalities)
        """
        # Collect all available data
        modalities = []
        if thermal:
            modalities.append(('thermal', self._extract_thermal_features(thermal)))
        if acoustic:
            modalities.append(('acoustic', self._extract_acoustic_features(acoustic)))
        if media:
            modalities.append(('media', self._extract_media_features(media)))
        if wifi:
            modalities.append(('wifi', self._extract_wifi_features(wifi)))
        if parasitic_rf:
            modalities.append(('parasitic_rf', self._extract_parasitic_rf_features(parasitic_rf)))
        if bluetooth:
            modalities.append(('bluetooth', self._extract_bluetooth_features(bluetooth)))
        
        if not modalities:
            return self._empty_fusion_result()
        
        # Perform multi-modal fusion
        fusion_result = await self._fuse_modalities(modalities)
        
        # Detect correlations
        correlations = await self.correlation_engine.find_correlations(modalities)
        fusion_result['correlations'] = correlations
        
        # Detect anomalies
        anomalies = await self.anomaly_detector.detect_anomalies(fusion_result)
        fusion_result['anomalies'] = anomalies
        
        # Generate predictions
        predictions = await self.predictive_model.predict(fusion_result)
        fusion_result['predictions'] = predictions
        
        # Update pattern memory
        self._update_pattern_memory(fusion_result)
        
        return fusion_result
    
    def _extract_thermal_features(self, thermal: ThermalAwareness) -> Dict[str, Any]:
        """Extract relevant features from thermal data"""
        return {
            'temperature': thermal.cpu_temperature,
            'thermal_state': thermal.thermal_state,
            'thermal_pressure': thermal.thermal_pressure,
            'fan_activity': np.mean(list(thermal.fan_speeds.values())) if thermal.fan_speeds else 0,
            'trend': thermal.trend,
            'alerts': len(thermal.alerts)
        }
    
    def _extract_acoustic_features(self, acoustic: AcousticAwareness) -> Dict[str, Any]:
        """Extract relevant features from acoustic data"""
        return {
            'sound_level': acoustic.sound_level_avg,
            'environment': acoustic.environment_type,
            'dominant_freq': acoustic.dominant_frequencies[0] if acoustic.dominant_frequencies else 0,
            'spatial_size': acoustic.room_properties.get('size', 'unknown'),
            'events': len(acoustic.acoustic_events),
            'alerts': len(acoustic.alerts)
        }
    
    def _extract_media_features(self, media: MediaAwareness) -> Dict[str, Any]:
        """Extract relevant features from media data"""
        return {
            'devices_count': len(media.active_devices),
            'motion_detected': media.motion_detected,
            'faces_detected': media.faces_detected,
            'patterns': len(media.detected_patterns),
            'anomaly_score': media.anomaly_score,
            'alerts': len(media.alerts)
        }
    
    def _extract_wifi_features(self, wifi: WiFiAwareness) -> Dict[str, Any]:
        """Extract relevant features from WiFi data"""
        return {
            'networks_count': wifi.networks_detected,
            'avg_signal': 0, # Placeholder if missing in WiFiAwareness
            'motion_detected': wifi.motion_detected,
            'materials': len(wifi.material_signatures),
            'interference': 0, # Placeholder
            'alerts': len(wifi.security_alerts)
        }

    def _extract_parasitic_rf_features(self, prf: ParasiticRFAwareness) -> Dict[str, Any]:
        """Extract relevant features from Parasitic RF EMI data"""
        return {
            'mains_hum': prf.mains_hum_magnitude,
            'snr': prf.signal_to_noise_ratio,
            'activity': prf.emi_activity_level,
            'motion_interference': prf.motion_interference_detected,
            'alerts': len(prf.alerts)
        }

    def _extract_bluetooth_features(self, bt: BluetoothAwareness) -> Dict[str, Any]:
        """Extract relevant features from Bluetooth data"""
        return {
            'devices': bt.devices_count,
            'active': bt.active_connections,
            'density': bt.spatial_density,
            'proximity': bt.proximity_alert,
            'alerts': len(bt.alerts)
        }
    
    async def _fuse_modalities(self, modalities: List[Tuple[str, Dict]]) -> Dict[str, Any]:
        """Perform actual fusion of modality data"""
        # Initialize fusion result
        fusion = {
            'confidence': 0.0,
            'environmental_map': {},
            'detected_entities': [],
            'behavioral_patterns': [],
            'active_modalities': [m[0] for m in modalities]
        }
        
        # Calculate fusion confidence based on modality agreement
        if len(modalities) > 1:
            fusion['confidence'] = self._calculate_fusion_confidence(modalities)
        else:
            fusion['confidence'] = 0.5  # Single modality
        
        # Build environmental map
        fusion['environmental_map'] = self._build_environmental_map(modalities)
        
        # Detect entities across modalities
        fusion['detected_entities'] = await self._detect_entities(modalities)
        
        # Extract behavioral patterns
        fusion['behavioral_patterns'] = self._extract_behavioral_patterns(modalities)
        
        return fusion
    
    def _calculate_fusion_confidence(self, modalities: List[Tuple[str, Dict]]) -> float:
        """Calculate confidence based on modality agreement"""
        # Check for motion detection agreement
        motion_votes = []
        for name, features in modalities:
            if name in ['media', 'wifi'] and 'motion_detected' in features:
                motion_votes.append(features['motion_detected'])
        
        # Check for anomaly agreement
        anomaly_scores = []
        for name, features in modalities:
            if 'anomaly_score' in features:
                anomaly_scores.append(features['anomaly_score'])
            elif 'alerts' in features and features['alerts'] > 0:
                anomaly_scores.append(0.5)  # Alerts indicate some anomaly
        
        # Calculate confidence
        confidence = 0.5  # Base confidence
        
        # Motion agreement bonus
        if len(motion_votes) > 1 and all(motion_votes):
            confidence += 0.2
        
        # Anomaly agreement bonus
        if anomaly_scores and np.std(anomaly_scores) < 0.2:
            confidence += 0.2
        
        # Multi-modality bonus
        confidence += min(0.1 * (len(modalities) - 1), 0.3)
        
        return min(confidence, 1.0)
    
    def _build_environmental_map(self, modalities: List[Tuple[str, Dict]]) -> Dict[str, Any]:
        """Build unified environmental understanding"""
        env_map = {
            'thermal': {},
            'acoustic': {},
            'electromagnetic': {},
            'visual': {},
            'combined': {}
        }
        
        for name, features in modalities:
            if name == 'thermal':
                env_map['thermal'] = {
                    'temperature': features['temperature'],
                    'state': features['thermal_state'],
                    'activity_level': features['fan_activity']
                }
            elif name == 'acoustic':
                env_map['acoustic'] = {
                    'environment': features['environment'],
                    'sound_level': features['sound_level'],
                    'space_size': features['spatial_size']
                }
            elif name == 'wifi':
                env_map['electromagnetic'] = {
                    'rf_density': features['networks_count'],
                    'signal_quality': features['avg_signal'],
                    'interference': features['interference']
                }
            elif name == 'media':
                env_map['visual'] = {
                    'devices': features['devices_count'],
                    'activity': features['motion_detected'],
                    'occupancy': features['faces_detected']
                }
        
        # Combined understanding
        env_map['combined'] = {
            'activity_level': self._assess_activity_level(modalities),
            'occupancy_confidence': self._assess_occupancy(modalities),
            'environmental_stress': self._assess_environmental_stress(modalities)
        }
        
        return env_map
    
    async def _detect_entities(self, modalities: List[Tuple[str, Dict]]) -> List[Dict[str, Any]]:
        """Detect entities across modalities"""
        entities = []
        
        # Check for human presence
        human_indicators = 0
        for name, features in modalities:
            if name == 'media' and features.get('faces_detected', 0) > 0:
                human_indicators += 2  # Strong indicator
            if name == 'acoustic' and features.get('environment') == 'normal':
                human_indicators += 1
            if name == 'wifi' and features.get('motion_detected'):
                human_indicators += 1
        
        if human_indicators >= 2:
            entities.append({
                'type': 'human',
                'confidence': min(human_indicators / 4, 1.0),
                'detected_by': [m[0] for m in modalities if self._modality_detects_human(m)]
            })
        
        # Check for devices
        device_count = 0
        for name, features in modalities:
            if name == 'media':
                device_count += features.get('devices_count', 0)
            if name == 'wifi':
                device_count += features.get('networks_count', 0)
        
        if device_count > 0:
            entities.append({
                'type': 'electronic_devices',
                'count': device_count,
                'confidence': 0.9
            })
        
        return entities
    
    def _modality_detects_human(self, modality: Tuple[str, Dict]) -> bool:
        """Check if modality detected human presence"""
        name, features = modality
        if name == 'media' and features.get('faces_detected', 0) > 0:
            return True
        if name == 'acoustic' and features.get('environment') in ['normal', 'noisy']:
            return True
        if name == 'wifi' and features.get('motion_detected'):
            return True
        return False
    
    def _extract_behavioral_patterns(self, modalities: List[Tuple[str, Dict]]) -> List[Dict[str, Any]]:
        """Extract behavioral patterns from multi-modal data"""
        patterns = []
        
        # Activity pattern
        activity_level = self._assess_activity_level(modalities)
        if activity_level > 0.7:
            patterns.append({
                'type': 'high_activity',
                'confidence': activity_level,
                'description': 'Elevated activity detected across multiple sensors'
            })
        
        # Thermal-acoustic correlation
        thermal_data = next((m[1] for m in modalities if m[0] == 'thermal'), None)
        acoustic_data = next((m[1] for m in modalities if m[0] == 'acoustic'), None)
        
        if thermal_data and acoustic_data:
            if thermal_data['thermal_state'] == 'hot' and acoustic_data['environment'] == 'noisy':
                patterns.append({
                    'type': 'system_stress',
                    'confidence': 0.8,
                    'description': 'High thermal load correlated with noisy environment'
                })
        
        return patterns
    
    def _assess_activity_level(self, modalities: List[Tuple[str, Dict]]) -> float:
        """Assess overall activity level"""
        activity_scores = []
        
        for name, features in modalities:
            if name == 'thermal':
                # High temperature indicates activity
                temp_score = min(features['temperature'] / 100, 1.0)
                activity_scores.append(temp_score)
            elif name == 'acoustic':
                # Sound level indicates activity
                sound_score = min(features['sound_level'] / 80, 1.0)
                activity_scores.append(sound_score)
            elif name == 'media' and features['motion_detected']:
                activity_scores.append(1.0)
            elif name == 'wifi' and features['motion_detected']:
                activity_scores.append(0.8)
        
        return np.mean(activity_scores) if activity_scores else 0.0
    
    def _assess_occupancy(self, modalities: List[Tuple[str, Dict]]) -> float:
        """Assess occupancy confidence"""
        occupancy_indicators = []
        
        for name, features in modalities:
            if name == 'media' and features.get('faces_detected', 0) > 0:
                occupancy_indicators.append(1.0)
            if name == 'acoustic' and features.get('environment') != 'quiet':
                occupancy_indicators.append(0.7)
            if name == 'wifi' and features.get('motion_detected'):
                occupancy_indicators.append(0.6)
        
        return np.mean(occupancy_indicators) if occupancy_indicators else 0.0
    
    def _assess_environmental_stress(self, modalities: List[Tuple[str, Dict]]) -> float:
        """Assess environmental stress level"""
        stress_factors = []
        
        for name, features in modalities:
            if name == 'thermal' and features['thermal_state'] in ['warm', 'hot']:
                stress_factors.append(0.5 if features['thermal_state'] == 'warm' else 1.0)
            if name == 'acoustic' and features['sound_level'] > 70:
                stress_factors.append(min((features['sound_level'] - 70) / 30, 1.0))
            if name == 'wifi' and features['interference'] > 0.5:
                stress_factors.append(features['interference'])
        
        return np.mean(stress_factors) if stress_factors else 0.0
    
    def _update_pattern_memory(self, fusion_result: Dict[str, Any]):
        """Update pattern memory for learning"""
        timestamp = datetime.now(timezone.utc)
        
        # Store patterns with timestamp
        for pattern in fusion_result.get('behavioral_patterns', []):
            pattern_key = pattern['type']
            self.pattern_memory[pattern_key].append({
                'timestamp': timestamp,
                'confidence': pattern['confidence'],
                'context': fusion_result['environmental_map']
            })
        
        # Limit memory size
        max_memory = 1000
        for key in self.pattern_memory:
            if len(self.pattern_memory[key]) > max_memory:
                self.pattern_memory[key] = self.pattern_memory[key][-max_memory:]
    
    def _empty_fusion_result(self) -> Dict[str, Any]:
        """Return empty fusion result"""
        return {
            'confidence': 0.0,
            'environmental_map': {},
            'detected_entities': [],
            'behavioral_patterns': [],
            'active_modalities': [],
            'correlations': [],
            'anomalies': [],
            'predictions': []
        }

class AnomalyDetector:
    """Multi-modal anomaly detection"""
    
    def __init__(self):
        self.baseline_stats = {}
        self.anomaly_threshold = 2.5  # Standard deviations
        
    async def detect_anomalies(self, fusion_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in fused data"""
        anomalies = []
        
        # Check environmental stress anomaly
        env_stress = fusion_data['environmental_map']['combined'].get('environmental_stress', 0)
        if env_stress > 0.8:
            anomalies.append({
                'type': 'environmental_stress',
                'severity': 'high' if env_stress > 0.9 else 'medium',
                'confidence': env_stress,
                'description': 'Abnormal environmental stress detected'
            })
        
        # Check activity anomaly
        activity = fusion_data['environmental_map']['combined'].get('activity_level', 0)
        if activity > 0.9:
            anomalies.append({
                'type': 'hyperactivity',
                'severity': 'medium',
                'confidence': activity,
                'description': 'Unusually high activity across sensors'
            })
        
        # Check modality disagreement
        if len(fusion_data['active_modalities']) > 2 and fusion_data['confidence'] < 0.3:
            anomalies.append({
                'type': 'sensor_disagreement',
                'severity': 'low',
                'confidence': 0.7,
                'description': 'Significant disagreement between sensor modalities'
            })
        
        return anomalies

class PredictiveModel:
    """Predictive modeling for consciousness system"""
    
    def __init__(self):
        self.history_window = 300  # 5 minutes
        self.prediction_models = {
            'thermal': ThermalPredictor(),
            'activity': ActivityPredictor(),
            'anomaly': AnomalyPredictor()
        }
    
    async def predict(self, fusion_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate predictions based on current state"""
        predictions = []
        
        # Thermal predictions
        thermal_map = fusion_data['environmental_map'].get('thermal', {})
        if thermal_map:
            thermal_pred = self.prediction_models['thermal'].predict(thermal_map)
            if thermal_pred:
                predictions.append(thermal_pred)
        
        # Activity predictions
        activity_level = fusion_data['environmental_map']['combined'].get('activity_level', 0)
        activity_pred = self.prediction_models['activity'].predict(activity_level)
        if activity_pred:
            predictions.append(activity_pred)
        
        # Anomaly predictions
        current_anomalies = fusion_data.get('anomalies', [])
        anomaly_pred = self.prediction_models['anomaly'].predict(current_anomalies)
        if anomaly_pred:
            predictions.append(anomaly_pred)
        
        return predictions

class ThermalPredictor:
    """Thermal trend prediction"""
    
    def predict(self, thermal_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Predict thermal trends"""
        temp = thermal_data.get('temperature', 0)
        state = thermal_data.get('state', 'normal')
        
        if state == 'warm' and temp > 70:
            return {
                'type': 'thermal_warning',
                'timeframe': '5-10 minutes',
                'confidence': 0.7,
                'description': 'Temperature likely to reach critical levels',
                'recommendation': 'Consider increasing cooling or reducing load'
            }
        
        return None

class ActivityPredictor:
    """Activity pattern prediction"""
    
    def predict(self, activity_level: float) -> Optional[Dict[str, Any]]:
        """Predict activity patterns"""
        if activity_level > 0.8:
            return {
                'type': 'sustained_high_activity',
                'timeframe': '10-15 minutes',
                'confidence': 0.6,
                'description': 'High activity likely to continue',
                'recommendation': 'Monitor system resources closely'
            }
        
        return None

class AnomalyPredictor:
    """Anomaly trend prediction"""
    
    def predict(self, current_anomalies: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Predict anomaly trends"""
        if len(current_anomalies) >= 2:
            high_severity = any(a['severity'] == 'high' for a in current_anomalies)
            
            if high_severity:
                return {
                    'type': 'anomaly_cascade',
                    'timeframe': '2-5 minutes',
                    'confidence': 0.8,
                    'description': 'Multiple anomalies may lead to system instability',
                    'recommendation': 'Immediate investigation recommended'
                }
        
        return None

class CorrelationEngine:
    """Find correlations between modalities"""
    
    async def find_correlations(self, modalities: List[Tuple[str, Dict]]) -> List[Dict[str, Any]]:
        """Find correlations between different modalities"""
        correlations = []
        
        # Thermal-Acoustic correlation
        thermal = next((m[1] for m in modalities if m[0] == 'thermal'), None)
        acoustic = next((m[1] for m in modalities if m[0] == 'acoustic'), None)
        
        if thermal and acoustic:
            if thermal['temperature'] > 70 and acoustic['sound_level'] > 70:
                correlations.append({
                    'modalities': ['thermal', 'acoustic'],
                    'type': 'stress_correlation',
                    'strength': 0.8,
                    'description': 'High temperature correlated with high noise level'
                })
        
        # Media-WiFi motion correlation
        media = next((m[1] for m in modalities if m[0] == 'media'), None)
        wifi = next((m[1] for m in modalities if m[0] == 'wifi'), None)
        
        if media and wifi:
            if media.get('motion_detected') and wifi.get('motion_detected'):
                correlations.append({
                    'modalities': ['media', 'wifi'],
                    'type': 'motion_agreement',
                    'strength': 0.9,
                    'description': 'Motion detected by both visual and RF sensors'
                })
        
        # Acoustic-ParasiticRF 60Hz cross-validation
        parasitic_rf = next((m[1] for m in modalities if m[0] == 'parasitic_rf'), None)
        if acoustic and parasitic_rf:
            acoustic_60hz = (acoustic.get('dominant_freq') == 60.0)
            rf_60hz = (parasitic_rf.get('mains_hum', 0) > 0.05)
            
            if acoustic_60hz and rf_60hz:
                correlations.append({
                    'modalities': ['acoustic', 'parasitic_rf'],
                    'type': 'emi_acoustic_agreement',
                    'strength': 0.95,
                    'description': 'Verified 60Hz EMI signature cross-validated with acoustic hum.'
                })
        
        return correlations

class PerfectConsciousnessOrchestrator:
    """
    Master orchestrator for the complete consciousness system
    """
    
    def __init__(self, config: ConsciousnessConfig, redis_client: Optional[redis.Redis] = None):
        self.config = config
        self.redis_client = redis_client
        
        # Initialize all consciousness components
        self.thermal_consciousness = PerfectThermalConsciousness(config.thermal, redis_client)
        self.acoustic_consciousness = PerfectAcousticConsciousness(config.acoustic, redis_client)
        self.media_consciousness = PerfectMediaConsciousness(config.media, redis_client)
        self.wifi_consciousness = PerfectWiFiConsciousness(config.wifi, redis_client)
        self.parasitic_rf_consciousness = PerfectParasiticRFConsciousness(config.parasitic_rf)
        self.bluetooth_consciousness = PerfectBluetoothConsciousness(config.bluetooth)
        
        # Initialize fusion and management
        self.fusion_engine = ConsciousnessFusion()
        self.websocket_manager = PerfectWebSocketManager(redis_client)
        
        # State management
        self._running = False
        self._monitoring_task: Optional[asyncio.Task] = None
        self._state_history: List[UnifiedConsciousnessState] = []
        self._performance_monitor = PerformanceMonitor()
        
        logger.info("Perfect Consciousness Orchestrator initialized")
    
    async def start(self):
        """Start all consciousness systems"""
        logger.info("Starting Perfect Consciousness Orchestrator...")
        
        # Start individual consciousness systems
        await asyncio.gather(
            self.thermal_consciousness.start(),
            self.acoustic_consciousness.start(),
            self.media_consciousness.start(),
            self.wifi_consciousness.start(),
            self.parasitic_rf_consciousness.start() if hasattr(self.parasitic_rf_consciousness, 'start') else asyncio.sleep(0),
            self.bluetooth_consciousness.start() if hasattr(self.bluetooth_consciousness, 'start') else asyncio.sleep(0),
            self.websocket_manager.start()
        )
        
        # Start monitoring loop
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Perfect Consciousness Orchestrator started successfully")
    
    async def stop(self):
        """Stop all consciousness systems"""
        logger.info("Stopping Perfect Consciousness Orchestrator...")
        
        self._running = False
        
        # Stop monitoring
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Stop individual systems
        await asyncio.gather(
            self.thermal_consciousness.stop(),
            self.acoustic_consciousness.stop(),
            self.media_consciousness.stop(),
            self.wifi_consciousness.stop(),
            self.parasitic_rf_consciousness.stop() if hasattr(self.parasitic_rf_consciousness, 'stop') else asyncio.sleep(0),
            self.bluetooth_consciousness.stop() if hasattr(self.bluetooth_consciousness, 'stop') else asyncio.sleep(0),
            self.websocket_manager.stop()
        )
        
        logger.info("Perfect Consciousness Orchestrator stopped")
    
    async def get_unified_consciousness(
        self,
        user_id: str,
        correlation_id: Optional[str] = None
    ) -> UnifiedConsciousnessState:
        """
        Get complete unified consciousness state
        """
        if not correlation_id:
            correlation_id = str(uuid.uuid4())
        
        start_time = datetime.now(timezone.utc)
        
        # Gather all consciousness data in parallel (6 Modalities)
        results = await asyncio.gather(
            self._get_thermal_safe(user_id, correlation_id),
            self._get_acoustic_safe(user_id, correlation_id),
            self._get_media_safe(user_id, correlation_id),
            self._get_wifi_safe(user_id, correlation_id),
            self._get_parasitic_rf_safe(user_id, correlation_id),
            self._get_bluetooth_safe(user_id, correlation_id),
            return_exceptions=True
        )
        
        # Process results
        thermal, acoustic, media, wifi, parasitic_rf, bluetooth = results
        
        # Handle exceptions
        thermal = None if isinstance(thermal, Exception) else thermal
        acoustic = None if isinstance(acoustic, Exception) else acoustic
        media = None if isinstance(media, Exception) else media
        wifi = None if isinstance(wifi, Exception) else wifi
        parasitic_rf = None if isinstance(parasitic_rf, Exception) else parasitic_rf
        bluetooth = None if isinstance(bluetooth, Exception) else bluetooth
        
        # Perform consciousness fusion
        fusion_result = await self.fusion_engine.fuse_consciousness(
            thermal, acoustic, media, wifi, parasitic_rf, bluetooth
        )
        
        # Calculate performance metrics
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        active_modalities = [
            name for name, data in [
                ('thermal', thermal),
                ('acoustic', acoustic),
                ('media', media),
                ('wifi', wifi)
            ] if data is not None
        ]
        
        # Build unified state
        state = UnifiedConsciousnessState(
            correlation_id=correlation_id,
            thermal=thermal,
            acoustic=acoustic,
            media=media,
            wifi=wifi,
            parasitic_rf=parasitic_rf,
            bluetooth=bluetooth,
            fusion_confidence=fusion_result['confidence'],
            environmental_map=fusion_result['environmental_map'],
            detected_entities=fusion_result['detected_entities'],
            behavioral_patterns=fusion_result['behavioral_patterns'],
            active_modalities=active_modalities,
            system_health=self._assess_system_health(active_modalities),
            performance_metrics={
                'response_time_ms': duration * 1000,
                'active_modalities': len(active_modalities),
                'fusion_confidence': fusion_result['confidence']
            },
            alerts=self._consolidate_alerts(thermal, acoustic, media, wifi, parasitic_rf, bluetooth),
            predictions=fusion_result.get('predictions', []),
            anomalies=fusion_result.get('anomalies', [])
        )
        
        # Store in history
        self._state_history.append(state)
        if len(self._state_history) > 100:
            self._state_history = self._state_history[-100:]
        
        # Update performance metrics
        self._performance_monitor.record_request(duration, len(active_modalities))
        
        # Broadcast to WebSocket clients
        await self._broadcast_consciousness_update(state)
        
        return state
    
    async def _get_thermal_safe(self, user_id: str, correlation_id: str) -> Optional[ThermalAwareness]:
        """Safely get thermal awareness"""
        try:
            return await self.thermal_consciousness.get_thermal_awareness(user_id, correlation_id)
        except Exception as e:
            logger.error(f"Failed to get thermal awareness: {e}")
            return None
    
    async def _get_acoustic_safe(self, user_id: str, correlation_id: str) -> Optional[AcousticAwareness]:
        """Safely get acoustic awareness"""
        try:
            return await self.acoustic_consciousness.get_acoustic_awareness(user_id, correlation_id)
        except Exception as e:
            logger.error(f"Failed to get acoustic awareness: {e}")
            return None
    
    async def _get_media_safe(self, user_id: str, correlation_id: str) -> Optional[MediaAwareness]:
        """Safely get media awareness"""
        try:
            return await self.media_consciousness.get_media_awareness(user_id, correlation_id)
        except Exception as e:
            logger.error(f"Failed to get media awareness: {e}")
            return None
    
    async def _get_wifi_safe(self, user_id: str, correlation_id: str) -> Optional[WiFiAwareness]:
        """Safely get WiFi awareness"""
        try:
            return await self.wifi_consciousness.get_wifi_awareness(user_id, correlation_id)
        except Exception as e:
            logger.error(f"Failed to get WiFi awareness: {e}")
            return None

    async def _get_parasitic_rf_safe(self, user_id: str, correlation_id: str) -> Optional[ParasiticRFAwareness]:
        """Safely get Parasitic RF awareness"""
        try:
            return await self.parasitic_rf_consciousness.get_rf_awareness(user_id, correlation_id)
        except Exception as e:
            logger.error(f"Failed to get Parasitic RF awareness: {e}")
            return None

    async def _get_bluetooth_safe(self, user_id: str, correlation_id: str) -> Optional[BluetoothAwareness]:
        """Safely get Bluetooth awareness"""
        try:
            return await self.bluetooth_consciousness.get_bluetooth_awareness(user_id, correlation_id)
        except Exception as e:
            logger.error(f"Failed to get Bluetooth awareness: {e}")
            return None
    
    def _assess_system_health(self, active_modalities: List[str]) -> str:
        """Assess overall system health"""
        total_modalities = 6
        active_count = len(active_modalities)
        
        if active_count == total_modalities:
            return "healthy"
        elif active_count >= total_modalities * 0.75:
            return "degraded"
        elif active_count >= total_modalities * 0.5:
            return "impaired"
        else:
            return "critical"
    
    def _consolidate_alerts(
        self,
        thermal: Optional[ThermalAwareness],
        acoustic: Optional[AcousticAwareness],
        media: Optional[MediaAwareness],
        wifi: Optional[WiFiAwareness],
        parasitic_rf: Optional[ParasiticRFAwareness],
        bluetooth: Optional[BluetoothAwareness]
    ) -> List[Dict[str, Any]]:
        """Consolidate alerts from all modalities"""
        all_alerts = []
        
        # Add thermal alerts
        if thermal and thermal.alerts:
            for alert in thermal.alerts:
                all_alerts.append({
                    'source': 'thermal',
                    'message': alert,
                    'timestamp': thermal.timestamp.isoformat(),
                    'severity': self._classify_alert_severity(alert)
                })
        
        # Add acoustic alerts
        if acoustic and acoustic.alerts:
            for alert in acoustic.alerts:
                all_alerts.append({
                    'source': 'acoustic',
                    'message': alert,
                    'timestamp': acoustic.timestamp.isoformat(),
                    'severity': self._classify_alert_severity(alert)
                })
        
        # Add media alerts
        if media and media.alerts:
            for alert in media.alerts:
                all_alerts.append({
                    'source': 'media',
                    'message': alert,
                    'timestamp': media.timestamp.isoformat(),
                    'severity': self._classify_alert_severity(alert)
                })
        
        # Add WiFi alerts
        if wifi and wifi.alerts:
            for alert in wifi.alerts:
                all_alerts.append({
                    'source': 'wifi',
                    'message': alert,
                    'timestamp': wifi.timestamp.isoformat(),
                    'severity': self._classify_alert_severity(alert)
                })

        # Add Parasitic RF alerts
        if parasitic_rf and parasitic_rf.alerts:
            for alert in parasitic_rf.alerts:
                all_alerts.append({
                    'source': 'parasitic_rf',
                    'message': alert,
                    'timestamp': parasitic_rf.timestamp.isoformat(),
                    'severity': self._classify_alert_severity(alert)
                })

        # Add Bluetooth alerts
        if bluetooth and bluetooth.alerts:
            for alert in bluetooth.alerts:
                all_alerts.append({
                    'source': 'bluetooth',
                    'message': alert,
                    'timestamp': bluetooth.timestamp.isoformat(),
                    'severity': self._classify_alert_severity(alert)
                })
        
        # Sort by severity and timestamp
        all_alerts.sort(key=lambda x: (
            {'critical': 0, 'warning': 1, 'info': 2}.get(x['severity'], 3),
            x['timestamp']
        ))
        
        return all_alerts
    
    def _classify_alert_severity(self, alert: str) -> str:
        """Classify alert severity based on content"""
        alert_lower = alert.lower()
        
        if any(word in alert_lower for word in ['critical', 'danger', 'emergency', 'failure']):
            return 'critical'
        elif any(word in alert_lower for word in ['warning', 'caution', 'elevated', 'high']):
            return 'warning'
        else:
            return 'info'
    
    async def _broadcast_consciousness_update(self, state: UnifiedConsciousnessState):
        """Broadcast consciousness update to WebSocket clients"""
        message = WebSocketMessage(
            type="consciousness_update",
            data={
                'correlation_id': state.correlation_id,
                'timestamp': state.timestamp.isoformat(),
                'fusion_confidence': state.fusion_confidence,
                'system_health': state.system_health,
                'active_modalities': state.active_modalities,
                'environmental_map': state.environmental_map,
                'detected_entities': state.detected_entities,
                'behavioral_patterns': state.behavioral_patterns,
                'alerts': state.alerts[:5],  # Top 5 alerts
                'predictions': state.predictions[:3],  # Top 3 predictions
                'anomalies': state.anomalies[:3]  # Top 3 anomalies
            }
        )
        
        await self.websocket_manager.broadcast_to_channel(
            'consciousness',
            message.dict()
        )
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        logger.info("Consciousness monitoring loop started")
        
        while self._running:
            try:
                # Get unified consciousness
                state = await self.get_unified_consciousness("system", "monitor")
                
                # Check for critical conditions
                if state.system_health == "critical":
                    logger.error("System health critical!")
                    await self._handle_critical_state(state)
                
                # Check for high-severity alerts
                critical_alerts = [
                    alert for alert in state.alerts
                    if alert['severity'] == 'critical'
                ]
                
                if critical_alerts:
                    logger.warning(f"Critical alerts detected: {len(critical_alerts)}")
                    await self._handle_critical_alerts(critical_alerts)
                
                # Wait for next cycle
                await asyncio.sleep(self.config.thermal.poll_interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(5)
    
    async def _handle_critical_state(self, state: UnifiedConsciousnessState):
        """Handle critical system state"""
        # Log critical state
        logger.critical(f"Critical system state detected: {state.dict()}")
        
        # Send emergency notification
        emergency_message = WebSocketMessage(
            type="emergency",
            data={
                'severity': 'critical',
                'message': 'System in critical state',
                'active_modalities': state.active_modalities,
                'timestamp': state.timestamp.isoformat()
            }
        )
        
        await self.websocket_manager.broadcast_to_channel(
            'alerts',
            emergency_message.dict(),
            permission='receive_emergency_alerts'
        )
        
        # Store in Redis if available
        if self.redis_client:
            await self.redis_client.lpush(
                'consciousness:critical_events',
                json.dumps({
                    'timestamp': state.timestamp.isoformat(),
                    'correlation_id': state.correlation_id,
                    'state': state.dict()
                })
            )
    
    async def _handle_critical_alerts(self, alerts: List[Dict[str, Any]]):
        """Handle critical alerts"""
        for alert in alerts:
            logger.critical(f"Critical alert from {alert['source']}: {alert['message']}")
        
        # Broadcast critical alerts
        alert_message = WebSocketMessage(
            type="critical_alerts",
            data={
                'alerts': alerts,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        )
        
        await self.websocket_manager.broadcast_to_channel(
            'alerts',
            alert_message.dict(),
            permission='receive_alerts'
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'orchestrator': {
                'running': self._running,
                'uptime': self._performance_monitor.get_uptime(),
                'state_history_size': len(self._state_history)
            },
            'components': {
                'thermal': self.thermal_consciousness.get_health_status(),
                'acoustic': self.acoustic_consciousness.get_health_status(),
                'media': self.media_consciousness.get_health_status(),
                'wifi': self.wifi_consciousness.get_health_status(),
                'parasitic_rf': self.parasitic_rf_consciousness.get_health_status(),
                'bluetooth': self.bluetooth_consciousness.get_health_status(),
                'websocket': self.websocket_manager.get_status()
            },
            'performance': self._performance_monitor.get_metrics(),
            'last_state': self._state_history[-1].dict() if self._state_history else None
        }

class PerformanceMonitor:
    """Monitor system performance"""
    
    def __init__(self):
        self.start_time = datetime.now(timezone.utc)
        self.request_count = 0
        self.total_duration = 0.0
        self.modality_counts = defaultdict(int)
        
    def record_request(self, duration: float, modality_count: int):
        """Record request metrics"""
        self.request_count += 1
        self.total_duration += duration
        self.modality_counts[modality_count] += 1
    
    def get_uptime(self) -> float:
        """Get system uptime in seconds"""
        return (datetime.now(timezone.utc) - self.start_time).total_seconds()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        avg_duration = self.total_duration / self.request_count if self.request_count > 0 else 0
        
        return {
            'uptime_seconds': self.get_uptime(),
            'total_requests': self.request_count,
            'average_response_ms': avg_duration * 1000,
            'requests_per_second': self.request_count / self.get_uptime() if self.get_uptime() > 0 else 0,
            'modality_distribution': dict(self.modality_counts)
        }

# Global orchestrator instance
orchestrator: Optional[PerfectConsciousnessOrchestrator] = None

async def get_orchestrator() -> PerfectConsciousnessOrchestrator:
    """Get or create orchestrator instance"""
    global orchestrator
    if orchestrator is None:
        config = ConsciousnessConfig()
        # In production, use real Redis
        redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        orchestrator = PerfectConsciousnessOrchestrator(config, redis_client)
        await orchestrator.start()
    return orchestrator

# Example usage
async def main():
    """Example usage of perfect consciousness orchestrator"""
    config = ConsciousnessConfig(
        fusion_enabled=True,
        prediction_horizon=300,
        alert_threshold=0.7
    )
    
    # Initialize with Redis
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
    
    orchestrator = PerfectConsciousnessOrchestrator(config, redis_client)
    
    # Start the system
    await orchestrator.start()
    
    try:
        # Get unified consciousness
        state = await orchestrator.get_unified_consciousness("demo-user")
        
        print(f"System Health: {state.system_health}")
        print(f"Active Modalities: {state.active_modalities}")
        print(f"Fusion Confidence: {state.fusion_confidence:.2%}")
        print(f"Detected Entities: {len(state.detected_entities)}")
        print(f"Behavioral Patterns: {len(state.behavioral_patterns)}")
        print(f"Active Alerts: {len(state.alerts)}")
        
        # Get system status
        status = orchestrator.get_system_status()
        print(f"\nSystem Status: {json.dumps(status, indent=2)}")
        
    finally:
        # Stop the system
        await orchestrator.stop()

if __name__ == "__main__":
    asyncio.run(main())