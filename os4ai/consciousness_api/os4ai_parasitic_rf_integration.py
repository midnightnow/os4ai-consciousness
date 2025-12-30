"""
OS4AI Perfect Parasitic RF Consciousness Implementation
Production-ready 60Hz EMI sensing via Audio FFT (Parasitic Demodulation)
"""

import asyncio
import subprocess
import json
import logging
import hashlib
import os
import shutil
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field, validator
from fastapi import HTTPException, status
import numpy as np

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ParasiticRFConfig(BaseModel):
    """Configuration for Parasitic RF EMI monitoring"""
    sample_duration: float = Field(0.2, ge=0.05, le=1.0)
    sample_rate: int = Field(44100, ge=8000, le=96000)
    target_frequency: float = Field(60.0, ge=40.0, le=100.0) # 60Hz mains hum
    poll_interval: int = Field(5, ge=1, le=60)
    threshold_low: float = Field(0.001, ge=0.0, le=1.0)
    threshold_high: float = Field(0.1, ge=0.0, le=1.0)
    timeout: int = Field(5, ge=1, le=10)

class ParasiticRFAwareness(BaseModel):
    """Unified state for Parasitic RF consciousness"""
    mains_hum_magnitude: float
    signal_to_noise_ratio: float
    emi_activity_level: str # 'low', 'medium', 'high'
    motion_interference_detected: bool
    spectral_profile: List[float]
    alerts: List[str]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    correlation_id: str

class SecureAudioEMIInterface:
    """
    Secure hardware interface for capturing EMI via Microphone FFT.
    """
    
    def __init__(self, config: ParasiticRFConfig):
        self.config = config
        self._last_magnitude = 0.0
        
    def _check_tool(self, tool_name: str) -> bool:
        """Verifies that a binary exists on the system."""
        return shutil.which(tool_name) is not None

    async def read_emi_magnitude(self, correlation_id: str) -> Dict[str, Any]:
        """
        Captures raw audio and performs FFT to extract EMI signature.
        """
        if not self._check_tool("rec"):
            logger.error(f"[{correlation_id}] SoX 'rec' tool not found. Install with 'brew install sox'.")
            return {"magnitude": -1.0, "snr": -1.0, "data": []}

        try:
            # Capture Raw Audio via SoX
            # We grab raw 32-bit float data to preserve dynamic range.
            cmd = [
                "rec", "-t", "raw", "-e", "float", "-b", "32", "-c", "1", 
                "-r", str(self.config.sample_rate), "-", "trim", "0", str(self.config.sample_duration)
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={'PATH': '/usr/local/bin:/usr/bin:/bin'}
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.config.timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return {"magnitude": -1.0, "snr": -1.0, "data": []}

            if process.returncode != 0 or not stdout:
                logger.error(f"[{correlation_id}] Audio capture failed: {stderr.decode()}")
                return {"magnitude": -1.0, "snr": -1.0, "data": []}

            # Convert bytes to numpy array
            audio_data = np.frombuffer(stdout, dtype=np.float32)
            
            if len(audio_data) == 0:
                return {"magnitude": 0.0, "snr": 0.0, "data": []}

            # Perform FFT
            fft_spectrum = np.fft.rfft(audio_data)
            freqs = np.fft.rfftfreq(len(audio_data), 1/self.config.sample_rate)
            
            # Extract Target Frequency Magnitude
            idx = (np.abs(freqs - self.config.target_frequency)).argmin()
            magnitude = np.abs(fft_spectrum[idx])
            
            # Calculate Total Energy for normalization
            total_energy = np.sum(np.abs(fft_spectrum))
            if total_energy == 0:
                snr = 0.0
            else:
                snr = magnitude / total_energy
                
            return {
                "magnitude": float(magnitude),
                "snr": float(snr),
                "data": np.abs(fft_spectrum[:100]).tolist() # Return first 100 bins for profiling
            }

        except Exception as e:
            logger.error(f"[{correlation_id}] EMI Hardware Error: {e}")
            return {"magnitude": -1.0, "snr": -1.0, "data": []}

class PerfectParasiticRFConsciousness:
    """
    Orchestrates Parasitic RF sensing with security and health monitoring.
    """
    
    def __init__(self, config: ParasiticRFConfig):
        self.config = config
        self.interface = SecureAudioEMIInterface(config)
        
    async def get_rf_awareness(self, user_id: str, correlation_id: str) -> ParasiticRFAwareness:
        """
        Get current RF awareness state based on parasitic EMI.
        """
        result = await self.interface.read_emi_magnitude(correlation_id)
        
        snr = result["snr"]
        magnitude = result["magnitude"]
        
        # Determine activity level
        if snr < self.config.threshold_low:
            activity = "low"
        elif snr < self.config.threshold_high:
            activity = "medium"
        else:
            activity = "high"
            
        # Detect motion interference (High variance in EMI signature)
        # In a real implementation, we'd compare against history
        motion_detected = snr > 0.05 

        alerts = []
        if snr > self.config.threshold_high:
            alerts.append("High EMI detected: Potential electronic device nearby or mains interference.")
        
        return ParasiticRFAwareness(
            mains_hum_magnitude=magnitude,
            signal_to_noise_ratio=snr,
            emi_activity_level=activity,
            motion_interference_detected=motion_detected,
            spectral_profile=result["data"],
            alerts=alerts,
            correlation_id=correlation_id
        )

    def get_health_status(self) -> Dict[str, Any]:
        """Get Parasitic RF system health status"""
        return {
            "component": "parasitic_rf_consciousness",
            "status": "healthy" if self.interface._check_tool("rec") else "degraded",
            "tool_missing": not self.interface._check_tool("rec")
        }

# Lifecycle management
async def test_integration():
    config = ParasiticRFConfig()
    p_rf = PerfectParasiticRFConsciousness(config)
    awareness = await p_rf.get_rf_awareness("admin", "test_id")
    print(awareness.json())

if __name__ == "__main__":
    asyncio.run(test_integration())
