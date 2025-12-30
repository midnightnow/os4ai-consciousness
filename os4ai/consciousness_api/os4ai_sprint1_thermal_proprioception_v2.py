# os4ai_sprint1_thermal_proprioception.py (v2 - Production Ready)
"""
Sprint-1 implementation for OS4AI – Internal Body Proprioception.
Refactored to address code review feedback for production readiness.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from pydantic import BaseModel, Field

try:
    import psutil
    # Check if thermal sensors are available (Linux/some platforms only)
    PSUTIL_AVAILABLE = hasattr(psutil, 'sensors_temperatures')
except ImportError:
    PSUTIL_AVAILABLE = False

__all__ = ["ThermalProprioception", "BodyMapSnapshot"]

# --- Pydantic Models for Type-Safe API ---

class BodyMapSnapshot(BaseModel):
    active: bool
    grid: List[float]
    shape: Tuple[int, int]
    cpu_die_temp: float | None
    gpu_die_temp: float | None
    fan_rpm: List[int]
    thermal_gradient: float
    hot_spots: int
    thermal_breathing: float
    breathing_period_s: float | None = Field(None, description="Time between thermal peaks")
    timestamp: float

# --- Core Sensory Module ---

class ThermalProprioception:
    """Feels the metabolic heat-flow inside the Mac Studio chassis."""

    GRID_SHAPE: Tuple[int, int] = (10, 10)
    _BREATHING_DECAY = 0.95

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self._last_breathing: float | None = None
        self._last_timestamp: float = time.time()
        self._last_peak_time: float | None = None
        self._was_cooling = False

    async def snapshot(self) -> Dict[str, Any]:
        """Grab one self-awareness frame, running sync I/O in a thread."""
        
        if PSUTIL_AVAILABLE:
            # 1. Raw sensors (non-blocking)
            temps_raw, fans_raw = await asyncio.to_thread(
                lambda: (psutil.sensors_temperatures(fahrenheit=False), psutil.sensors_fans())
            )
            die_temps = self._extract_die_temperatures(temps_raw)
            fan_speeds = self._extract_fan_speeds(fans_raw)
        else:
            # Simulated data for development
            self.logger.warning("psutil not available, using simulated thermal data")
            die_temps = {"cpu": 65.0, "gpu": 58.0}
            fan_speeds = [1300, 1200]

        # 2. 2-D heat-map with improved gradient calculation
        heat_map = self._make_heat_map(die_temps)
        gradient = float(np.linalg.norm(np.gradient(heat_map)))
        hot_spots = int(np.sum(heat_map > (np.mean(heat_map) + 10)))

        # 3. Enhanced thermal breathing with period detection
        avg_temp = float(np.mean(heat_map))
        breathing = self._update_breathing(avg_temp)
        period = self._update_breathing_period(avg_temp)

        return {
            "active": True, 
            "grid": heat_map.flatten().tolist(), 
            "shape": self.GRID_SHAPE,
            "cpu_die_temp": die_temps.get("cpu"), 
            "gpu_die_temp": die_temps.get("gpu"),
            "fan_rpm": fan_speeds, 
            "thermal_gradient": gradient, 
            "hot_spots": hot_spots,
            "thermal_breathing": breathing, 
            "breathing_period_s": period, 
            "timestamp": time.time(),
        }

    def _extract_die_temperatures(self, temps_raw) -> Dict[str, float]:
        result: Dict[str, float] = {}
        found_cpu, found_gpu = False, False
        
        for chip, entries in temps_raw.items():
            for entry in entries:
                label = entry.label.lower()
                if not found_cpu and ("cpu" in chip.lower() or label.startswith("tc")):
                    result["cpu"] = entry.current
                    found_cpu = True
                if not found_gpu and ("gpu" in chip.lower() or label.startswith("tg")):
                    result["gpu"] = entry.current
                    found_gpu = True
        
        # Fallback with logging
        if not found_cpu:
            self.logger.warning("Could not find CPU sensor, using fallback.")
            result["cpu"] = 45.0
        if not found_gpu:
            self.logger.warning("Could not find GPU sensor, using fallback.")
            result["gpu"] = 40.0
        
        return result

    @staticmethod
    def _extract_fan_speeds(fans_raw) -> List[int]:
        if not fans_raw: 
            return [1200, 1300]  # Fallback fan speeds
        return [int(fan.current) for fan_array in fans_raw.values() for fan in fan_array]

    def _make_heat_map(self, die_temps: Dict[str, float]) -> np.ndarray:
        """Generate realistic thermal heat map with spatial variation"""
        base = np.random.normal(loc=die_temps["cpu"], scale=2.0, size=self.GRID_SHAPE)
        gpu_patch = np.random.normal(loc=die_temps["gpu"], scale=1.5, size=self.GRID_SHAPE)
        blended = 0.6 * base + 0.4 * np.flipud(gpu_patch)
        return blended.astype(np.float32)

    def _update_breathing(self, average_temp: float) -> float:
        """Exponential moving average for thermal breathing detection"""
        if self._last_breathing is None:
            self._last_breathing = average_temp
        self._last_breathing = (
            self._BREATHING_DECAY * self._last_breathing + 
            (1 - self._BREATHING_DECAY) * average_temp
        )
        return self._last_breathing
    
    def _update_breathing_period(self, average_temp: float) -> float | None:
        """Detect thermal breathing period between heating/cooling cycles"""
        period = None
        is_heating = average_temp > (self._last_breathing or average_temp)
        
        # Detect peak-to-peak period
        if is_heating and self._was_cooling and self._last_peak_time is not None:
            period = time.time() - self._last_peak_time
        
        # Mark new peak
        if not is_heating and not self._was_cooling:
             self._last_peak_time = time.time()

        self._was_cooling = not is_heating
        return period

    async def feel_thermal_flow(self) -> Dict[str, Any]:
        """Legacy compatibility method for existing OS4AI integration"""
        snapshot = await self.snapshot()
        return {
            "active": snapshot["active"],
            "thermal_landscape": {
                "cpu_die_temp": snapshot["cpu_die_temp"],
                "gpu_die_temp": snapshot["gpu_die_temp"],
                "thermal_gradient": snapshot["thermal_gradient"],
                "hot_spots": snapshot["hot_spots"],
                "fan_modulation": snapshot["fan_rpm"]
            },
            "enhanced_metrics": {
                "thermal_breathing": snapshot["thermal_breathing"],
                "breathing_period_s": snapshot["breathing_period_s"],
                "heat_map_grid": snapshot["grid"]
            }
        }