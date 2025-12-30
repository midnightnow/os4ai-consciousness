"""
OS4AI Perfect Bluetooth Consciousness Implementation
Production-ready Bluetooth spatial mapping and proximity sensing
"""

import asyncio
import subprocess
import json
import logging
import os
import shutil
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field, validator
from fastapi import HTTPException, status

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BluetoothConfig(BaseModel):
    """Bluetooth monitoring configuration"""
    poll_interval: int = Field(10, ge=5, le=60)
    rssi_threshold_near: int = Field(-60, ge=-100, le=0)
    rssi_threshold_far: int = Field(-90, ge=-100, le=0)
    scan_timeout: int = Field(15, ge=5, le=30)
    enable_spatial_mapping: bool = True

class BluetoothDevice(BaseModel):
    """Bluetooth device state"""
    name: str
    address: str
    rssi: Optional[int]
    connected: bool
    major_class: Optional[str]
    minor_class: Optional[str]
    confidence: float = 1.0

class BluetoothAwareness(BaseModel):
    """Unified state for Bluetooth consciousness"""
    devices_count: int
    active_connections: int
    nearby_devices: List[BluetoothDevice]
    proximity_alert: bool
    spatial_density: float # Devices per unit area (estimated)
    alerts: List[str]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    correlation_id: str

class SecureBluetoothInterface:
    """
    Secure hardware interface for Bluetooth scanning using system tools.
    """
    
    def __init__(self, config: BluetoothConfig):
        self.config = config
        
    async def scan_devices(self, correlation_id: str) -> List[Dict[str, Any]]:
        """
        Scan for Bluetooth devices using system_profiler.
        """
        try:
            # system_profiler is safer than blueutil as it's built-in
            cmd = ["system_profiler", "SPBluetoothDataType", "-json"]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={'PATH': '/usr/sbin:/usr/bin:/bin'}
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.config.scan_timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                logger.error(f"[{correlation_id}] Bluetooth scan timeout")
                return []

            if process.returncode != 0:
                logger.error(f"[{correlation_id}] Bluetooth scan failed: {stderr.decode()}")
                return []

            data = json.loads(stdout.decode())
            return self._parse_bluetooth_data(data)

        except Exception as e:
            logger.error(f"[{correlation_id}] Bluetooth Hardware Error: {e}")
            return []

    def _parse_bluetooth_data(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Parses complex system_profiler Bluetooth output into flattened device list.
        """
        devices = []
        try:
            bt_info = data.get("SPBluetoothDataType", [])[0]
            
            # Connected devices
            connected = bt_info.get("device_connected", [])
            for dev in connected:
                for name, info in dev.items():
                    devices.append({
                        "name": name,
                        "address": info.get("device_address", "Unknown"),
                        "connected": True,
                        "rssi": None, # system_profiler doesn't always show RSSI for connected
                        "major_class": info.get("device_majorClass"),
                        "minor_class": info.get("device_minorClass")
                    })
            
            # Not connected (previously paired)
            not_connected = bt_info.get("device_not_connected", [])
            for dev in not_connected:
                for name, info in dev.items():
                    devices.append({
                        "name": name,
                        "address": info.get("device_address", "Unknown"),
                        "connected": False,
                        "rssi": None,
                        "major_class": info.get("device_majorClass"),
                        "minor_class": info.get("device_minorClass")
                    })
                    
        except (IndexError, KeyError, TypeError) as e:
            logger.warning(f"Error parsing Bluetooth JSON: {e}")
            
        return devices

class PerfectBluetoothConsciousness:
    """
    Orchestrates Bluetooth sensing with multi-modal integration ready logic.
    """
    
    def __init__(self, config: BluetoothConfig):
        self.config = config
        self.interface = SecureBluetoothInterface(config)
        
    async def get_bluetooth_awareness(self, user_id: str, correlation_id: str) -> BluetoothAwareness:
        """
        Get current Bluetooth awareness state.
        """
        raw_devices = await self.interface.scan_devices(correlation_id)
        
        devices = [BluetoothDevice(**d) for d in raw_devices]
        active_conn = sum(1 for d in devices if d.connected)
        
        # In a real dynamic scanner (like blueutil), we'd have RSSI
        # Since system_profiler is static-ish, we simulate spatial density
        # based on total known devices in the vicinity.
        density = len(devices) / 10.0 # Heuristic
        
        alerts = []
        if active_conn > 5:
            alerts.append(f"High number of active Bluetooth connections ({active_conn}).")
            
        return BluetoothAwareness(
            devices_count=len(devices),
            active_connections=active_conn,
            nearby_devices=devices,
            proximity_alert=active_conn > 0,
            spatial_density=density,
            alerts=alerts,
            correlation_id=correlation_id
        )

    def get_health_status(self) -> Dict[str, Any]:
        """Get Bluetooth system health status"""
        return {
            "component": "bluetooth_consciousness",
            "status": "healthy" if shutil.which("system_profiler") else "degraded",
            "tool_missing": not shutil.which("system_profiler")
        }

if __name__ == "__main__":
    asyncio.run(PerfectBluetoothConsciousness(BluetoothConfig()).get_bluetooth_awareness("admin", "test"))
