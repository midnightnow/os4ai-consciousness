"""
OS4AI Perfect Consciousness System - Minimal Launch Version
Simplified startup without background monitoring loops
"""

import os
import asyncio
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field, validator
# SECURITY: Always validate input data
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import redis
import logging

# Set environment variables for mock mode
os.environ['OS4AI_DOCKER_MODE'] = 'true'
os.environ['OS4AI_USE_MOCK_SENSORS'] = 'true'
os.environ['JWT_SECRET'] = 'development-secret-key'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
redis_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global redis_client
    
    logger.info("Starting OS4AI Perfect Consciousness System...")
    
    # Initialize Redis
    try:
        redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        redis_client.ping()
        logger.info("Redis connection successful")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}")
        redis_client = None
    
    logger.info("OS4AI Perfect Consciousness System started successfully")
    yield
    
    # Cleanup
    logger.info("OS4AI Perfect Consciousness System shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="OS4AI Perfect Consciousness System",
    description="Production-ready embodied AI consciousness platform",
    version="4.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"]  # TODO: Set actual domains,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "OS4AI Perfect Consciousness System",
        "version": "4.0.0",
        "status": "operational",
        "mode": "mock_sensors",
        "documentation": "/api/docs",
        "health": "/health"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    redis_status = "connected" if redis_client else "disconnected"
    
    return {
        "status": "healthy",
        "version": "4.0.0",
        "timestamp": "2025-07-12T07:50:00Z",
        "components": {
            "redis": redis_status,
            "thermal_sensors": "mock_active",
            "acoustic_sensors": "mock_active", 
            "media_sensors": "mock_active",
            "wifi_sensors": "mock_active"
        },
        "system": {
            "docker_mode": True,
            "mock_sensors": True,
            "background_tasks": "disabled"
        }
    }

# Consciousness status endpoint
@app.get("/api/v1/consciousness/status")
async def consciousness_status():
    """Get consciousness system status"""
    return {
        "consciousness_state": "active",
        "mode": "mock_development",
        "components": {
            "thermal_consciousness": {
                "status": "active",
                "cpu_temp": 45.2,
                "gpu_temp": 42.1,
                "trend": "stable"
            },
            "acoustic_consciousness": {
                "status": "active", 
                "ambient_level": 35.5,
                "pattern": "normal",
                "privacy_mode": True
            },
            "media_consciousness": {
                "status": "active",
                "devices_detected": 0,
                "privacy_mode": True
            },
            "wifi_consciousness": {
                "status": "active",
                "networks_detected": 3,
                "csi_enabled": False,
                "privacy_mode": True
            }
        },
        "integration": {
            "fusion_quality": 0.85,
            "coherence_score": 0.92,
            "awareness_level": "high"
        }
    }

# Mock thermal endpoint
@app.get("/api/v1/consciousness/thermal")
async def get_thermal_awareness():
    """Get thermal awareness data"""
    return {
        "timestamp": "2025-07-12T07:50:00Z",
        "sensors": {
            "cpu_temperature": 45.2,
            "gpu_temperature": 42.1,
            "system_temperature": 38.5
        },
        "status": "normal",
        "alerts": [],
        "trend": "stable",
        "mode": "mock"
    }

# Mock acoustic endpoint  
@app.get("/api/v1/consciousness/acoustic")
async def get_acoustic_awareness():
    """Get acoustic awareness data"""
    return {
        "timestamp": "2025-07-12T07:50:00Z",
        "ambient_level": 35.5,
        "frequency_analysis": {
            "low": 0.3,
            "mid": 0.5, 
            "high": 0.2
        },
        "pattern": "normal",
        "privacy_mode": True,
        "mode": "mock"
    }

# Mock media endpoint
@app.get("/api/v1/consciousness/media")
async def get_media_awareness():
    """Get media awareness data"""
    return {
        "timestamp": "2025-07-12T07:50:00Z",
        "devices": [],
        "activity": "none",
        "privacy_mode": True,
        "mode": "mock"
    }

# Mock WiFi endpoint
@app.get("/api/v1/consciousness/wifi")
async def get_wifi_awareness():
    """Get WiFi awareness data"""
    return {
        "timestamp": "2025-07-12T07:50:00Z",
        "networks": [
            {"ssid": "MockNetwork1", "signal": -45, "channel": 6},
            {"ssid": "MockNetwork2", "signal": -62, "channel": 11},
            {"ssid": "MockNetwork3", "signal": -78, "channel": 1}
        ],
        "csi_data": None,
        "privacy_mode": True,
        "mode": "mock"
    }

# Metrics endpoint
@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return """# HELP os4ai_consciousness_status OS4AI consciousness component status
# TYPE os4ai_consciousness_status gauge
os4ai_consciousness_status{component="thermal"} 1
os4ai_consciousness_status{component="acoustic"} 1
os4ai_consciousness_status{component="media"} 1
os4ai_consciousness_status{component="wifi"} 1

# HELP os4ai_system_info OS4AI system information
# TYPE os4ai_system_info counter
os4ai_system_info{version="4.0.0",mode="mock"} 1
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8003)