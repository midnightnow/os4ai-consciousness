#!/usr/bin/env python3
"""
OS4AI Integration API
FastAPI-based service exposing consciousness capabilities to ecosystem applications
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio
import json
import redis.asyncio as redis
from contextlib import asynccontextmanager
import uvicorn
import logging

from os4ai_ecosystem_integration import (
    OS4AIEcosystemOrchestrator,
    IntegrationMode,
    ConsciousnessContext
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models for API
class ClinicalRoutingRequest(BaseModel):
    """Request model for clinical routing enhancement"""
    command: str = Field(..., description="Clinical command to route")
    patient_context: Dict[str, Any] = Field(default_factory=dict)
    urgency: Optional[str] = Field(None, description="Urgency level override")
    
    class Config:
        schema_extra = {
            "example": {
                "command": "analyze chest x-ray for pneumonia",
                "patient_context": {
                    "patient_id": "P12345",
                    "age": 65,
                    "symptoms": ["cough", "fever"]
                },
                "urgency": "high"
            }
        }


class DevelopmentOperationRequest(BaseModel):
    """Request model for development operations"""
    operation: str = Field(..., description="Type of development operation")
    context: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        schema_extra = {
            "example": {
                "operation": "code_review",
                "context": {
                    "file_path": "/src/api/handlers.py",
                    "commit_hash": "abc123",
                    "author": "developer@example.com"
                }
            }
        }


class AppEnhancementRequest(BaseModel):
    """Request model for app enhancement"""
    app_id: str = Field(..., description="Unique app identifier")
    request_type: str = Field(..., description="Type of request from app")
    context: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        schema_extra = {
            "example": {
                "app_id": "vetclinic_mobile",
                "request_type": "get_appointments",
                "context": {
                    "user_id": "U789",
                    "date_range": "next_7_days"
                }
            }
        }


class ConsciousnessResponse(BaseModel):
    """Response model for consciousness-enhanced results"""
    request_id: str
    timestamp: str
    mode: str
    result: Dict[str, Any]
    consciousness_metrics: Dict[str, float]
    processing_time_ms: float


# Global orchestrator instance
orchestrator: Optional[OS4AIEcosystemOrchestrator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global orchestrator
    
    # Startup
    logger.info("🚀 Starting OS4AI Integration API")
    config = {
        "redis_url": "redis://localhost:6379",
        "monitoring_interval": 5.0
    }
    
    orchestrator = OS4AIEcosystemOrchestrator(config)
    await orchestrator.initialize()
    
    # Start background monitoring
    asyncio.create_task(continuous_monitoring())
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down OS4AI Integration API")
    if orchestrator:
        await orchestrator.shutdown()


# Create FastAPI app
app = FastAPI(
    title="OS4AI Integration API",
    description="Consciousness-enhanced services for Cursive, HardCard, and ecosystem apps",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"]  # TODO: Set actual domains,  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Check API health status"""
    return {
        "status": "healthy",
        "service": "OS4AI Integration API",
        "timestamp": datetime.utcnow().isoformat()
    }


# Clinical routing endpoints
@app.post("/api/v1/cursive/enhance-routing", response_model=ConsciousnessResponse)
async def enhance_clinical_routing(
    request: ClinicalRoutingRequest,
    background_tasks: BackgroundTasks
):
    """
    Enhance Cursive clinical routing with consciousness insights
    
    Returns routing recommendations with confidence scores and alternatives
    """
    start_time = datetime.utcnow()
    request_id = f"cursive_{int(start_time.timestamp() * 1000)}"
    
    try:
        result = await orchestrator.process_request(
            IntegrationMode.CURSIVE_CLINICAL,
            {
                "command": request.command,
                "patient_context": request.patient_context,
                "urgency": request.urgency
            }
        )
        
        # Track usage in background
        background_tasks.add_task(
            track_api_usage,
            "cursive_routing",
            request_id
        )
        
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return ConsciousnessResponse(
            request_id=request_id,
            timestamp=datetime.utcnow().isoformat(),
            mode="clinical_routing",
            result=result,
            consciousness_metrics={
                "confidence": result.get("consciousness_factors", {}).get("confidence", 0),
                "complexity": result.get("clinical_insights", {}).get("complexity_score", 0),
                "urgency_score": 1.0 if result.get("clinical_insights", {}).get("urgency_level") == "high" else 0.5
            },
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Clinical routing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Development operations endpoints
@app.post("/api/v1/hardcard/enhance-development", response_model=ConsciousnessResponse)
async def enhance_development_operation(
    request: DevelopmentOperationRequest,
    background_tasks: BackgroundTasks
):
    """
    Enhance HardCard development operations with consciousness insights
    
    Supports: code_review, performance_optimization, security_audit, deployment_readiness
    """
    start_time = datetime.utcnow()
    request_id = f"hardcard_{int(start_time.timestamp() * 1000)}"
    
    try:
        result = await orchestrator.process_request(
            IntegrationMode.HARDCARD_DEV,
            {
                "operation": request.operation,
                **request.context
            }
        )
        
        background_tasks.add_task(
            track_api_usage,
            "hardcard_dev",
            request_id
        )
        
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Extract consciousness metrics based on operation type
        consciousness_metrics = {}
        if request.operation == "code_review":
            consciousness_metrics = {
                "code_harmony": result.get("ai_insights", {}).get("code_harmony", 0),
                "complexity": result.get("metrics", {}).get("complexity_score", 0) / 100,
                "maintainability": result.get("metrics", {}).get("maintainability_index", 0) / 100
            }
        elif request.operation == "security_audit":
            threat_level_scores = {"low": 0.2, "medium": 0.6, "high": 0.9, "critical": 1.0}
            consciousness_metrics = {
                "threat_level": threat_level_scores.get(result.get("threat_level", "low"), 0.2),
                "anomaly_count": len(result.get("anomalies", [])),
                "vulnerability_count": len(result.get("vulnerabilities", []))
            }
        
        return ConsciousnessResponse(
            request_id=request_id,
            timestamp=datetime.utcnow().isoformat(),
            mode="development_enhancement",
            result=result,
            consciousness_metrics=consciousness_metrics,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Development operation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# App consumer endpoints
@app.post("/api/v1/apps/enhance-experience", response_model=ConsciousnessResponse)
async def enhance_app_experience(
    request: AppEnhancementRequest,
    background_tasks: BackgroundTasks
):
    """
    Enhance third-party app experience with consciousness-driven optimizations
    
    Provides predictive caching, resource optimization, and performance improvements
    """
    start_time = datetime.utcnow()
    request_id = f"app_{request.app_id}_{int(start_time.timestamp() * 1000)}"
    
    try:
        result = await orchestrator.process_request(
            IntegrationMode.APP_CONSUMER,
            {
                "app_id": request.app_id,
                "request_type": request.request_type,
                **request.context
            }
        )
        
        background_tasks.add_task(
            track_api_usage,
            f"app_{request.app_id}",
            request_id
        )
        
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return ConsciousnessResponse(
            request_id=request_id,
            timestamp=datetime.utcnow().isoformat(),
            mode="app_enhancement",
            result=result,
            consciousness_metrics={
                "optimization_score": len(result.get("optimizations", {}).get("applied", [])) / 5,
                "performance_boost": (
                    result.get("performance_boost", {}).get("latency_reduction", "0%").rstrip("%")
                ) / 100,
                "confidence": result.get("consciousness_insights", {}).get("confidence", 0)
            },
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"App enhancement error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Monitoring endpoints
@app.get("/api/v1/monitoring/ecosystem-health")
async def get_ecosystem_health():
    """
    Get real-time ecosystem health status with consciousness correlation
    """
    try:
        result = await orchestrator.process_request(
            IntegrationMode.MONITORING,
            {}
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Monitoring error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Analytics endpoints
@app.post("/api/v1/analytics/consciousness-insights")
async def get_consciousness_insights(
    time_range: str = "last_hour",
    systems: Optional[List[str]] = None
):
    """
    Get consciousness-driven analytics insights
    """
    try:
        result = await orchestrator.process_request(
            IntegrationMode.ANALYTICS,
            {
                "time_range": time_range,
                "systems": systems or ["all"]
            }
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket endpoint for real-time consciousness streaming
@app.websocket("/ws/consciousness-stream")
async def consciousness_websocket(websocket):
    """
    WebSocket endpoint for real-time consciousness updates
    """
    await websocket.accept()
    
    try:
        while True:
            # Generate consciousness update
            consciousness_data = await orchestrator._generate_consciousness_context({})
            
            update = {
                "type": "consciousness_update",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "confidence": consciousness_data.confidence_score,
                    "thermal_avg": sum(consciousness_data.thermal_signature.values()) / len(consciousness_data.thermal_signature),
                    "acoustic_harmony": consciousness_data.acoustic_pattern.get("harmonic_ratio", 0),
                    "network_nodes": len(consciousness_data.wifi_topology.get("nodes", [])),
                    "decision_factors": consciousness_data.decision_factors[:3]
                }
            }
            
            await websocket.send_json(update)
            await asyncio.sleep(1)  # Update every second
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()


# Batch processing endpoint
@app.post("/api/v1/batch/process")
async def process_batch_requests(
    requests: List[Dict[str, Any]],
    mode: str = "parallel"
):
    """
    Process multiple requests in batch
    
    Modes: parallel (default), sequential
    """
    results = []
    
    if mode == "parallel":
        # Process requests in parallel
        tasks = []
        for req in requests:
            integration_mode = IntegrationMode(req.get("mode", "analytics"))
            task = orchestrator.process_request(
                integration_mode,
                req.get("context", {})
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
    else:  # sequential
        for req in requests:
            integration_mode = IntegrationMode(req.get("mode", "analytics"))
            result = await orchestrator.process_request(
                integration_mode,
                req.get("context", {})
            )
            results.append(result)
    
    return {
        "batch_id": f"batch_{int(datetime.utcnow().timestamp() * 1000)}",
        "mode": mode,
        "total_requests": len(requests),
        "successful": sum(1 for r in results if not isinstance(r, Exception)),
        "results": results
    }


# Helper functions
async def track_api_usage(endpoint: str, request_id: str):
    """Track API usage for analytics"""
    try:
        # Store usage data in Redis
        usage_data = {
            "endpoint": endpoint,
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Implementation would store in Redis
        logger.info(f"Tracked usage: {endpoint} - {request_id}")
        
    except Exception as e:
        logger.error(f"Usage tracking error: {e}")


async def continuous_monitoring():
    """Background task for continuous monitoring"""
    while True:
        try:
            # Run ecosystem health check
            health = await orchestrator.process_request(
                IntegrationMode.MONITORING,
                {}
            )
            
            # Log critical alerts
            for alert in health.get("alerts", []):
                if alert.get("severity") == "critical":
                    logger.error(f"CRITICAL ALERT: {alert}")
            
            await asyncio.sleep(30)  # Check every 30 seconds
            
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
            await asyncio.sleep(60)  # Back off on error


# API documentation enhancement
@app.get("/api/v1/documentation/integration-guide")
async def get_integration_guide():
    """
    Get integration guide for using OS4AI services
    """
    return {
        "title": "OS4AI Integration Guide",
        "version": "1.0.0",
        "endpoints": {
            "clinical_routing": {
                "path": "/api/v1/cursive/enhance-routing",
                "method": "POST",
                "description": "Enhance clinical command routing with consciousness insights",
                "benefits": [
                    "AI-powered expert selection",
                    "Confidence-based routing",
                    "Alternative expert suggestions",
                    "Resource requirement prediction"
                ]
            },
            "development_ops": {
                "path": "/api/v1/hardcard/enhance-development",
                "method": "POST",
                "description": "Enhance development operations with consciousness analysis",
                "operations": [
                    "code_review - AI-enhanced code quality analysis",
                    "performance_optimization - Consciousness-based performance insights",
                    "security_audit - Anomaly detection using consciousness patterns",
                    "deployment_readiness - System harmony assessment"
                ]
            },
            "app_enhancement": {
                "path": "/api/v1/apps/enhance-experience",
                "method": "POST",
                "description": "Optimize third-party app performance",
                "features": [
                    "Predictive request caching",
                    "Resource pre-allocation",
                    "Performance optimization suggestions",
                    "Usage pattern learning"
                ]
            },
            "monitoring": {
                "path": "/api/v1/monitoring/ecosystem-health",
                "method": "GET",
                "description": "Real-time ecosystem health monitoring",
                "provides": [
                    "System health status",
                    "Consciousness correlation analysis",
                    "Alert generation",
                    "Performance metrics"
                ]
            },
            "websocket": {
                "path": "/ws/consciousness-stream",
                "protocol": "WebSocket",
                "description": "Real-time consciousness updates",
                "update_frequency": "1 second"
            }
        },
        "authentication": "Include API key in Authorization header",
        "rate_limits": {
            "standard": "1000 requests/hour",
            "websocket": "Unlimited while connected",
            "batch": "100 requests per batch"
        },
        "support": "support@os4ai.example.com"
    }


if __name__ == "__main__":
    uvicorn.run(
        "os4ai_integration_api:app",
        host="0.0.0.0",
        port=8005,
        reload=True,
        log_level="info"
    )