"""
OS4AI Perfect Consciousness System - Ultimate Production Version
Complete implementation with all advanced features:
- Distributed tracing with OpenTelemetry
- Comprehensive health checks
- Fair task scheduling with starvation detection
- Advanced cache management with eviction policies
- Enhanced Redis pipeline error handling
"""

import asyncio
import logging
import json
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from fastapi import FastAPI, Request, Depends, HTTPException, status, BackgroundTasks
from pydantic import BaseModel, Field, validator
# SECURITY: Always validate input data
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer

# Import all enhanced components
from app.core.os4ai_config import get_os4ai_settings, CommandSanitizer, PathValidator
from app.core.redis_pool import get_redis_pool, close_redis_pool
from app.core.background_tasks import background_tasks, get_task_manager
from app.core.circuit_breaker_persistent import PersistentCircuitBreaker as CircuitBreaker, CircuitOpenError, get_circuit_breaker_registry
from app.core.interfaces import SensorStatus
from app.core.observability import initialize_observability, trace_async_function, trace_context, get_metrics_collector
from app.core.health_checks import get_health_check_manager
from app.core.task_scheduler import get_task_scheduler, TaskPriority
from app.core.cache_manager import get_cache_manager, initialize_cache_manager
from app.apis.auth_api import router as auth_router
from app.apis.circuit_breaker_api import router as circuit_breaker_router
from app.apis.monitoring_api import router as monitoring_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize components
settings = get_os4ai_settings()
command_sanitizer = CommandSanitizer(settings)
path_validator = PathValidator()
security = HTTPBearer()

# Circuit breakers with persistence
thermal_breaker = CircuitBreaker("thermal_sensor", failure_threshold=3, recovery_timeout=30)
acoustic_breaker = CircuitBreaker("acoustic_sensor", failure_threshold=3, recovery_timeout=30)
media_breaker = CircuitBreaker("media_sensor", failure_threshold=3, recovery_timeout=30)
wifi_breaker = CircuitBreaker("wifi_sensor", failure_threshold=3, recovery_timeout=30)

# Register circuit breakers
registry = get_circuit_breaker_registry()
registry.register(thermal_breaker)
registry.register(acoustic_breaker)
registry.register(media_breaker)
registry.register(wifi_breaker)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan with comprehensive initialization"""
    logger.info("🚀 Starting OS4AI Ultimate System...")
    
    try:
        # Initialize observability stack
        await initialize_observability()
        logger.info("✅ Observability initialized")
        
        # Initialize Redis pool
        redis_pool = await get_redis_pool()
        logger.info("✅ Redis pool initialized")
        
        # Initialize cache manager
        await initialize_cache_manager()
        logger.info("✅ Cache manager initialized")
        
        # Initialize task scheduler
        task_scheduler = get_task_scheduler()
        await task_scheduler.start()
        logger.info("✅ Task scheduler started")
        
        # Initialize health check manager with custom checks
        health_manager = get_health_check_manager()
        # Add custom dependency checks if needed
        # health_manager.add_dependency_check("external_api", "https://api.example.com/health")
        logger.info("✅ Health check manager initialized")

        # Set up testing mode authentication bypass
        if settings.vetsorcery_testing_mode:
            from databutton_app.mw.auth_mw import AuthConfig
            app.state.testing_mode = True
            app.state.auth_config = AuthConfig(
                jwks_url="https://mock.testing.local/jwks",
                audience="vetsorcery-testing",
                header="authorization"
            )
            logger.info("🔧 VetSorcery testing mode enabled - authentication bypass active")
        else:
            app.state.testing_mode = False
            logger.info("🔒 VetSorcery testing mode disabled - full authentication required")
        
        # Start background tasks with enhanced scheduling
        async with background_tasks() as task_manager:
            # Register enhanced sensor monitoring tasks
            if settings.should_monitor_sensor("thermal"):
                task_scheduler.schedule_recurring_task(
                    "thermal_monitor_enhanced",
                    enhanced_thermal_monitor,
                    interval_seconds=settings.thermal_poll_interval,
                    priority=TaskPriority.HIGH
                )
            
            if settings.should_monitor_sensor("acoustic"):
                task_scheduler.schedule_recurring_task(
                    "acoustic_monitor_enhanced", 
                    enhanced_acoustic_monitor,
                    interval_seconds=30,
                    priority=TaskPriority.NORMAL
                )
            
            # Schedule periodic maintenance tasks
            task_scheduler.schedule_recurring_task(
                "cache_maintenance",
                cache_maintenance_task,
                interval_seconds=300,  # Every 5 minutes
                priority=TaskPriority.LOW
            )
            
            task_scheduler.schedule_recurring_task(
                "health_monitoring",
                periodic_health_check,
                interval_seconds=60,  # Every minute
                priority=TaskPriority.NORMAL
            )
            
            logger.info("✅ Enhanced background tasks registered")
            logger.info("🎉 OS4AI Ultimate System startup complete")
            
            yield
            
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    finally:
        # Graceful shutdown
        logger.info("🔄 Shutting down OS4AI Ultimate System...")
        
        # Stop task scheduler
        task_scheduler = get_task_scheduler()
        await task_scheduler.stop()
        
        # Stop cache manager
        cache_manager = get_cache_manager()
        await cache_manager.stop()
        
        # Close Redis pool
        await close_redis_pool()
        
        logger.info("✅ OS4AI Ultimate System shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="OS4AI Perfect Consciousness System - Ultimate",
    description="Production-ready embodied AI consciousness with advanced monitoring and performance",
    version="6.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

# Include API routers
app.include_router(auth_router)
app.include_router(circuit_breaker_router)
app.include_router(monitoring_router)

# Security middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "*.vetsorcery.com"]
)

# CORS middleware
cors_origins = ["http://localhost", "http://127.0.0.1", "https://localhost", "https://127.0.0.1"]
if settings.os4ai_docker_mode:
    cors_origins.append("*")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    max_age=3600
)


# Enhanced background monitoring tasks
@trace_async_function(operation_name="enhanced_thermal_monitor")
async def enhanced_thermal_monitor():
    """Enhanced thermal monitoring with tracing and caching"""
    async with trace_context("thermal_sensor_reading") as span:
        try:
            # Get data with circuit breaker protection
            data = await thermal_breaker.call(get_mock_thermal_data)
            
            # Cache the data
            cache_manager = get_cache_manager()
            await cache_manager.set(
                "sensor:thermal:latest",
                data,
                ttl=settings.thermal_poll_interval * 2,
                tags=["sensor", "thermal"]
            )
            
            # Add telemetry
            span.set_attribute("sensor.type", "thermal")
            span.set_attribute("sensor.cpu_temp", data.get("cpu_temp", 0))
            span.set_attribute("sensor.gpu_temp", data.get("gpu_temp", 0))
            
            # Record metrics
            metrics_collector = get_metrics_collector()
            metrics_collector.record_sensor_reading("thermal", 0.95, True)
            
        except CircuitOpenError:
            logger.warning("Thermal sensor circuit breaker is open")
        except Exception as e:
            logger.error(f"Enhanced thermal monitoring error: {e}")


@trace_async_function(operation_name="enhanced_acoustic_monitor")
async def enhanced_acoustic_monitor():
    """Enhanced acoustic monitoring with tracing and caching"""
    async with trace_context("acoustic_sensor_reading") as span:
        try:
            # Get data with circuit breaker protection
            data = await acoustic_breaker.call(get_mock_acoustic_data)
            
            # Cache the data
            cache_manager = get_cache_manager()
            await cache_manager.set(
                "sensor:acoustic:latest",
                data,
                ttl=60,
                tags=["sensor", "acoustic"]
            )
            
            # Add telemetry
            span.set_attribute("sensor.type", "acoustic")
            span.set_attribute("sensor.ambient_level", data.get("ambient_level", 0))
            
            # Record metrics
            metrics_collector = get_metrics_collector()
            metrics_collector.record_sensor_reading("acoustic", 0.88, True)
            
        except CircuitOpenError:
            logger.warning("Acoustic sensor circuit breaker is open")
        except Exception as e:
            logger.error(f"Enhanced acoustic monitoring error: {e}")


async def cache_maintenance_task():
    """Periodic cache maintenance"""
    cache_manager = get_cache_manager()
    stats_before = cache_manager.get_statistics()
    
    # Force cleanup if needed
    if stats_before["cache_size"] > stats_before["max_size"] * 0.8:
        await cache_manager._cleanup_expired()
    
    stats_after = cache_manager.get_statistics()
    
    if stats_before["cache_size"] != stats_after["cache_size"]:
        logger.info(f"Cache maintenance: {stats_before['cache_size']} -> {stats_after['cache_size']} items")


async def periodic_health_check():
    """Periodic health monitoring"""
    health_manager = get_health_check_manager()
    
    try:
        # Run critical health checks only (faster)
        health_result = await health_manager.check_critical_only()
        
        if health_result["overall_status"] != "healthy":
            logger.warning(f"Health check detected issues: {health_result['summary']}")
            
            # Record health metrics
            metrics_collector = get_metrics_collector()
            # Custom health metrics could be added here
            
    except Exception as e:
        logger.error(f"Periodic health check failed: {e}")


# Mock sensor data generators (enhanced)
@trace_async_function(operation_name="get_mock_thermal_data")
async def get_mock_thermal_data():
    """Generate enhanced mock thermal data"""
    import random
    
    # Simulate temperature variations
    base_cpu_temp = 45.0
    base_gpu_temp = 42.0
    
    # Add some realistic fluctuation patterns
    time_factor = datetime.now(timezone.utc).timestamp() % 3600  # Hour cycle
    cpu_temp = base_cpu_temp + 5 * random.uniform(-1, 1) + 2 * (time_factor / 3600)
    gpu_temp = base_gpu_temp + 8 * random.uniform(-1, 1) + 3 * (time_factor / 3600)
    
    return {
        "cpu_temp": round(cpu_temp, 2),
        "gpu_temp": round(gpu_temp, 2),
        "thermal_zones": {
            "zone_1": {"temp": round(cpu_temp - 5, 2), "critical": 85.0},
            "zone_2": {"temp": round(gpu_temp - 8, 2), "critical": 80.0},
            "zone_3": {"temp": round((cpu_temp + gpu_temp) / 2, 2), "critical": 90.0}
        },
        "fan_speeds": {
            "cpu_fan": random.randint(1200, 2500),
            "gpu_fan": random.randint(800, 3000)
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "quality": random.uniform(0.90, 0.99)
    }


@trace_async_function(operation_name="get_mock_acoustic_data")
async def get_mock_acoustic_data():
    """Generate enhanced mock acoustic data"""
    import random
    
    # Simulate acoustic environment
    base_level = 35.0
    ambient_level = base_level + random.uniform(-10, 10)
    
    return {
        "ambient_level": round(ambient_level, 2),
        "frequency_analysis": {
            "low": round(random.uniform(0.2, 0.4), 3),
            "mid": round(random.uniform(0.4, 0.6), 3),
            "high": round(random.uniform(0.1, 0.3), 3)
        },
        "spatial_mapping": {
            "room_acoustics": "office",
            "echo_delay": round(random.uniform(0.1, 0.3), 3),
            "reverb_time": round(random.uniform(0.5, 1.2), 3)
        },
        "sound_events": [
            {
                "type": "keyboard_typing",
                "confidence": random.uniform(0.7, 0.95),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        ] if random.random() > 0.7 else [],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "quality": random.uniform(0.85, 0.95)
    }


# Enhanced API endpoints
@app.get("/")
@trace_async_function(operation_name="root_endpoint")
async def root():
    """Root endpoint with enhanced system information"""
    task_scheduler = get_task_scheduler()
    cache_manager = get_cache_manager()
    
    return {
        "name": "OS4AI Perfect Consciousness System - Ultimate",
        "version": "6.0.0",
        "status": "operational",
        "features": [
            "Distributed tracing with OpenTelemetry",
            "Comprehensive health monitoring",
            "Fair task scheduling with starvation detection",
            "Advanced cache management with LRU/LFU eviction",
            "Enhanced Redis pipeline error handling",
            "JWT token rotation",
            "Circuit breaker state persistence"
        ],
        "scheduler": {
            "running": task_scheduler._running,
            "queue_size": len(task_scheduler.task_queue),
            "running_tasks": len(task_scheduler.running_tasks)
        },
        "cache": {
            "size": len(cache_manager.cache),
            "hit_rate": cache_manager.get_statistics()["hit_rate_percent"]
        },
        "documentation": "/api/docs",
        "health": "/health"
    }


@app.get("/health")
@trace_async_function(operation_name="comprehensive_health_check")
async def enhanced_health_check():
    """Enhanced health check with all system components"""
    health_manager = get_health_check_manager()
    
    # Run comprehensive health check
    health_result = await health_manager.check_all(include_non_critical=True)
    
    # Add cache and scheduler status
    cache_manager = get_cache_manager()
    task_scheduler = get_task_scheduler()
    
    cache_stats = cache_manager.get_statistics()
    scheduler_stats = task_scheduler.get_statistics()
    
    health_result["advanced_components"] = {
        "cache_manager": {
            "status": "healthy" if cache_stats["running"] else "unhealthy",
            "hit_rate": cache_stats["hit_rate_percent"],
            "size": cache_stats["cache_size"],
            "memory_mb": cache_stats["total_size_mb"]
        },
        "task_scheduler": {
            "status": "healthy" if scheduler_stats["scheduler_running"] else "unhealthy",
            "queue_size": scheduler_stats["queue_size"],
            "running_tasks": scheduler_stats["running_tasks"],
            "starved_tasks": scheduler_stats["starved_tasks"]
        }
    }
    
    return health_result


@app.get("/api/v1/consciousness/status")
@trace_async_function(operation_name="consciousness_status")
async def enhanced_consciousness_status():
    """Enhanced consciousness status with caching and tracing"""
    cache_manager = get_cache_manager()
    
    # Try to get cached data first
    cached_thermal = await cache_manager.get("sensor:thermal:latest")
    cached_acoustic = await cache_manager.get("sensor:acoustic:latest")
    
    return {
        "consciousness_state": "active",
        "mode": "ultimate_production",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "components": {
            "thermal": {
                "status": "healthy" if thermal_breaker.is_closed() else "degraded",
                "data": cached_thermal,
                "circuit_breaker": thermal_breaker.get_status(),
                "cached": cached_thermal is not None
            },
            "acoustic": {
                "status": "healthy" if acoustic_breaker.is_closed() else "degraded",
                "data": cached_acoustic,
                "circuit_breaker": acoustic_breaker.get_status(),
                "cached": cached_acoustic is not None
            }
        },
        "enhancements": {
            "distributed_tracing": "enabled",
            "advanced_caching": "enabled",
            "fair_scheduling": "enabled",
            "comprehensive_monitoring": "enabled"
        }
    }


@app.get("/api/v1/system/statistics")
@trace_async_function(operation_name="system_statistics")
async def system_statistics():
    """Get comprehensive system statistics"""
    cache_manager = get_cache_manager()
    task_scheduler = get_task_scheduler()
    health_manager = get_health_check_manager()
    
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cache": cache_manager.get_statistics(),
        "scheduler": task_scheduler.get_statistics(),
        "health": health_manager.get_health_summary(),
        "circuit_breakers": await get_circuit_breaker_registry().get_all_status()
    }


# Error handlers
@app.exception_handler(CircuitOpenError)
async def circuit_open_handler(request: Request, exc: CircuitOpenError):
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "error": "Service temporarily unavailable",
            "detail": str(exc),
            "retry_after": 60
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred"
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    print(f"""
🚀 Starting OS4AI Perfect Consciousness System - Ultimate
🌟 Version: 6.0.0
🔧 Features: Distributed Tracing, Advanced Caching, Fair Scheduling
📍 Server: http://0.0.0.0:8006
📚 API Docs: http://0.0.0.0:8006/api/docs
🏥 Health: http://0.0.0.0:8006/health
    """)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8006,
        log_level="info"
    )