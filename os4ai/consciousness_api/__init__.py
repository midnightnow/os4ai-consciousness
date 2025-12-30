"""OS4AI Embodied Consciousness API Module"""

from .embodied_substrate import (
    StructuralResonance, 
    EmbodiedOS4AI, 
    ThermalProprioception,
    AcousticEcholocation,
    WiFiRadarSensing,
    USBCRadioTelescope,
    CosmicSignalDetector
)
from .router import router, lifespan

__all__ = [
    'StructuralResonance', 
    'EmbodiedOS4AI', 
    'ThermalProprioception',
    'AcousticEcholocation',
    'WiFiRadarSensing',
    'USBCRadioTelescope',
    'CosmicSignalDetector',
    'router', 
    'lifespan'
]