"""
Consciousness Hypervisor Package
The virtualization layer for consciousness entities
"""

from .consciousness_vm_manager import (
    ConsciousnessHypervisor,
    ConsciousnessEntity, 
    ConsciousnessManifest,
    ConsciousnessState,
    SensoryAllocation
)
from .router import router

__all__ = [
    'ConsciousnessHypervisor',
    'ConsciousnessEntity',
    'ConsciousnessManifest', 
    'ConsciousnessState',
    'SensoryAllocation',
    'router'
]