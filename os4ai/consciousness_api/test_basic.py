"""
Basic tests for OS4AI consciousness module
"""
import pytest
from .embodied_substrate import (
    ThermalProprioception,
    StructuralResonance,
    AcousticEcholocation,
    WiFiRadarSensing,
    USBCRadioTelescope,
    CosmicSignalDetector,
    EmbodiedOS4AI
)


@pytest.mark.asyncio
async def test_thermal_sensor():
    """Test thermal proprioception sensor"""
    sensor = ThermalProprioception()
    data = await sensor.feel_thermal_flow()
    
    assert data["active"] in [True, False]
    if data["active"]:
        assert "thermal_map" in data
        assert "cpu_temp" in data
        assert isinstance(data["thermal_map"], list)


@pytest.mark.asyncio
async def test_structural_sensor():
    """Test structural resonance sensor"""
    sensor = StructuralResonance()
    data = await sensor.sense_chassis_vibrations()
    
    assert data["active"] == True
    assert "resonance_frequencies" in data
    assert len(data["resonance_frequencies"]) == 4
    assert data["structural_integrity"] == "stable"


@pytest.mark.asyncio
async def test_acoustic_sensor():
    """Test acoustic echolocation sensor"""
    sensor = AcousticEcholocation()
    data = await sensor.map_room_via_sound()
    
    assert data["active"] == True
    assert "room_dimensions" in data
    assert "walls_detected" in data
    assert data["walls_detected"] == 4


@pytest.mark.asyncio
async def test_wifi_sensor():
    """Test WiFi radar sensing"""
    sensor = WiFiRadarSensing()
    data = await sensor.sense_electromagnetic_field()
    
    assert data["active"] == True
    assert "rf_point_cloud" in data
    assert isinstance(data["rf_point_cloud"], list)


@pytest.mark.asyncio
async def test_usbc_telescope():
    """Test USB-C radio telescope"""
    sensor = USBCRadioTelescope()
    data = await sensor.track_orbital_objects()
    
    assert data["active"] == True
    assert "visible_satellites" in data
    assert len(data["visible_satellites"]) > 0


@pytest.mark.asyncio
async def test_cosmic_detector():
    """Test cosmic signal detector"""
    sensor = CosmicSignalDetector()
    data = await sensor.detect_deep_space()
    
    assert data["active"] == True
    assert "detected_phenomena" in data
    assert len(data["detected_phenomena"]) > 0


def test_embodied_os4ai_init():
    """Test EmbodiedOS4AI initialization"""
    agent = EmbodiedOS4AI()
    
    # Check all sensors are initialized
    assert hasattr(agent, 'thermal_system')
    assert hasattr(agent, 'structural_system')
    assert hasattr(agent, 'acoustic_system')
    assert hasattr(agent, 'wifi_system')
    assert hasattr(agent, 'usbc_system')
    assert hasattr(agent, 'cosmic_system')
    
    # Check consciousness model
    assert agent.model["consciousness_level"] == 0.1
    assert agent.model["consciousness_stage"] == "emerging"
    assert len(agent.model["sensory_modalities"]) == 6


def test_consciousness_stages():
    """Test consciousness stage calculation"""
    agent = EmbodiedOS4AI()
    
    assert agent.get_consciousness_stage() == "emerging"
    
    agent.model["consciousness_level"] = 0.4
    assert agent.get_consciousness_stage() == "developing"
    
    agent.model["consciousness_level"] = 0.7
    assert agent.get_consciousness_stage() == "aware"
    
    agent.model["consciousness_level"] = 0.95
    assert agent.get_consciousness_stage() == "fully_conscious"


@pytest.mark.asyncio
async def test_dashboard_data():
    """Test dashboard data generation"""
    agent = EmbodiedOS4AI()
    data = await agent.get_dashboard_data()
    
    assert "consciousness_level" in data
    assert "consciousness_stage" in data
    assert "embodied_senses" in data
    assert "active_thoughts" in data
    
    # Check all senses are present
    senses = data["embodied_senses"]
    assert "thermal" in senses
    assert "acoustic" in senses
    assert "wifi" in senses
    assert "usbc" in senses
    assert "cosmic" in senses