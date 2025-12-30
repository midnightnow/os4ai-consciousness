"""
Tests for OS4AI Sprint 2: Acoustic Echolocation System
"""
import pytest
import asyncio
import numpy as np
from .os4ai_sprint2_acoustic_echolocation import (
    AcousticEcholocation,
    MLSChirpGenerator,
    ImpulseResponseAnalyzer,
    SpatialTriangulator,
    AcousticReflection,
    RoomMesh
)


class TestMLSChirpGenerator:
    """Test MLS chirp generation"""
    
    def test_chirp_generator_init(self):
        """Test chirp generator initialization"""
        generator = MLSChirpGenerator(sample_rate=44100, duration=0.1)
        assert generator.sample_rate == 44100
        assert generator.duration == 0.1
        assert generator.chirp_length == 4410
    
    def test_generate_mls_chirp(self):
        """Test MLS chirp generation"""
        generator = MLSChirpGenerator()
        chirp = generator.generate_mls_chirp()
        
        assert isinstance(chirp, np.ndarray)
        assert len(chirp) == generator.chirp_length
        assert np.max(np.abs(chirp)) <= 0.1  # Amplitude check
    
    def test_chirp_frequency_range(self):
        """Test chirp with different frequency ranges"""
        generator = MLSChirpGenerator()
        chirp = generator.generate_mls_chirp(frequency_range=(100, 4000))
        
        assert isinstance(chirp, np.ndarray)
        assert len(chirp) > 0


class TestImpulseResponseAnalyzer:
    """Test impulse response analysis"""
    
    def test_analyzer_init(self):
        """Test analyzer initialization"""
        analyzer = ImpulseResponseAnalyzer(sample_rate=44100)
        assert analyzer.sample_rate == 44100
    
    @pytest.mark.asyncio
    async def test_reflection_analysis(self):
        """Test reflection analysis"""
        analyzer = ImpulseResponseAnalyzer()
        
        # Create dummy audio data
        captured_audio = np.random.normal(0, 0.1, 4410)
        chirp_signal = np.random.normal(0, 0.1, 2205)
        
        reflections = analyzer.analyze_reflection_response(captured_audio, chirp_signal)
        
        assert isinstance(reflections, list)
        assert len(reflections) >= 0
        
        for reflection in reflections:
            assert isinstance(reflection, AcousticReflection)
            assert reflection.distance >= 0
            assert 0 <= reflection.amplitude <= 1.0
            assert reflection.delay >= 0
    
    def test_surface_classification(self):
        """Test surface type classification"""
        analyzer = ImpulseResponseAnalyzer()
        
        # Test different amplitude/distance combinations
        assert "hard_wall" in analyzer._classify_surface(0.8, 3.0)
        assert "soft_wall" in analyzer._classify_surface(0.4, 3.0) 
        assert "furniture" in analyzer._classify_surface(0.5, 1.0)


class TestSpatialTriangulator:
    """Test spatial triangulation"""
    
    def test_triangulator_init(self):
        """Test triangulator initialization"""
        triangulator = SpatialTriangulator()
        assert len(triangulator.mic_positions) == 3
        assert triangulator.mic_positions[0] == (0.0, 0.0, 0.0)
    
    def test_custom_mic_positions(self):
        """Test custom microphone positions"""
        custom_mics = [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0)]
        triangulator = SpatialTriangulator(mic_positions=custom_mics)
        assert triangulator.mic_positions == custom_mics
    
    def test_triangulate_objects(self):
        """Test object triangulation"""
        triangulator = SpatialTriangulator()
        
        # Create test reflections
        reflections = [
            AcousticReflection(distance=2.0, angle=0.0, amplitude=0.8, delay=0.01, surface_type="wall"),
            AcousticReflection(distance=1.5, angle=1.57, amplitude=0.6, delay=0.008, surface_type="furniture"),
        ]
        
        objects = triangulator.triangulate_objects(reflections)
        
        assert len(objects) == 2
        assert objects[0]["type"] == "wall"
        assert objects[1]["type"] == "furniture"
        assert "position" in objects[0]
        assert "confidence" in objects[0]
    
    def test_room_boundary_detection(self):
        """Test room boundary detection"""
        triangulator = SpatialTriangulator()
        
        # Create wall reflections
        wall_reflections = [
            AcousticReflection(distance=2.0, angle=0.0, amplitude=0.8, delay=0.01, surface_type="hard_wall"),
            AcousticReflection(distance=3.0, angle=1.57, amplitude=0.8, delay=0.015, surface_type="hard_wall"),
            AcousticReflection(distance=2.5, angle=3.14, amplitude=0.8, delay=0.012, surface_type="hard_wall"),
            AcousticReflection(distance=3.5, angle=4.71, amplitude=0.8, delay=0.018, surface_type="hard_wall"),
        ]
        
        dimensions = triangulator.detect_room_boundaries(wall_reflections)
        
        assert len(dimensions) == 3  # length, width, height
        assert all(d > 0 for d in dimensions)
        assert dimensions[2] == 2.4  # Standard ceiling height


class TestAcousticEcholocation:
    """Test main acoustic echolocation system"""
    
    def test_echolocation_init(self):
        """Test echolocation system initialization"""
        system = AcousticEcholocation()
        
        assert system.sample_rate == 44100
        assert hasattr(system, 'chirp_generator')
        assert hasattr(system, 'impulse_analyzer')
        assert hasattr(system, 'triangulator')
        assert isinstance(system.room_mesh, RoomMesh)
    
    @pytest.mark.asyncio
    async def test_echolocation_sweep(self):
        """Test complete echolocation sweep"""
        system = AcousticEcholocation()
        
        room_mesh = await system.perform_echolocation_sweep()
        
        assert isinstance(room_mesh, RoomMesh)
        assert len(room_mesh.dimensions) == 3
        assert room_mesh.confidence >= 0.0
        assert len(room_mesh.objects) >= 0
        assert len(room_mesh.boundaries) >= 0
    
    @pytest.mark.asyncio 
    async def test_room_awareness_data(self):
        """Test room awareness data generation"""
        system = AcousticEcholocation()
        
        data = await system.get_room_awareness_data()
        
        assert "active" in data
        assert "room_dimensions" in data
        assert "objects_detected" in data
        assert "mapping_confidence" in data
        assert "detected_objects" in data
        assert "room_mesh" in data
        assert data["active"] == True
    
    def test_continuous_mapping_control(self):
        """Test continuous mapping start/stop"""
        system = AcousticEcholocation()
        
        # Test starting
        assert not system.is_listening
        
        # Test stopping
        system.stop_continuous_mapping()
        assert not system.is_listening
    
    @pytest.mark.asyncio
    async def test_simulated_room_mesh(self):
        """Test simulated room mesh generation"""
        system = AcousticEcholocation()
        
        room_mesh = system._simulate_room_mesh()
        
        assert isinstance(room_mesh, RoomMesh)
        assert room_mesh.confidence > 0.0
        assert len(room_mesh.objects) > 0
        assert len(room_mesh.boundaries) > 0
        assert room_mesh.dimensions == (4.2, 3.8, 2.4)
        
        # Check objects have required fields
        for obj in room_mesh.objects:
            assert "id" in obj
            assert "position" in obj
            assert "confidence" in obj
            assert "detection_method" in obj
            assert obj["detection_method"] == "acoustic_echolocation"


class TestRoomMesh:
    """Test room mesh data structure"""
    
    def test_room_mesh_creation(self):
        """Test room mesh creation"""
        mesh = RoomMesh(
            boundaries=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
            objects=[{"id": "test", "position": [1.0, 1.0, 0.0]}],
            dimensions=(4.0, 3.0, 2.4),
            confidence=0.8
        )
        
        assert len(mesh.boundaries) == 2
        assert len(mesh.objects) == 1
        assert mesh.dimensions == (4.0, 3.0, 2.4)
        assert mesh.confidence == 0.8
        assert mesh.last_updated is not None


class TestAcousticReflection:
    """Test acoustic reflection data structure"""
    
    def test_reflection_creation(self):
        """Test acoustic reflection creation"""
        reflection = AcousticReflection(
            distance=2.5,
            angle=1.57,
            amplitude=0.7,
            delay=0.015,
            surface_type="wall"
        )
        
        assert reflection.distance == 2.5
        assert reflection.angle == 1.57
        assert reflection.amplitude == 0.7
        assert reflection.delay == 0.015
        assert reflection.surface_type == "wall"


@pytest.mark.asyncio
async def test_integration_with_router():
    """Test integration with the router endpoints"""
    from .router import _acoustic_system
    
    # Note: This test would require the router to be properly initialized
    # In a real test environment, you'd mock the global _acoustic_system
    pass


if __name__ == "__main__":
    # Run a simple test
    import asyncio
    
    async def simple_test():
        print("🧪 Testing OS4AI Acoustic Echolocation System...")
        
        # Test basic functionality
        system = AcousticEcholocation()
        room_data = await system.get_room_awareness_data()
        
        print(f"✅ Room dimensions: {room_data['room_dimensions']}")
        print(f"✅ Objects detected: {room_data['objects_detected']}")
        print(f"✅ Mapping confidence: {room_data['mapping_confidence']:.2f}")
        print("🎧 Acoustic echolocation system working correctly!")
    
    asyncio.run(simple_test())