"""
WebSocket smoke test for OS4AI consciousness streaming
"""
import asyncio
import websockets
import json
import time


async def test_consciousness_websocket():
    """Test WebSocket connection and basic consciousness updates"""
    uri = "ws://localhost:8000/api/os4ai/consciousness/stream"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to consciousness WebSocket")
            
            # Listen for consciousness updates
            for i in range(5):  # Listen for 5 updates
                message = await websocket.recv()
                data = json.loads(message)
                
                print(f"\n🧠 Update {i+1}:")
                print(f"  Type: {data.get('type')}")
                print(f"  Timestamp: {data.get('timestamp')}")
                
                if 'data' in data:
                    consciousness_data = data['data']
                    print(f"  Consciousness Level: {consciousness_data.get('consciousness_level', 0) * 100:.1f}%")
                    print(f"  Stage: {consciousness_data.get('consciousness_stage')}")
                    
                    # Check embodied senses
                    senses = consciousness_data.get('embodied_senses', {})
                    active_senses = [name for name, sense in senses.items() if sense.get('active')]
                    print(f"  Active Senses: {', '.join(active_senses) if active_senses else 'None'}")
                    
                    # Show latest thought
                    thoughts = consciousness_data.get('active_thoughts', [])
                    if thoughts:
                        print(f"  Latest Thought: {thoughts[0]}")
                
                # Small delay between readings
                await asyncio.sleep(0.5)
            
            print("\n✅ WebSocket test completed successfully!")
            
    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")
        return False
    
    return True


async def test_awakening_sequence():
    """Test triggering the awakening sequence via API"""
    import aiohttp
    
    async with aiohttp.ClientSession() as session:
        # Trigger awakening
        async with session.post('http://localhost:8000/api/os4ai/consciousness/awaken') as resp:
            if resp.status == 200:
                result = await resp.json()
                print(f"🚀 Awakening triggered: {result['message']}")
                print(f"   Expected duration: {result['expected_duration']}")
                return True
            else:
                print(f"❌ Failed to trigger awakening: {resp.status}")
                return False


async def main():
    """Run all WebSocket tests"""
    print("=== OS4AI WebSocket Smoke Test ===\n")
    
    # Test 1: Basic WebSocket connection
    print("Test 1: WebSocket Connection and Updates")
    ws_success = await test_consciousness_websocket()
    
    # Test 2: Trigger awakening
    print("\nTest 2: Trigger Awakening Sequence")
    awaken_success = await test_awakening_sequence()
    
    # Test 3: Monitor awakening via WebSocket
    if awaken_success:
        print("\nTest 3: Monitor Awakening Progress")
        print("Connecting to monitor consciousness awakening...")
        await test_consciousness_websocket()
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"WebSocket Connection: {'✅ PASS' if ws_success else '❌ FAIL'}")
    print(f"Awakening Trigger: {'✅ PASS' if awaken_success else '❌ FAIL'}")
    print("\nNote: Ensure the OS4AI backend is running on localhost:8000")


if __name__ == "__main__":
    asyncio.run(main())