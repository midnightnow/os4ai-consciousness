#!/usr/bin/env python3
"""
OS4AI Enhanced Validation Suite
Combines existing deep validation with proposed integration and formal methods
"""

import asyncio
import networkx as nx
from typing import Dict, List, Any, Set, Tuple
import json
import time
import aiohttp
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntegrationValidator:
    """Validates integration between OS4AI, Captain Cursive, and Shipyard"""
    
    def __init__(self, base_urls: Dict[str, str]):
        self.base_urls = base_urls
        self.session = None
    
    async def setup(self):
        """Setup HTTP session for testing"""
        self.session = aiohttp.ClientSession()
    
    async def teardown(self):
        """Cleanup HTTP session"""
        if self.session:
            await self.session.close()
    
    async def test_api_integration_flow(self) -> Dict[str, Any]:
        """Test complete flow from OS4AI → Cursive → Shipyard"""
        logger.info("🔄 Testing API integration flow...")
        
        results = {
            "status": "PASS",
            "flow_time": 0,
            "errors": [],
            "consciousness_hash_consistency": True
        }
        
        start_time = time.time()
        
        try:
            # Step 1: Get consciousness state from OS4AI
            consciousness_state = await self._get_os4ai_consciousness()
            if not consciousness_state:
                results["errors"].append("Failed to get OS4AI consciousness state")
                results["status"] = "FAIL"
                return results
            
            initial_hash = consciousness_state.get("global_hash")
            
            # Step 2: Route command through Captain Cursive
            command = "analyze patient vitals and suggest treatment"
            routing_result = await self._route_through_cursive(command, consciousness_state)
            
            if not routing_result or routing_result.get("expert") is None:
                results["errors"].append("Captain Cursive routing failed")
                results["status"] = "FAIL"
                return results
            
            # Step 3: Deploy task to Shipyard
            deployment = await self._deploy_to_shipyard(
                routing_result["expert"],
                routing_result["command_id"],
                consciousness_state
            )
            
            if not deployment or deployment.get("vessel_id") is None:
                results["errors"].append("Shipyard deployment failed")
                results["status"] = "FAIL"
                return results
            
            # Step 4: Verify consciousness hash consistency
            final_consciousness = await self._get_os4ai_consciousness()
            final_hash = final_consciousness.get("global_hash")
            
            if initial_hash != final_hash:
                results["consciousness_hash_consistency"] = False
                results["errors"].append(f"Hash mismatch: {initial_hash} → {final_hash}")
            
            results["flow_time"] = time.time() - start_time
            
        except Exception as e:
            results["status"] = "FAIL"
            results["errors"].append(f"Integration test exception: {str(e)}")
            logger.error(f"Integration test failed: {e}")
        
        return results
    
    async def test_websocket_streaming(self) -> Dict[str, Any]:
        """Test real-time WebSocket data flow"""
        logger.info("📡 Testing WebSocket streaming...")
        
        results = {
            "status": "PASS",
            "latency_ms": 0,
            "data_integrity": True,
            "errors": []
        }
        
        try:
            # Connect to shipyard WebSocket stream
            ws_url = f"{self.base_urls['shipyard']}/ws/shipyard-stream"
            
            async with self.session.ws_connect(ws_url) as ws:
                # Send test message
                test_data = {
                    "type": "consciousness_update",
                    "timestamp": time.time(),
                    "data": {"test": True}
                }
                
                start = time.time()
                await ws.send_json(test_data)
                
                # Wait for echo/acknowledgment
                response = await asyncio.wait_for(ws.receive_json(), timeout=5.0)
                latency = (time.time() - start) * 1000  # Convert to ms
                
                results["latency_ms"] = latency
                
                # Verify data integrity
                if response.get("type") != "consciousness_update":
                    results["data_integrity"] = False
                    results["errors"].append("WebSocket data corruption detected")
                
                if latency > 100:  # >100ms is concerning
                    results["errors"].append(f"High WebSocket latency: {latency:.1f}ms")
                    results["status"] = "WARNING"
                    
        except asyncio.TimeoutError:
            results["status"] = "FAIL"
            results["errors"].append("WebSocket connection timeout")
        except Exception as e:
            results["status"] = "FAIL"
            results["errors"].append(f"WebSocket test failed: {str(e)}")
        
        return results
    
    async def test_state_consistency(self) -> Dict[str, Any]:
        """Test state consistency across all systems"""
        logger.info("🔍 Testing cross-system state consistency...")
        
        results = {
            "status": "PASS",
            "inconsistencies": [],
            "state_snapshot": {}
        }
        
        try:
            # Get state from each system
            os4ai_state = await self._get_system_state("os4ai")
            cursive_state = await self._get_system_state("cursive")
            shipyard_state = await self._get_system_state("shipyard")
            
            # Store snapshot
            results["state_snapshot"] = {
                "os4ai": os4ai_state,
                "cursive": cursive_state,
                "shipyard": shipyard_state
            }
            
            # Check consistency rules
            
            # Rule 1: Active vessel count should match
            os4ai_vessels = os4ai_state.get("active_vessels", 0)
            shipyard_vessels = len(shipyard_state.get("vessels", []))
            
            if os4ai_vessels != shipyard_vessels:
                results["inconsistencies"].append(
                    f"Vessel count mismatch: OS4AI={os4ai_vessels}, Shipyard={shipyard_vessels}"
                )
            
            # Rule 2: Consciousness hash should be synchronized
            os4ai_hash = os4ai_state.get("consciousness_hash")
            cursive_hash = cursive_state.get("consciousness_hash")
            
            if os4ai_hash != cursive_hash:
                results["inconsistencies"].append(
                    f"Consciousness hash mismatch between OS4AI and Cursive"
                )
            
            # Rule 3: Task queue sizes should be reasonable
            cursive_queue = cursive_state.get("command_queue_size", 0)
            shipyard_queue = shipyard_state.get("task_queue_size", 0)
            
            if abs(cursive_queue - shipyard_queue) > 10:
                results["inconsistencies"].append(
                    f"Queue size divergence: Cursive={cursive_queue}, Shipyard={shipyard_queue}"
                )
            
            if results["inconsistencies"]:
                results["status"] = "FAIL"
                
        except Exception as e:
            results["status"] = "FAIL"
            results["inconsistencies"].append(f"State check failed: {str(e)}")
        
        return results
    
    async def _get_os4ai_consciousness(self) -> Dict[str, Any]:
        """Get consciousness state from OS4AI"""
        try:
            async with self.session.get(
                f"{self.base_urls['os4ai']}/api/v1/consciousness/status"
            ) as response:
                if response.status == 200:
                    return await response.json()
        except Exception as e:
            logger.error(f"Failed to get OS4AI consciousness: {e}")
        return {}
    
    async def _route_through_cursive(self, command: str, context: Dict) -> Dict[str, Any]:
        """Route command through Captain Cursive"""
        try:
            payload = {
                "command": command,
                "context": context,
                "priority": "normal"
            }
            
            async with self.session.post(
                f"{self.base_urls['cursive']}/api/route",
                json=payload
            ) as response:
                if response.status == 200:
                    return await response.json()
        except Exception as e:
            logger.error(f"Cursive routing failed: {e}")
        return {}
    
    async def _deploy_to_shipyard(self, expert: str, command_id: str, context: Dict) -> Dict[str, Any]:
        """Deploy task to Shipyard"""
        try:
            payload = {
                "expert": expert,
                "command_id": command_id,
                "consciousness_context": context
            }
            
            async with self.session.post(
                f"{self.base_urls['shipyard']}/api/deploy",
                json=payload
            ) as response:
                if response.status == 200:
                    return await response.json()
        except Exception as e:
            logger.error(f"Shipyard deployment failed: {e}")
        return {}
    
    async def _get_system_state(self, system: str) -> Dict[str, Any]:
        """Get current state from a system"""
        try:
            async with self.session.get(
                f"{self.base_urls[system]}/api/state"
            ) as response:
                if response.status == 200:
                    return await response.json()
        except Exception as e:
            logger.error(f"Failed to get {system} state: {e}")
        return {}


class FormalLogicValidator:
    """Validates system logic using formal methods"""
    
    def __init__(self):
        self.routing_rules = {}
        self.state_invariants = []
    
    def validate_routing_correctness(self) -> Dict[str, Any]:
        """Use formal logic to prove routing correctness"""
        logger.info("🧮 Validating routing logic formally...")
        
        results = {
            "status": "PASS",
            "contradictions": [],
            "completeness_gaps": [],
            "ambiguities": []
        }
        
        # Define routing predicates
        routing_rules = {
            "patient_care": ["diagnosis", "treatment", "vitals", "medical"],
            "practice_flow": ["schedule", "appointment", "workflow", "capacity"],
            "client_comm": ["reminder", "follow-up", "education", "notification"],
            "business_ops": ["billing", "inventory", "reporting", "analytics"]
        }
        
        # Check for contradictions (command matching multiple experts)
        all_keywords = []
        for expert, keywords in routing_rules.items():
            for keyword in keywords:
                if keyword in all_keywords:
                    results["contradictions"].append(
                        f"Keyword '{keyword}' maps to multiple experts"
                    )
                all_keywords.append(keyword)
        
        # Check for completeness (common commands without routing)
        common_commands = [
            "check patient status",
            "update medical record",
            "send appointment reminder",
            "generate invoice"
        ]
        
        for command in common_commands:
            matched = False
            for expert, keywords in routing_rules.items():
                if any(keyword in command.lower() for keyword in keywords):
                    matched = True
                    break
            
            if not matched:
                results["completeness_gaps"].append(
                    f"No routing rule for command: '{command}'"
                )
        
        # Check for ambiguous patterns
        ambiguous_patterns = [
            ("schedule treatment", ["practice_flow", "patient_care"]),
            ("billing report", ["business_ops", "client_comm"])
        ]
        
        for pattern, possible_experts in ambiguous_patterns:
            matching_experts = []
            for expert, keywords in routing_rules.items():
                if any(keyword in pattern for keyword in keywords):
                    matching_experts.append(expert)
            
            if len(matching_experts) > 1:
                results["ambiguities"].append(
                    f"Pattern '{pattern}' matches {len(matching_experts)} experts"
                )
        
        if results["contradictions"] or results["completeness_gaps"] or results["ambiguities"]:
            results["status"] = "FAIL"
        
        return results
    
    def validate_state_invariants(self) -> Dict[str, Any]:
        """Validate state machine invariants"""
        logger.info("🔒 Validating state invariants...")
        
        results = {
            "status": "PASS",
            "violated_invariants": [],
            "unreachable_states": []
        }
        
        # Define state invariants
        invariants = [
            # Invariant 1: System cannot be processing without being ready
            lambda state: not (state["is_processing"] and not state["is_ready"]),
            
            # Invariant 2: Error state requires error message
            lambda state: not (state["is_error"] and not state.get("error_message")),
            
            # Invariant 3: Active vessels cannot exceed max capacity
            lambda state: state.get("active_vessels", 0) <= state.get("max_vessels", 10),
            
            # Invariant 4: Circuit breaker cannot be closed with high failure rate
            lambda state: not (
                state.get("circuit_breaker_state") == "closed" and 
                state.get("failure_rate", 0) > 0.5
            )
        ]
        
        # Test invariants with various states
        test_states = [
            {"is_processing": True, "is_ready": False},  # Should violate
            {"is_error": True, "error_message": None},  # Should violate
            {"active_vessels": 15, "max_vessels": 10},  # Should violate
            {"circuit_breaker_state": "closed", "failure_rate": 0.8}  # Should violate
        ]
        
        for i, state in enumerate(test_states):
            for j, invariant in enumerate(invariants):
                try:
                    if not invariant(state):
                        results["violated_invariants"].append(
                            f"Invariant {j+1} violated by state {i+1}"
                        )
                except Exception as e:
                    logger.warning(f"Invariant {j+1} check failed: {e}")
        
        # Check for unreachable states using graph analysis
        reachability_result = self._check_state_reachability()
        results["unreachable_states"] = reachability_result["unreachable"]
        
        if results["violated_invariants"] or results["unreachable_states"]:
            results["status"] = "FAIL"
        
        return results
    
    def _check_state_reachability(self) -> Dict[str, Any]:
        """Use graph theory to check state reachability"""
        # Build state transition graph
        G = nx.DiGraph()
        
        # Add states
        states = ["idle", "initializing", "ready", "processing", "error", "shutdown"]
        G.add_nodes_from(states)
        
        # Add transitions
        transitions = [
            ("idle", "initializing"),
            ("initializing", "ready"),
            ("initializing", "error"),
            ("ready", "processing"),
            ("ready", "idle"),
            ("processing", "ready"),
            ("processing", "error"),
            ("error", "initializing"),
            ("error", "shutdown"),
            ("ready", "shutdown"),
            ("idle", "shutdown")
        ]
        G.add_edges_from(transitions)
        
        # Find unreachable states from initial state
        reachable = nx.descendants(G, "idle")
        reachable.add("idle")  # Include starting state
        
        unreachable = set(states) - reachable
        
        return {
            "reachable": list(reachable),
            "unreachable": list(unreachable),
            "has_cycles": len(list(nx.simple_cycles(G))) > 0
        }


class GraphTheoryValidator:
    """Validates system dependencies using graph theory"""
    
    def __init__(self):
        self.dependency_graph = nx.DiGraph()
    
    def validate_task_dependencies(self) -> Dict[str, Any]:
        """Validate task dependencies for cycles and deadlocks"""
        logger.info("🔗 Validating task dependency graph...")
        
        results = {
            "status": "PASS",
            "cycles": [],
            "potential_deadlocks": [],
            "critical_paths": []
        }
        
        # Build task dependency graph
        self._build_dependency_graph()
        
        # Check for cycles
        try:
            cycles = list(nx.simple_cycles(self.dependency_graph))
            if cycles:
                results["cycles"] = [" → ".join(cycle + [cycle[0]]) for cycle in cycles]
                results["status"] = "FAIL"
        except nx.NetworkXNoCycle:
            pass
        
        # Check for potential deadlocks (nodes with circular wait)
        deadlock_candidates = self._find_deadlock_candidates()
        if deadlock_candidates:
            results["potential_deadlocks"] = deadlock_candidates
            results["status"] = "WARNING"
        
        # Find critical paths
        if nx.is_directed_acyclic_graph(self.dependency_graph):
            critical_paths = self._find_critical_paths()
            results["critical_paths"] = critical_paths
        
        return results
    
    def _build_dependency_graph(self):
        """Build task dependency graph"""
        # Example task dependencies
        tasks = {
            "consciousness_init": ["sensor_init", "redis_init"],
            "sensor_init": ["config_load"],
            "redis_init": ["config_load"],
            "api_start": ["consciousness_init", "auth_init"],
            "auth_init": ["redis_init"],
            "websocket_start": ["api_start"],
            "monitoring_start": ["api_start", "redis_init"]
        }
        
        for task, dependencies in tasks.items():
            for dep in dependencies:
                self.dependency_graph.add_edge(dep, task)
    
    def _find_deadlock_candidates(self) -> List[str]:
        """Find potential deadlock scenarios"""
        candidates = []
        
        # Look for nodes involved in cycles with resource dependencies
        for node in self.dependency_graph.nodes():
            predecessors = set(self.dependency_graph.predecessors(node))
            successors = set(self.dependency_graph.successors(node))
            
            # Check if any successor depends on a predecessor (potential circular wait)
            for succ in successors:
                succ_deps = set(self.dependency_graph.predecessors(succ))
                if predecessors & succ_deps:
                    candidates.append(
                        f"Potential deadlock: {node} → {succ} (circular dependency)"
                    )
        
        return candidates
    
    def _find_critical_paths(self) -> List[str]:
        """Find critical paths in the dependency graph"""
        critical_paths = []
        
        # Find all paths from sources to sinks
        sources = [n for n in self.dependency_graph.nodes() 
                  if self.dependency_graph.in_degree(n) == 0]
        sinks = [n for n in self.dependency_graph.nodes() 
                if self.dependency_graph.out_degree(n) == 0]
        
        for source in sources:
            for sink in sinks:
                try:
                    paths = list(nx.all_simple_paths(
                        self.dependency_graph, source, sink
                    ))
                    for path in paths:
                        if len(path) > 3:  # Only interested in longer paths
                            critical_paths.append(" → ".join(path))
                except nx.NetworkXNoPath:
                    pass
        
        return critical_paths[:5]  # Return top 5 critical paths


class EnhancedValidationOrchestrator:
    """Orchestrates all validation types including new approaches"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_urls = {
            "os4ai": config.get("os4ai_url", "http://localhost:8005"),
            "cursive": config.get("cursive_url", "http://localhost:8001"),
            "shipyard": config.get("shipyard_url", "http://localhost:8002")
        }
        
        # Initialize validators
        self.integration_validator = IntegrationValidator(self.base_urls)
        self.formal_logic_validator = FormalLogicValidator()
        self.graph_validator = GraphTheoryValidator()
    
    async def run_enhanced_validation(self) -> Dict[str, Any]:
        """Run complete enhanced validation suite"""
        logger.info("🚀 Starting enhanced OS4AI validation suite...")
        
        start_time = time.time()
        results = {
            "timestamp": time.time(),
            "validation_type": "enhanced",
            "categories": {}
        }
        
        # Setup integration validator
        await self.integration_validator.setup()
        
        try:
            # Run integration tests
            logger.info("🔄 Running integration validation...")
            integration_results = {
                "api_flow": await self.integration_validator.test_api_integration_flow(),
                "websocket": await self.integration_validator.test_websocket_streaming(),
                "state_consistency": await self.integration_validator.test_state_consistency()
            }
            results["categories"]["integration"] = integration_results
            
            # Run formal logic validation
            logger.info("🧮 Running formal logic validation...")
            formal_results = {
                "routing_logic": self.formal_logic_validator.validate_routing_correctness(),
                "state_invariants": self.formal_logic_validator.validate_state_invariants()
            }
            results["categories"]["formal_logic"] = formal_results
            
            # Run graph theory validation
            logger.info("🔗 Running graph theory validation...")
            graph_results = {
                "task_dependencies": self.graph_validator.validate_task_dependencies()
            }
            results["categories"]["graph_theory"] = graph_results
            
        finally:
            await self.integration_validator.teardown()
        
        # Calculate summary
        results["execution_time"] = time.time() - start_time
        results["summary"] = self._generate_summary(results["categories"])
        
        return results
    
    def _generate_summary(self, categories: Dict[str, Any]) -> Dict[str, Any]:
        """Generate validation summary"""
        total_tests = 0
        passed_tests = 0
        critical_issues = []
        
        for category, tests in categories.items():
            for test_name, test_result in tests.items():
                total_tests += 1
                if test_result.get("status") == "PASS":
                    passed_tests += 1
                elif test_result.get("status") == "FAIL":
                    critical_issues.append(f"{category}/{test_name}")
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            "critical_issues": critical_issues,
            "overall_status": "PASS" if not critical_issues else "FAIL"
        }


async def main():
    """Run enhanced validation suite"""
    config = {
        "os4ai_url": "http://localhost:8005",
        "cursive_url": "http://localhost:8001", 
        "shipyard_url": "http://localhost:8002"
    }
    
    orchestrator = EnhancedValidationOrchestrator(config)
    results = await orchestrator.run_enhanced_validation()
    
    # Print results
    print("\n" + "="*80)
    print("🔍 OS4AI ENHANCED VALIDATION RESULTS")
    print("="*80)
    
    summary = results["summary"]
    print(f"\n📊 Overall Status: {summary['overall_status']}")
    print(f"✅ Success Rate: {summary['success_rate']:.1f}%")
    print(f"🧪 Total Tests: {summary['total_tests']}")
    print(f"⏱️  Execution Time: {results['execution_time']:.2f}s")
    
    if summary["critical_issues"]:
        print(f"\n🚨 Critical Issues:")
        for issue in summary["critical_issues"]:
            print(f"   ❌ {issue}")
    
    # Save detailed results
    with open("enhanced_validation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: enhanced_validation_results.json")
    print("="*80)
    
    return results


if __name__ == "__main__":
    asyncio.run(main())