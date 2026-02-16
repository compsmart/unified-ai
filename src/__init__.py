"""
Unified AI - The 2026 AI Architecture

A complete AI system that:
- Runs on edge CPUs (no GPU required)
- Responds in microseconds
- Cannot hallucinate (grounded symbolic memory)
- Learns online from experience

Components:
- HippocampusVSA: Clean-up memory with hypervectors
- PredictiveLattice: Fast pattern detection
- RobustWaveToHVBridge: Continuous-to-symbolic translation
- UnifiedAI: Integration layer
- GroundedLLM: Natural language formatting
"""

from .hippocampus_vsa import HippocampusVSA, FactMemory
from .predictive_lattice import PredictiveLattice, LatticeConfig, AnomalyDetector
from .wave_to_hv_bridge import RobustWaveToHVBridge, BridgeConfig, SemanticBridge
from .unified_ai import UnifiedAI, UnifiedAIConfig
from .llm_interface import GroundedLLM, LLMConfig, ConversationalInterface

__version__ = "0.1.0"
__all__ = [
    'HippocampusVSA',
    'FactMemory',
    'PredictiveLattice',
    'LatticeConfig',
    'AnomalyDetector',
    'RobustWaveToHVBridge',
    'BridgeConfig',
    'SemanticBridge',
    'UnifiedAI',
    'UnifiedAIConfig',
    'GroundedLLM',
    'LLMConfig',
    'ConversationalInterface',
]
