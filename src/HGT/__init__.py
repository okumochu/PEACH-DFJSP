"""
HGT: Heterogeneous Graph Transformer for Flexible Job Shop Scheduling

This module contains the implementation of HGT-based approaches for solving
Flexible Job Shop Scheduling Problems (FJSP) using heterogeneous graph
representations and graph neural networks.

The implementation follows DANIEL's structure and supports both single and
parallel environment processing with automatic padding for different operation
numbers.

Components:
- hgt_env.py: Unified FJSP environment implementation (single + parallel)
- hgt_network.py: HGT-based network implementation
- hgt_trainer.py: Training and evaluation utilities

Key Features:
- Parallel environment processing (multiple FJSP instances simultaneously)
- Automatic padding for different operation numbers
- Heterogeneous graph construction for HGT processing
- DANIEL-compatible interface and data structures
- Batch processing for efficient training
"""

from .hgt_env import FJSPEnvHGT, HGTEnvState
from .hgt_env_no_pt import FJSPEnvHGTNoPT, HGTEnvStateNoPT
from .hgt_network import HGT, HGTFeatureExtractor, HGTFeatureExtractorNoPT
from .hgt_trainer import HGTTrainer, PPO, Memory

__all__ = [
    'FJSPEnvHGT',
    'HGTEnvState',
    'FJSPEnvHGTNoPT',
    'HGTEnvStateNoPT',
    'HGT',
    'HGTFeatureExtractor',
    'HGTFeatureExtractorNoPT',
    'HGTTrainer',
    'PPO',
    'Memory'
]
