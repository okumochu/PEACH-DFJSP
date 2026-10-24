"""
Data Generation Module for Multi-Criteria FJSP

This module provides data generation capabilities for FJSP training,
supporting both static and dynamic multi-criteria environments.

Data types:
- 'static': Multi-criteria FJSP with weights and due dates (no dynamic events)
- 'dynamic': Multi-criteria FJSP with dynamic events (job arrivals, breakdowns, etc.)
"""

from .data_generator import DataGenerator, create_data_generator

__all__ = [
    'DataGenerator',
    'create_data_generator',
]
