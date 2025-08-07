"""
Research module for Lunar Habitat RL Suite.

This module contains research tools, comparative studies, and academic
publication frameworks for evaluating novel RL algorithms.
"""

from .comparative_study import (
    ComparativeStudyRunner,
    ComparativeStudyConfig,
    AlgorithmWrapper,
    run_comparative_study
)

__all__ = [
    'ComparativeStudyRunner',
    'ComparativeStudyConfig',
    'AlgorithmWrapper',
    'run_comparative_study'
]