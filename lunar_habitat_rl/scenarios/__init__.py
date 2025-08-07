"""
Scenarios module for Lunar Habitat RL Suite.

This module provides advanced scenario generation capabilities including:
- Procedural mission scenario creation
- Emergency event simulation
- Multi-habitat coordination scenarios
- Realistic failure mode modeling
- Adversarial scenario generation
- Real-time scenario adaptation
"""

from .advanced_scenario_generator import (
    AdvancedScenarioGenerator,
    ScenarioConfig,
    ScenarioEvent,
    EventTemplate,
    EventSeverity,
    EventCategory,
    CascadingFailureModel,
    AdversarialScenarioGenerator,
    create_advanced_scenario_generator
)

__all__ = [
    'AdvancedScenarioGenerator',
    'ScenarioConfig',
    'ScenarioEvent', 
    'EventTemplate',
    'EventSeverity',
    'EventCategory',
    'CascadingFailureModel',
    'AdversarialScenarioGenerator',
    'create_advanced_scenario_generator'
]