
"""Lightweight Pydantic replacement for basic validation."""

import json
from typing import Dict, Any, Optional, Union, get_type_hints


class Field:
    """Lightweight field descriptor."""
    
    def __init__(self, default=None, ge=None, le=None, description="", **kwargs):
        self.default = default
        self.ge = ge
        self.le = le
        self.description = description
        self.kwargs = kwargs


def validator(field_name: str, pre: bool = False, always: bool = False):
    """Lightweight validator decorator."""
    def decorator(func):
        func._validator_field = field_name
        func._validator_pre = pre
        func._validator_always = always
        return func
    return decorator


class BaseModel:
    """Lightweight BaseModel replacement."""
    
    def __init__(self, **data):
        # Set defaults from field definitions
        for name, field in self._get_fields().items():
            if hasattr(field, 'default') and field.default is not None:
                setattr(self, name, field.default)
        
        # Set provided data
        for key, value in data.items():
            setattr(self, key, value)
        
        # Run validators
        self._run_validators()
    
    @classmethod
    def _get_fields(cls) -> Dict[str, Field]:
        """Get field definitions from class annotations."""
        fields = {}
        for name, annotation in getattr(cls, '__annotations__', {}).items():
            default_value = getattr(cls, name, Field())
            if isinstance(default_value, Field):
                fields[name] = default_value
        return fields
    
    def _run_validators(self):
        """Run validation methods."""
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if callable(attr) and hasattr(attr, '_validator_field'):
                field_value = getattr(self, attr._validator_field, None)
                if field_value is not None:
                    validated_value = attr(field_value)
                    setattr(self, attr._validator_field, validated_value)
    
    def dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {}
        for name in self._get_fields():
            if hasattr(self, name):
                result[name] = getattr(self, name)
        return result
    
    @classmethod
    def from_preset(cls, preset: str):
        """Create instance from preset (placeholder)."""
        return cls()
