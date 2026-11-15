"""fastblur/schema_validator.py — Self-contained JSON Schema validator

Validates privacy data sheet against schema definitions without requiring
external dependencies like jsonschema.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def validate_against_schema(instance: dict[str, Any], schema: dict[str, Any], path_prefix: str = "") -> list[str]:
    """Validates an instance dictionary against a schema object."""
    errors: list[str] = []

    # 1. Required properties
    for req in schema.get("required", []):
        if req not in instance:
            prop_name = f"{path_prefix}.{req}" if path_prefix else req
            errors.append(f"Missing required property: '{prop_name}'")

    # 2. Check each defined property
    props = schema.get("properties", {})
    for prop, spec in props.items():
        if prop not in instance:
            continue
        val = instance[prop]
        prop_path = f"{path_prefix}.{prop}" if path_prefix else prop

        # Enum check
        if "enum" in spec:
            if val not in spec["enum"]:
                errors.append(f"Value '{val}' for property '{prop_path}' is not in allowed enum: {spec['enum']}")

        # Const check
        if "const" in spec:
            if val != spec["const"]:
                errors.append(f"Value '{val}' for property '{prop_path}' does not match const '{spec['const']}'")

        # Type check
        expected_type = spec.get("type")
        if expected_type == "object":
            if not isinstance(val, dict):
                errors.append(f"Property '{prop_path}' expected object, got {type(val).__name__}")
            else:
                # Recurse
                errors.extend(validate_against_schema(val, spec, path_prefix=prop_path))
        elif expected_type == "array":
            if not isinstance(val, list):
                errors.append(f"Property '{prop_path}' expected array, got {type(val).__name__}")
        elif expected_type == "integer":
            if not isinstance(val, int) or isinstance(val, bool):
                errors.append(f"Property '{prop_path}' expected integer, got {type(val).__name__}")
        elif expected_type == "number":
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                errors.append(f"Property '{prop_path}' expected number, got {type(val).__name__}")
        elif expected_type == "string":
            if not isinstance(val, str):
                errors.append(f"Property '{prop_path}' expected string, got {type(val).__name__}")

    return errors
