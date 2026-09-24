"""SOMA context engine plugin.

Exports SomaEngine — a ContextEngine subclass that shrinks oversized tool
results extractively (vendored SOMA core, MIT) before each provider call.
Selected via config.yaml: context.engine: "soma".
"""
from .engine import SomaEngine

__all__ = ["SomaEngine"]
