"""SOMA context engine plugin.

Exports SomaEngine — a ContextEngine subclass that shrinks oversized tool
results extractively (vendored SOMA core, MIT) before each provider call.
Selected via config.yaml: context.engine: "soma".
"""
from .engine import SomaEngine

__all__ = ["SomaEngine"]


def register(ctx):
    """Plugin entry point (plugins/plugin_loader.py contract).

    Registers one context engine instance with the host; the host captures it
    via the collector's register_context_engine() and clones it per agent.
    """
    ctx.register_context_engine(SomaEngine())
