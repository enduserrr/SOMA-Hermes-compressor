"""Plugin-system contract tests: register(ctx) + clone_for_agent().

The host loads plugins as packages and calls register(collector) if present
(plugins/plugin_loader.py); a context engine is captured via the collector's
register_context_engine(engine). Separately, the host clones plugin-registered
engines per agent via clone_for_agent() (agent/context_engine.py contract) —
a fresh SomaEngine is the correct, cheap clone.
"""
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)


def _load_plugin_package():
    """Load the plugin's __init__.py as a real package so its relative
    `from .engine import SomaEngine` resolves (mirrors plugins/plugin_loader.py)."""
    pkg_name = "soma_plugin_under_test"
    if pkg_name in sys.modules:
        return sys.modules[pkg_name]
    spec = importlib.util.spec_from_file_location(
        pkg_name, os.path.join(_PKG_ROOT, "__init__.py"),
        submodule_search_locations=[_PKG_ROOT],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = mod
    spec.loader.exec_module(mod)
    return mod


class RecordingCtx:
    def __init__(self):
        self.engines = []

    def register_context_engine(self, engine):
        self.engines.append(engine)


def test_register_registers_one_engine_named_soma():
    mod = _load_plugin_package()
    ctx = RecordingCtx()
    mod.register(ctx)
    assert len(ctx.engines) == 1
    assert type(ctx.engines[0]).__name__ == "SomaEngine"
