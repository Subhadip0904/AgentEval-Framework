"""
Tool registry - all composable tools in one place.
The agent loads tools by name from config.yaml["tools"]["enabled"].
"""
from tools.code_explainer import code_explainer
from tools.log_classifier import log_classifier
from tools.spec_search import spec_search

TOOL_REGISTRY: dict = {
    "spec_search": spec_search,
    "log_classifier": log_classifier,
    "code_explainer": code_explainer,
}


def get_tools(names: list[str]) -> list:
    """Return tool objects for a list of tool names, skipping unknown names."""
    missing = [name for name in names if name not in TOOL_REGISTRY]
    if missing:
        print(f"[tool_registry] Warning: unknown tools requested: {missing}")
    return [TOOL_REGISTRY[name] for name in names if name in TOOL_REGISTRY]
