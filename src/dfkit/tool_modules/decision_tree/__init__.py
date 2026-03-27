"""Decision tree sub-package: models, preprocessing, fitting, and toolkit integration."""

from __future__ import annotations

from dfkit.tool_modules.decision_tree.fitting import analyze_with_decision_tree
from dfkit.tool_modules.decision_tree.models import DecisionTreeResult
from dfkit.tool_modules.decision_tree.tool_module import DecisionTreeModule

__all__ = [
    "DecisionTreeModule",
    "DecisionTreeResult",
    "analyze_with_decision_tree",
]
