"""Decision tree sub-package: models, preprocessing, and fitting."""

from __future__ import annotations

from dfkit.decision_tree.fitting import build_decision_tree_result
from dfkit.decision_tree.models import DecisionTreeResult

__all__ = [
    "DecisionTreeResult",
    "build_decision_tree_result",
]
