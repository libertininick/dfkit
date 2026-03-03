"""Decision tree sub-package: models, preprocessing, and fitting."""

from __future__ import annotations

from dfkit.decision_tree.fitting import analyze_with_decision_tree
from dfkit.decision_tree.models import DecisionTreeResult

__all__ = [
    "DecisionTreeResult",
    "analyze_with_decision_tree",
]
