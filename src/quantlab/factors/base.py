"""Shared factor input checks kept deliberately small and explicit."""

from __future__ import annotations

import pandas as pd


def require_same_panel(left: pd.DataFrame, right: pd.DataFrame) -> None:
    """Reject implicit pandas alignment when two research panels differ."""

    if not left.index.equals(right.index) or not left.columns.equals(right.columns):
        raise ValueError("factor and return panels must have identical index and columns")
