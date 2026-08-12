"""Cross-sectional ranking signals."""

from __future__ import annotations

import pandas as pd


def ranking_signal(scores: pd.DataFrame, quantile: float = 0.2) -> pd.DataFrame:
    """Return +1 for the top fraction, -1 for the bottom, and 0 otherwise."""

    if not 0.0 < quantile <= 0.5:
        raise ValueError("quantile must be in (0, 0.5]")
    ranks = scores.rank(axis=1, pct=True, method="average")
    signal = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    signal = signal.mask(ranks > 1.0 - quantile, 1.0)
    signal = signal.mask(ranks <= quantile, -1.0)
    return signal.where(scores.notna())
