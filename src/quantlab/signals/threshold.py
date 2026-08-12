"""Threshold-based trading intentions."""

from __future__ import annotations

import pandas as pd


def threshold_signal(values: pd.DataFrame, *, long_below: float, short_above: float) -> pd.DataFrame:
    """Return +1 below the long threshold and -1 above the short threshold."""

    if long_below >= short_above:
        raise ValueError("long_below must be less than short_above")
    signal = pd.DataFrame(0.0, index=values.index, columns=values.columns)
    signal = signal.mask(values < long_below, 1.0)
    signal = signal.mask(values > short_above, -1.0)
    return signal.where(values.notna())
