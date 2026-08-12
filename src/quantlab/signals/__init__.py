"""Transform observations or predictions into trading intentions."""

from quantlab.signals.ranking import ranking_signal
from quantlab.signals.threshold import threshold_signal

__all__ = ["ranking_signal", "threshold_signal"]
