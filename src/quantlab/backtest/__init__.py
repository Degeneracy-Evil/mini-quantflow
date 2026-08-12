"""A small close-to-next-close vectorized backtest with explicit semantics."""

from quantlab.backtest.costs import TransactionCostModel
from quantlab.backtest.engine import run_backtest
from quantlab.backtest.result import BacktestResult

__all__ = ["BacktestResult", "TransactionCostModel", "run_backtest"]
