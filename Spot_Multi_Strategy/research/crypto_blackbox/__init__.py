"""Crypto-only, point-in-time observation contracts.

Labels in this package are descriptive research observations, never orders or
evidence that a signal earns a positive return.
"""

from .schemas import CryptoObservation, ObservationClassification
from .signal_intake import SignalIntake, classify_observation

__all__ = (
    "CryptoObservation",
    "ObservationClassification",
    "SignalIntake",
    "classify_observation",
)
