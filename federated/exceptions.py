"""
Custom exception types used in the federated IDS project.
"""


class DataLoadingError(Exception):
    """Raised when there is a problem loading CSV data files."""


class DataValidationError(Exception):
    """Raised when the dataset is missing required columns or is invalid."""

