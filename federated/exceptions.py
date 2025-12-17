"""
Custom exceptions for data loading and validation issues.
"""


class DataLoadingError(Exception):
    """Raised when there is a problem loading CSV data files."""


class DataValidationError(Exception):
    """Raised when the dataset is missing required columns or is invalid."""

