"""lindex: learned index structures for sorted, in-memory data."""

from .base import Index
from .binary_search import BinarySearch

__version__ = "0.1.0"

__all__ = ["Index", "BinarySearch"]
